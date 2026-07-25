#include "core/frame_extractor.hpp"
#include "core/media_util.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
}

namespace vlm {

namespace {

void logFfmpegError(int err, const char* msg)
{
    char buf[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(err, buf, sizeof(buf));
    std::cerr << msg << ": " << buf << '\n';
}

enum AVPixelFormat getHwFormat(AVCodecContext* /*ctx*/, const enum AVPixelFormat* pix_fmts)
{
    for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_DRM_PRIME) {
            return AV_PIX_FMT_DRM_PRIME;
        }
    }
    return AV_PIX_FMT_NONE;
}

struct AvPacketDeleter {
    void operator()(AVPacket* packet) const noexcept { av_packet_free(&packet); }
};

struct AvFrameDeleter {
    void operator()(AVFrame* frame) const noexcept { av_frame_free(&frame); }
};

using PacketPtr = std::unique_ptr<AVPacket, AvPacketDeleter>;
using FramePtr = std::unique_ptr<AVFrame, AvFrameDeleter>;

[[nodiscard]] PacketPtr makePacket() { return PacketPtr(av_packet_alloc()); }

[[nodiscard]] FramePtr makeFrame() { return FramePtr(av_frame_alloc()); }

const char* rkmppDecoderName(AVCodecID id)
{
    switch (id) {
        case AV_CODEC_ID_H264:
            return "h264_rkmpp";
        case AV_CODEC_ID_HEVC:
            return "hevc_rkmpp";
        case AV_CODEC_ID_MPEG4:
            return "mpeg4_rkmpp";
        case AV_CODEC_ID_H263:
            return "h263_rkmpp";
        case AV_CODEC_ID_VP8:
            return "vp8_rkmpp";
        case AV_CODEC_ID_VP9:
            return "vp9_rkmpp";
        case AV_CODEC_ID_AV1:
            return "av1_rkmpp";
        case AV_CODEC_ID_MJPEG:
            return "mjpeg_rkmpp";
        default:
            return nullptr;
    }
}

[[nodiscard]] bool fillVideoInfo(const AVFormatContext* fmt_ctx, int video_stream_idx,
                                 VideoInfo& info)
{
    if (video_stream_idx < 0) {
        return false;
    }
    const AVStream* stream = fmt_ctx->streams[video_stream_idx];
    const AVCodecParameters* par = stream->codecpar;
    if (par->width <= 0 || par->height <= 0) {
        return false;
    }

    info.width = par->width;
    info.height = par->height;

    AVRational frame_rate = stream->avg_frame_rate;
    if (frame_rate.num <= 0 || frame_rate.den <= 0) {
        frame_rate = stream->r_frame_rate;
    }
    if (frame_rate.num > 0 && frame_rate.den > 0) {
        info.fps = av_q2d(frame_rate);
    } else {
        info.fps = 25.0;
    }

    if (stream->duration > 0) {
        info.duration_sec = stream->duration * av_q2d(stream->time_base);
    } else if (fmt_ctx->duration > 0) {
        info.duration_sec = static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE;
    } else {
        info.duration_sec = 0.0;
    }

    return info.width > 0 && info.height > 0 && info.fps > 0 && info.duration_sec > 0;
}

class LibavFrameSession {
public:
    LibavFrameSession() = default;
    ~LibavFrameSession() { close(); }

    LibavFrameSession(const LibavFrameSession&) = delete;
    LibavFrameSession& operator=(const LibavFrameSession&) = delete;

    [[nodiscard]] bool open(std::string_view path, bool use_hw, int target_w, int target_h)
    {
        close();
        target_w_ = target_w;
        target_h_ = target_h;
        use_hw_ = use_hw;

        if (avformat_open_input(&fmt_ctx_, std::string(path).c_str(), nullptr, nullptr) < 0) {
            std::cerr << "libav: avformat_open_input failed for " << path << '\n';
            return false;
        }
        if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
            std::cerr << "libav: avformat_find_stream_info failed\n";
            close();
            return false;
        }

        video_stream_idx_ =
            av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        if (video_stream_idx_ < 0) {
            std::cerr << "libav: no video stream\n";
            close();
            return false;
        }

        const AVCodecParameters* par = fmt_ctx_->streams[video_stream_idx_]->codecpar;
        const AVCodec* decoder = nullptr;
        bool use_rkmpp_decoder = false;
        if (use_hw_) {
            if (const char* hw_name = rkmppDecoderName(par->codec_id)) {
                decoder = avcodec_find_decoder_by_name(hw_name);
                use_rkmpp_decoder = decoder != nullptr;
            }
        }
        if (decoder == nullptr) {
            decoder = avcodec_find_decoder(par->codec_id);
            use_hw_ = false;
        }

        codec_ctx_ = avcodec_alloc_context3(decoder);
        if (codec_ctx_ == nullptr) {
            close();
            return false;
        }
        if (avcodec_parameters_to_context(codec_ctx_, par) < 0) {
            close();
            return false;
        }

        if (use_rkmpp_decoder) {
            // Stream codecpar often sets yuv420p; rkmpp needs NONE/DRM_PRIME for HW surfaces.
            codec_ctx_->pix_fmt = AV_PIX_FMT_NONE;
            codec_ctx_->get_format = getHwFormat;
            if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_RKMPP, nullptr, nullptr,
                                       0) == 0) {
                codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
            }
        }

        if (avcodec_open2(codec_ctx_, decoder, nullptr) < 0) {
            std::cerr << "libav: avcodec_open2 failed\n";
            close();
            return false;
        }

        return true;
    }

    [[nodiscard]] bool probe(VideoInfo& info) const
    {
        return fillVideoInfo(fmt_ctx_, video_stream_idx_, info);
    }

    [[nodiscard]] int extractAtTimes(const std::vector<double>& times_sec,
                                     const std::function<bool(RgbFrame&, int index)>& on_frame)
    {
        if (fmt_ctx_ == nullptr || codec_ctx_ == nullptr || times_sec.empty() || !on_frame) {
            return 0;
        }

        if (av_seek_frame(fmt_ctx_, -1, 0, AVSEEK_FLAG_BACKWARD) < 0) {
            std::cerr << "libav: seek to start failed\n";
            return 0;
        }
        avformat_flush(fmt_ctx_);
        avcodec_flush_buffers(codec_ctx_);
        resetFilterGraph();

        AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
        std::vector<int64_t> target_pts;
        target_pts.reserve(times_sec.size());
        for (const double time_sec : times_sec) {
            target_pts.push_back(av_rescale_q(static_cast<int64_t>(time_sec * AV_TIME_BASE),
                                             AV_TIME_BASE_Q, stream->time_base));
        }

        PacketPtr packet = makePacket();
        FramePtr frame = makeFrame();
        FramePtr filtered = makeFrame();
        if (!packet || !frame || !filtered) {
            return 0;
        }

        int next = 0;
        int extracted = 0;
        bool done = false;
        bool have_packet = false;

        auto processDecodedFrame = [&]() {
            int64_t frame_pts = frame->best_effort_timestamp;
            if (frame_pts == AV_NOPTS_VALUE) {
                frame_pts = frame->pts;
            }
            if (frame_pts == AV_NOPTS_VALUE ||
                frame_pts < target_pts[static_cast<std::size_t>(next)]) {
                av_frame_unref(frame.get());
                return;
            }

            RgbFrame out_rgb;
            if (frameToRgb(frame.get(), filtered.get(), out_rgb)) {
                if (on_frame(out_rgb, next)) {
                    ++extracted;
                }
            }
            ++next;
            if (next >= static_cast<int>(times_sec.size())) {
                done = true;
            }
            av_frame_unref(frame.get());
        };

        while (!done) {
            if (!have_packet) {
                if (av_read_frame(fmt_ctx_, packet.get()) < 0) {
                    break;
                }
                if (packet->stream_index != video_stream_idx_) {
                    av_packet_unref(packet.get());
                    continue;
                }
                have_packet = true;
            }

            const int send_ret = avcodec_send_packet(codec_ctx_, packet.get());
            if (send_ret == AVERROR(EAGAIN)) {
                const int recv_ret = avcodec_receive_frame(codec_ctx_, frame.get());
                if (recv_ret == 0) {
                    processDecodedFrame();
                    continue;
                }
                if (recv_ret == AVERROR(EAGAIN)) {
                    std::cerr << "libav: decoder returned EAGAIN on both send and receive\n";
                    break;
                }
                if (recv_ret == AVERROR_EOF) {
                    break;
                }
                logFfmpegError(recv_ret, "libav: avcodec_receive_frame");
                break;
            }
            if (send_ret < 0) {
                logFfmpegError(send_ret, "libav: avcodec_send_packet");
                av_packet_unref(packet.get());
                have_packet = false;
                continue;
            }

            av_packet_unref(packet.get());
            have_packet = false;

            while (!done) {
                const int recv_ret = avcodec_receive_frame(codec_ctx_, frame.get());
                if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
                    break;
                }
                if (recv_ret < 0) {
                    logFfmpegError(recv_ret, "libav: avcodec_receive_frame");
                    break;
                }
                processDecodedFrame();
            }
        }

        return extracted;
    }

    [[nodiscard]] bool extractAtTime(double time_sec, RgbFrame& out_rgb)
    {
        bool got = false;
        const int extracted = extractAtTimes(
            {time_sec}, [&](RgbFrame& frame, int /*index*/) {
                out_rgb = std::move(frame);
                got = true;
                return true;
            });
        return got && extracted == 1;
    }

    void close()
    {
        resetFilterGraph();

        if (codec_ctx_ != nullptr) {
            avcodec_free_context(&codec_ctx_);
            codec_ctx_ = nullptr;
        }
        if (fmt_ctx_ != nullptr) {
            avformat_close_input(&fmt_ctx_);
            fmt_ctx_ = nullptr;
        }
        av_buffer_unref(&hw_device_ctx_);
        video_stream_idx_ = -1;
        use_hw_ = false;
    }

private:
    void resetFilterGraph()
    {
        if (filter_graph_ != nullptr) {
            avfilter_graph_free(&filter_graph_);
            filter_graph_ = nullptr;
        }
        buffersrc_ctx_ = nullptr;
        buffersink_ctx_ = nullptr;
    }

    [[nodiscard]] bool initFilterGraph(const AVFrame* ref_frame)
    {
        filter_graph_ = avfilter_graph_alloc();
        if (filter_graph_ == nullptr) {
            return false;
        }

        const auto fail = [this]() {
            resetFilterGraph();
            return false;
        };

        const AVFilter* buffersrc = avfilter_get_by_name("buffer");
        const AVFilter* buffersink = avfilter_get_by_name("buffersink");
        if (buffersrc == nullptr || buffersink == nullptr) {
            return fail();
        }

        AVStream* stream = fmt_ctx_->streams[video_stream_idx_];
        AVRational tb = stream->time_base;
        if (tb.num <= 0 || tb.den <= 0) {
            tb = AVRational{1, 25};
        }
        AVRational sar = codec_ctx_->sample_aspect_ratio;
        if (sar.num <= 0 || sar.den <= 0) {
            sar = AVRational{1, 1};
        }

        const int src_w = ref_frame->width > 0 ? ref_frame->width : codec_ctx_->width;
        const int src_h = ref_frame->height > 0 ? ref_frame->height : codec_ctx_->height;
        AVBufferRef* hw_frames = ref_frame->hw_frames_ctx;
        if (hw_frames == nullptr && codec_ctx_->hw_frames_ctx != nullptr) {
            hw_frames = codec_ctx_->hw_frames_ctx;
        }
        const bool hw_drm =
            use_hw_ && ref_frame->format == AV_PIX_FMT_DRM_PRIME && hw_frames != nullptr;
        const int src_fmt =
            hw_drm ? AV_PIX_FMT_DRM_PRIME
                   : (ref_frame->format != AV_PIX_FMT_NONE ? ref_frame->format
                                                           : codec_ctx_->pix_fmt);

        int buffer_pix_fmt = src_fmt;
        if (hw_drm) {
            const auto* frames_ctx =
                reinterpret_cast<const AVHWFramesContext*>(hw_frames->data);
            buffer_pix_fmt = frames_ctx->sw_format;
        }

        char args[512];
        std::snprintf(args, sizeof(args),
                      "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d", src_w,
                      src_h, buffer_pix_fmt, tb.num, tb.den, sar.num, sar.den);

        if (avfilter_graph_create_filter(&buffersrc_ctx_, buffersrc, "in", args, nullptr,
                                         filter_graph_) < 0) {
            std::cerr << "libav: buffersrc create failed\n";
            return fail();
        }

        if (hw_drm) {
            AVBufferSrcParameters* par = av_buffersrc_parameters_alloc();
            if (par == nullptr) {
                return fail();
            }
            par->format = AV_PIX_FMT_DRM_PRIME;
            par->hw_frames_ctx = av_buffer_ref(hw_frames);
            par->width = src_w;
            par->height = src_h;
            par->sample_aspect_ratio = sar;
            if (ref_frame->color_range != AVCOL_RANGE_UNSPECIFIED) {
                par->color_range = ref_frame->color_range;
            }
            if (ref_frame->colorspace != AVCOL_SPC_UNSPECIFIED) {
                par->color_space = ref_frame->colorspace;
            }
            if (par->hw_frames_ctx == nullptr ||
                av_buffersrc_parameters_set(buffersrc_ctx_, par) < 0) {
                av_freep(&par);
                return fail();
            }
            av_freep(&par);
        }

        if (avfilter_graph_create_filter(&buffersink_ctx_, buffersink, "out", nullptr, nullptr,
                                         filter_graph_) < 0) {
            std::cerr << "libav: buffersink create failed\n";
            return fail();
        }

        std::ostringstream vf;
        if (hw_drm) {
            vf << "scale_rkrga=w=" << target_w_ << ":h=" << target_h_
               << ":format=rgb24:force_original_aspect_ratio=decrease,"
               << "hwdownload,format=rgb24,"
               << "pad=" << target_w_ << ":" << target_h_
               << ":(ow-iw)/2:(oh-ih)/2:color=0x7F7F7F";
        } else {
            vf << "scale=" << target_w_ << ":" << target_h_
               << ":force_original_aspect_ratio=decrease,"
               << "pad=" << target_w_ << ":" << target_h_
               << ":(ow-iw)/2:(oh-ih)/2:color=0x7F7F7F,format=rgb24";
        }

        AVFilterInOut* outputs = avfilter_inout_alloc();
        AVFilterInOut* inputs = avfilter_inout_alloc();
        if (outputs == nullptr || inputs == nullptr) {
            avfilter_inout_free(&inputs);
            avfilter_inout_free(&outputs);
            return fail();
        }

        outputs->name = av_strdup("in");
        outputs->filter_ctx = buffersrc_ctx_;
        outputs->pad_idx = 0;
        outputs->next = nullptr;

        inputs->name = av_strdup("out");
        inputs->filter_ctx = buffersink_ctx_;
        inputs->pad_idx = 0;
        inputs->next = nullptr;

        const int ret =
            avfilter_graph_parse_ptr(filter_graph_, vf.str().c_str(), &inputs, &outputs, nullptr);
        avfilter_inout_free(&inputs);
        avfilter_inout_free(&outputs);
        if (ret < 0) {
            logFfmpegError(ret, "libav: filter graph parse");
            return fail();
        }
        if (avfilter_graph_config(filter_graph_, nullptr) < 0) {
            std::cerr << "libav: filter graph config failed\n";
            return fail();
        }
        return true;
    }

    [[nodiscard]] bool frameToRgb(AVFrame* frame, AVFrame* filtered, RgbFrame& out_rgb)
    {
        resetFilterGraph();
        if (!initFilterGraph(frame)) {
            resetFilterGraph();
            return false;
        }

        bool ok = false;
        if (av_buffersrc_add_frame_flags(buffersrc_ctx_, frame, AV_BUFFERSRC_FLAG_KEEP_REF) < 0) {
            std::cerr << "libav: buffersrc add frame failed\n";
            resetFilterGraph();
            return false;
        }
        if (av_buffersrc_add_frame_flags(buffersrc_ctx_, nullptr, 0) < 0) {
            std::cerr << "libav: buffersrc EOF failed\n";
            resetFilterGraph();
            return false;
        }

        while (true) {
            const int ret = av_buffersink_get_frame(buffersink_ctx_, filtered);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                break;
            }
            if (ret < 0) {
                logFfmpegError(ret, "libav: buffersink get frame");
                resetFilterGraph();
                return false;
            }

            if (filtered->width != target_w_ || filtered->height != target_h_ ||
                filtered->format != AV_PIX_FMT_RGB24) {
                std::cerr << "libav: unexpected filter output " << filtered->width << 'x'
                          << filtered->height << " fmt=" << filtered->format << '\n';
                av_frame_unref(filtered);
                continue;
            }

            const auto frame_bytes = static_cast<std::size_t>(target_w_) *
                                     static_cast<std::size_t>(target_h_) * 3;
            std::vector<std::uint8_t> pixels(frame_bytes);
            if (av_image_copy_to_buffer(pixels.data(), static_cast<int>(frame_bytes),
                                        filtered->data, filtered->linesize, AV_PIX_FMT_RGB24,
                                        target_w_, target_h_, 1) < 0) {
                av_frame_unref(filtered);
                resetFilterGraph();
                return false;
            }
            av_frame_unref(filtered);
            out_rgb = RgbFrame::fromRaw(target_w_, target_h_, std::move(pixels));
            ok = true;
            break;
        }

        resetFilterGraph();
        if (!ok) {
            std::cerr << "libav: no RGB frame from filter graph\n";
        }
        return ok;
    }

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    AVFilterGraph* filter_graph_ = nullptr;
    AVFilterContext* buffersrc_ctx_ = nullptr;
    AVFilterContext* buffersink_ctx_ = nullptr;
    int video_stream_idx_ = -1;
    int target_w_ = 0;
    int target_h_ = 0;
    bool use_hw_ = false;
};

[[nodiscard]] bool probeWithLibav(std::string_view filename, VideoInfo& info)
{
    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, std::string(filename).c_str(), nullptr, nullptr) < 0) {
        return false;
    }
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        return false;
    }
    const int video_idx =
        av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    const bool ok = fillVideoInfo(fmt_ctx, video_idx, info);
    avformat_close_input(&fmt_ctx);
    return ok;
}

}  // namespace

int planFrameCount(double duration_sec, double fps, int requested_frames)
{
    requested_frames = std::max(1, requested_frames);
    if (duration_sec <= 0.0 || fps <= 0.0) {
        return 0;
    }
    const int total_frames =
        std::max(1, static_cast<int>(std::lround(duration_sec * fps)));
    return std::min(requested_frames, total_frames);
}

bool FrameExtractor::probe(std::string_view filename, VideoInfo& info) const
{
    if (!std::filesystem::exists(filename)) {
        return false;
    }
    return probeWithLibav(filename, info);
}

bool FrameExtractor::extractFrameAtTime(std::string_view filename, double time_sec, int target_w,
                                        int target_h, RgbFrame& out_rgb) const
{
    if (target_w <= 0 || target_h <= 0 || !std::filesystem::exists(filename)) {
        return false;
    }

    LibavFrameSession session;
    const bool use_hw = !isGifPath(filename);
    if (!session.open(filename, use_hw, target_w, target_h)) {
        return false;
    }
    return session.extractAtTime(time_sec, out_rgb);
}

std::vector<RgbFrame> FrameExtractor::extractFrames(std::string_view filename, int requested_frames,
                                                   int target_w, int target_h,
                                                   FrameProgressCallback progress,
                                                   VideoInfo* out_info) const
{
    std::vector<RgbFrame> frames;
    const int got = extractFramesStreaming(
        filename, requested_frames,
        [&](RgbFrame frame, int /*index*/, int /*total*/) { frames.push_back(std::move(frame)); },
        target_w, target_h, progress, out_info);
    (void)got;
    return frames;
}

int FrameExtractor::extractFramesStreaming(std::string_view filename, int requested_frames,
                                           FrameReadyCallback on_frame, int target_w, int target_h,
                                           FrameProgressCallback progress,
                                           VideoInfo* out_info) const
{
    if (!on_frame || !std::filesystem::exists(filename)) {
        return 0;
    }

    LibavFrameSession session;
    const bool use_hw = !isGifPath(filename);
    if (!session.open(filename, use_hw, target_w, target_h)) {
        return 0;
    }

    VideoInfo info;
    if (!session.probe(info)) {
        return 0;
    }
    if (out_info) {
        *out_info = info;
    }

    const int target_count = planFrameCount(info.duration_sec, info.fps, requested_frames);
    if (target_count <= 0) {
        return 0;
    }

    const double total_frames = info.duration_sec * info.fps;
    const double step = total_frames / static_cast<double>(target_count);
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(target_count));
    for (int i = 0; i < target_count; ++i) {
        const int frame_idx = static_cast<int>(i * step);
        times.push_back(frame_idx / info.fps);
    }

    const int extracted = session.extractAtTimes(times, [&](RgbFrame& frame, int index) {
        if (progress) {
            progress(index, target_count);
        }
        on_frame(std::move(frame), index, target_count);
        return true;
    });

    if (progress) {
        progress(target_count, target_count);
    }
    return extracted;
}

}  // namespace vlm
