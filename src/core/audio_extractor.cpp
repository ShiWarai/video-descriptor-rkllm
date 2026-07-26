#include "core/audio_extractor.hpp"

#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace vlm {

namespace {

constexpr int kTargetSampleRate = 16000;
constexpr int kTargetChannels = 1;
constexpr AVSampleFormat kTargetSampleFmt = AV_SAMPLE_FMT_S16;

void logFfmpegError(int err, const char* msg)
{
    char buf[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(err, buf, sizeof(buf));
    std::cerr << msg << ": " << buf << '\n';
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

bool appendResampledFrame(SwrContext* swr_ctx, AVFrame* frame, std::vector<uint8_t>& pcm)
{
    const int out_samples = swr_get_out_samples(swr_ctx, frame->nb_samples);
    if (out_samples < 0) {
        return false;
    }

    const int out_buf_size =
        av_samples_get_buffer_size(nullptr, kTargetChannels, out_samples, kTargetSampleFmt, 1);
    if (out_buf_size < 0) {
        return false;
    }

    const auto old_size = pcm.size();
    pcm.resize(old_size + static_cast<std::size_t>(out_buf_size));
    uint8_t* out_ptr = pcm.data() + old_size;

    const int converted = swr_convert(swr_ctx, &out_ptr, out_samples,
                                        const_cast<const uint8_t**>(frame->extended_data),
                                        frame->nb_samples);
    if (converted < 0) {
        logFfmpegError(converted, "audio: swr_convert");
        pcm.resize(old_size);
        return false;
    }

    const int actual_bytes =
        av_samples_get_buffer_size(nullptr, kTargetChannels, converted, kTargetSampleFmt, 1);
    if (actual_bytes < 0) {
        pcm.resize(old_size);
        return false;
    }
    pcm.resize(old_size + static_cast<std::size_t>(actual_bytes));
    return true;
}

bool flushResampler(SwrContext* swr_ctx, std::vector<uint8_t>& pcm)
{
    while (true) {
        const int out_samples = 256;
        const int out_buf_size =
            av_samples_get_buffer_size(nullptr, kTargetChannels, out_samples, kTargetSampleFmt, 1);
        if (out_buf_size < 0) {
            return false;
        }

        const auto old_size = pcm.size();
        pcm.resize(old_size + static_cast<std::size_t>(out_buf_size));
        uint8_t* out_ptr = pcm.data() + old_size;

        const int converted = swr_convert(swr_ctx, &out_ptr, out_samples, nullptr, 0);
        if (converted == 0) {
            pcm.resize(old_size);
            break;
        }
        if (converted < 0) {
            logFfmpegError(converted, "audio: swr_convert flush");
            pcm.resize(old_size);
            return false;
        }

        const int actual_bytes =
            av_samples_get_buffer_size(nullptr, kTargetChannels, converted, kTargetSampleFmt, 1);
        if (actual_bytes < 0) {
            pcm.resize(old_size);
            return false;
        }
        pcm.resize(old_size + static_cast<std::size_t>(actual_bytes));
    }
    return true;
}

void writeLe32(std::vector<uint8_t>& out, uint32_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xFF));
}

void writeLe16(std::vector<uint8_t>& out, uint16_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

void appendWavHeader(std::vector<uint8_t>& wav, uint32_t pcm_bytes)
{
    constexpr uint16_t kBitsPerSample = 16;
    constexpr uint16_t kAudioFormatPcm = 1;
    const uint32_t byte_rate =
        static_cast<uint32_t>(kTargetSampleRate * kTargetChannels * kBitsPerSample / 8);
    const uint16_t block_align =
        static_cast<uint16_t>(kTargetChannels * kBitsPerSample / 8);
    const uint32_t riff_size = 36 + pcm_bytes;

    wav.clear();
    wav.reserve(44 + pcm_bytes);
    wav.insert(wav.end(), {'R', 'I', 'F', 'F'});
    writeLe32(wav, riff_size);
    wav.insert(wav.end(), {'W', 'A', 'V', 'E'});
    wav.insert(wav.end(), {'f', 'm', 't', ' '});
    writeLe32(wav, 16);
    writeLe16(wav, kAudioFormatPcm);
    writeLe16(wav, static_cast<uint16_t>(kTargetChannels));
    writeLe32(wav, static_cast<uint32_t>(kTargetSampleRate));
    writeLe32(wav, byte_rate);
    writeLe16(wav, block_align);
    writeLe16(wav, kBitsPerSample);
    wav.insert(wav.end(), {'d', 'a', 't', 'a'});
    writeLe32(wav, pcm_bytes);
}

}  // namespace

bool extractAudioWav16kMono(const std::filesystem::path& media_path, std::vector<uint8_t>& out_wav)
{
    out_wav.clear();

    AVFormatContext* fmt_ctx = nullptr;
    const std::string path = media_path.string();
    if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "audio: avformat_open_input failed for " << path << '\n';
        return false;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "audio: avformat_find_stream_info failed\n";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    const int audio_stream_idx =
        av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audio_stream_idx < 0) {
        std::cerr << "audio: no audio stream in " << path << '\n';
        avformat_close_input(&fmt_ctx);
        return false;
    }

    const AVCodecParameters* par = fmt_ctx->streams[audio_stream_idx]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(par->codec_id);
    if (decoder == nullptr) {
        std::cerr << "audio: decoder not found\n";
        avformat_close_input(&fmt_ctx);
        return false;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(decoder);
    if (codec_ctx == nullptr) {
        avformat_close_input(&fmt_ctx);
        return false;
    }

    bool ok = false;
    SwrContext* swr_ctx = nullptr;
    std::vector<uint8_t> pcm;
    pcm.reserve(1024 * 1024);
    AVChannelLayout out_layout = AV_CHANNEL_LAYOUT_MONO;

    if (avcodec_parameters_to_context(codec_ctx, par) < 0) {
        std::cerr << "audio: avcodec_parameters_to_context failed\n";
        goto cleanup;
    }

    if (avcodec_open2(codec_ctx, decoder, nullptr) < 0) {
        std::cerr << "audio: avcodec_open2 failed\n";
        goto cleanup;
    }

    swr_ctx = swr_alloc();
    if (swr_ctx == nullptr) {
        goto cleanup;
    }

    av_opt_set_chlayout(swr_ctx, "in_chlayout", &codec_ctx->ch_layout, 0);
    av_opt_set_int(swr_ctx, "in_sample_rate", codec_ctx->sample_rate, 0);
    av_opt_set_sample_fmt(swr_ctx, "in_sample_fmt", codec_ctx->sample_fmt, 0);

    av_opt_set_chlayout(swr_ctx, "out_chlayout", &out_layout, 0);
    av_opt_set_int(swr_ctx, "out_sample_rate", kTargetSampleRate, 0);
    av_opt_set_sample_fmt(swr_ctx, "out_sample_fmt", kTargetSampleFmt, 0);

    if (swr_init(swr_ctx) < 0) {
        std::cerr << "audio: swr_init failed\n";
        goto cleanup;
    }

    {
        PacketPtr packet = makePacket();
        FramePtr frame = makeFrame();
        if (!packet || !frame) {
            goto cleanup;
        }

        while (av_read_frame(fmt_ctx, packet.get()) >= 0) {
            if (packet->stream_index != audio_stream_idx) {
                av_packet_unref(packet.get());
                continue;
            }

            const int send_ret = avcodec_send_packet(codec_ctx, packet.get());
            av_packet_unref(packet.get());
            if (send_ret < 0 && send_ret != AVERROR(EAGAIN) && send_ret != AVERROR_EOF) {
                logFfmpegError(send_ret, "audio: avcodec_send_packet");
                goto cleanup;
            }

            while (true) {
                const int recv_ret = avcodec_receive_frame(codec_ctx, frame.get());
                if (recv_ret == AVERROR(EAGAIN) || recv_ret == AVERROR_EOF) {
                    break;
                }
                if (recv_ret < 0) {
                    logFfmpegError(recv_ret, "audio: avcodec_receive_frame");
                    goto cleanup;
                }

                if (!appendResampledFrame(swr_ctx, frame.get(), pcm)) {
                    goto cleanup;
                }
                av_frame_unref(frame.get());
            }
        }

        avcodec_send_packet(codec_ctx, nullptr);
        while (true) {
            const int recv_ret = avcodec_receive_frame(codec_ctx, frame.get());
            if (recv_ret == AVERROR_EOF || recv_ret == AVERROR(EAGAIN)) {
                break;
            }
            if (recv_ret < 0) {
                logFfmpegError(recv_ret, "audio: avcodec_receive_frame flush");
                goto cleanup;
            }

            if (!appendResampledFrame(swr_ctx, frame.get(), pcm)) {
                goto cleanup;
            }
            av_frame_unref(frame.get());
        }

        if (!flushResampler(swr_ctx, pcm)) {
            goto cleanup;
        }

        if (pcm.empty()) {
            std::cerr << "audio: decoded PCM is empty\n";
            goto cleanup;
        }

        appendWavHeader(out_wav, static_cast<uint32_t>(pcm.size()));
        out_wav.insert(out_wav.end(), pcm.begin(), pcm.end());
        ok = true;
    }

cleanup:
    if (swr_ctx != nullptr) {
        swr_free(&swr_ctx);
    }
    if (codec_ctx != nullptr) {
        avcodec_free_context(&codec_ctx);
    }
    avformat_close_input(&fmt_ctx);
    if (!ok) {
        out_wav.clear();
    }
    return ok;
}

}  // namespace vlm
