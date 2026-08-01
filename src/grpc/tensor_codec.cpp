#include "grpc/tensor_codec.hpp"

#include <arpa/inet.h>
#include <cstring>
#include <netdb.h>
#include <sys/socket.h>

#include <algorithm>
#include <cstdlib>

namespace vlm::grpc_util {

namespace {

void appendLe32(std::string& out, std::uint32_t value)
{
    const char bytes[4] = {
        static_cast<char>(value & 0xFF),
        static_cast<char>((value >> 8) & 0xFF),
        static_cast<char>((value >> 16) & 0xFF),
        static_cast<char>((value >> 24) & 0xFF),
    };
    out.append(bytes, sizeof(bytes));
}

}  // namespace

vlm::v1::Tensor floatVectorToTensor(const std::vector<float>& data,
                                    const std::vector<std::int64_t>& shape)
{
    vlm::v1::Tensor tensor;
    for (const auto dim : shape) {
        tensor.add_shape(dim);
    }
    tensor.set_dtype("float32");
    if (!data.empty()) {
        tensor.set_data(data.data(), data.size() * sizeof(float));
    }
    return tensor;
}

std::vector<float> tensorToFloatVector(const vlm::v1::Tensor& tensor)
{
    const std::string& raw = tensor.data();
    if (raw.empty()) {
        return {};
    }
    const std::size_t count = raw.size() / sizeof(float);
    std::vector<float> out(count);
    std::memcpy(out.data(), raw.data(), count * sizeof(float));
    return out;
}

vlm::v1::VisionModelInfo toProto(const VisionModelInfo& info)
{
    vlm::v1::VisionModelInfo out;
    out.set_width(info.width);
    out.set_height(info.height);
    out.set_channels(info.channels);
    out.set_image_tokens(info.image_tokens);
    out.set_embed_size(info.embed_size);
    return out;
}

VisionModelInfo fromProto(const vlm::v1::VisionModelInfo& info)
{
    return VisionModelInfo{
        .width = info.width(),
        .height = info.height(),
        .channels = info.channels(),
        .image_tokens = info.image_tokens(),
        .embed_size = info.embed_size(),
    };
}

vlm::v1::RgbFrameBatch framesToBatch(const std::vector<PendingVisionFrame>& frames)
{
    vlm::v1::RgbFrameBatch batch;
    for (const auto& pending : frames) {
        auto* frame = batch.add_frames();
        frame->set_time_sec(pending.time_sec);
        frame->set_width(pending.frame.width);
        frame->set_height(pending.frame.height);
        if (!pending.frame.pixels.empty()) {
            frame->set_rgb24(pending.frame.pixels.data(), pending.frame.pixels.size());
        }
    }
    return batch;
}

std::vector<PendingVisionFrame> batchToFrames(const vlm::v1::RgbFrameBatch& batch)
{
    std::vector<PendingVisionFrame> frames;
    frames.reserve(static_cast<std::size_t>(batch.frames_size()));
    for (const auto& frame : batch.frames()) {
        PendingVisionFrame pending;
        pending.time_sec = frame.time_sec();
        pending.frame.width = frame.width();
        pending.frame.height = frame.height();
        pending.frame.pixels.assign(frame.rgb24().begin(), frame.rgb24().end());
        frames.push_back(std::move(pending));
    }
    return frames;
}

std::vector<std::string> parseTargets(const char* env_name, const char* default_value)
{
    const char* raw = std::getenv(env_name);
    if (raw == nullptr || raw[0] == '\0') {
        raw = default_value;
    }
    std::vector<std::string> targets;
    std::string current;
    for (const char* p = raw; *p != '\0'; ++p) {
        if (*p == ',') {
            if (!current.empty()) {
                targets.push_back(current);
                current.clear();
            }
        } else if (*p != ' ') {
            current.push_back(*p);
        }
    }
    if (!current.empty()) {
        targets.push_back(current);
    }
    return targets;
}

std::vector<std::string> expandTargets(const std::vector<std::string>& targets)
{
    std::vector<std::string> expanded;
    for (const auto& target : targets) {
        const auto pos = target.rfind(':');
        if (pos == std::string::npos) {
            expanded.push_back(target);
            continue;
        }
        const std::string host = target.substr(0, pos);
        const std::string port = target.substr(pos + 1);
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* result = nullptr;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &result) != 0) {
            expanded.push_back(target);
            continue;
        }
        for (addrinfo* info = result; info != nullptr; info = info->ai_next) {
            char addr[INET6_ADDRSTRLEN] = {};
            if (info->ai_family == AF_INET) {
                const auto* sin = reinterpret_cast<sockaddr_in*>(info->ai_addr);
                inet_ntop(AF_INET, &sin->sin_addr, addr, sizeof(addr));
            } else if (info->ai_family == AF_INET6) {
                const auto* sin6 = reinterpret_cast<sockaddr_in6*>(info->ai_addr);
                inet_ntop(AF_INET6, &sin6->sin6_addr, addr, sizeof(addr));
            } else {
                continue;
            }
            expanded.push_back(std::string(addr) + ':' + port);
        }
        freeaddrinfo(result);
    }
    if (expanded.empty()) {
        return targets;
    }
    std::sort(expanded.begin(), expanded.end());
    expanded.erase(std::unique(expanded.begin(), expanded.end()), expanded.end());
    return expanded;
}

}  // namespace vlm::grpc_util
