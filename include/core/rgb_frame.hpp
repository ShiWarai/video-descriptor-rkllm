#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vlm {

/** Model-ready RGB888 frame (interleaved R,G,B per pixel). */
struct RgbFrame {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> pixels;

    [[nodiscard]] bool empty() const noexcept { return pixels.empty(); }

    [[nodiscard]] const std::uint8_t* data() const noexcept { return pixels.data(); }

    [[nodiscard]] std::size_t byteSize() const noexcept { return pixels.size(); }

    [[nodiscard]] bool matches(int w, int h) const noexcept
    {
        return width == w && height == h &&
               pixels.size() == static_cast<std::size_t>(w) * static_cast<std::size_t>(h) * 3;
    }

    static RgbFrame fromRaw(int w, int h, std::vector<std::uint8_t> raw)
    {
        RgbFrame frame;
        frame.width = w;
        frame.height = h;
        frame.pixels = std::move(raw);
        return frame;
    }
};

}  // namespace vlm
