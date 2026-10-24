#include "save_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace grassland::rhi;

namespace {
float EncodeSRGB(float x) {
  return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

float DecodePQ(float x) {
  const float p = std::pow(std::max(x, 0.0f), 1.0f / 78.84375f);
  return 10000.0f * std::pow(std::max(p - 0.8359375f, 0.0f) / (18.8515625f - 18.6875f * p), 1.0f / 0.1593017578125f);
}
}  // namespace

bool SaveImage(const std::string &directory_name,
               const std::string &name,
               const std::vector<uint8_t> &pixels,
               const grassland::rhi::ImageDesc &desc,
               grassland::rhi::ColorSpace space,
               float white_scale) {
  std::error_code error;
  const auto directory = std::filesystem::u8path(directory_name);
  std::filesystem::create_directories(directory, error);
  if (error) {
    std::cerr << error.message() << '\n';
    return false;
  }
  const bool depth = desc.format == IMAGE_FORMAT_D32_FLOAT;
  const auto base = directory / name;
  // Preserve exact HDR/10-bit/half-float values alongside the SDR PNG preview.
  auto raw_path = base;
  raw_path += ".bin";
  std::ofstream raw(raw_path, std::ios::binary);
  raw.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
  raw.close();
  auto metadata_path = base;
  metadata_path += ".txt";
  std::ofstream metadata(metadata_path);
  metadata << "width=" << desc.width << "\nheight=" << desc.height << "\nformat=" << ImageFormatName(desc.format)
           << "\ncolor_space=" << (depth ? "none (depth)" : ColorSpaceName(space))
           << "\nrow_pitch=" << uint64_t(desc.width) * ImageFormatSize(desc.format)
           << "\nreference_white_scale=" << white_scale
           << "\nlayout=top-to-bottom, tightly packed, native pixel encoding\n";
  if (depth)
    metadata << "depth_encoding=device depth in [0,1], not linear distance\n"
             << "preview=clamped depth mapped directly to 8-bit grayscale, 0 black, 1 white\n";
  metadata.close();
  if (!raw || !metadata)
    return false;
  std::vector<uint8_t> preview(size_t(desc.width) * desc.height * 4);
  const uint32_t stride = ImageFormatSize(desc.format);
  for (size_t i = 0; i < size_t(desc.width) * desc.height; ++i) {
    const auto source = pixels.data() + i * stride;
    float c[4]{};
    if (depth) {
      float value;
      std::memcpy(&value, source, sizeof(value));
      c[0] = c[1] = c[2] = value;
      c[3] = 1.0f;
    } else if (desc.format == IMAGE_FORMAT_RGBA16_FLOAT) {
      uint16_t halves[4];
      std::memcpy(halves, source, 8);
      for (int k = 0; k < 4; ++k)
        c[k] = glm::unpackHalf1x16(halves[k]);
    } else if (desc.format == IMAGE_FORMAT_RGB10A2_UNORM || desc.format == IMAGE_FORMAT_BGR10A2_UNORM) {
      uint32_t value;
      std::memcpy(&value, source, 4);
      for (int k = 0; k < 3; ++k)
        c[k] = float((value >> (10 * k)) & 1023) / 1023.0f;
      c[3] = float(value >> 30) / 3.0f;
      if (desc.format == IMAGE_FORMAT_BGR10A2_UNORM)
        std::swap(c[0], c[2]);
    } else {
      for (int k = 0; k < 4; ++k)
        c[k] = source[k] / 255.0f;
      if (desc.format == IMAGE_FORMAT_BGRA8_UNORM)
        std::swap(c[0], c[2]);
    }
    if (!depth && space == COLOR_SPACE_HDR10_ST2084) {
#if defined(_WIN32)
      const float white = 80.0f * white_scale;
#elif defined(__APPLE__)
      const float white = 100.0f * white_scale;
#else
      const float white = 203.0f * white_scale;
#endif
      const float r = DecodePQ(c[0]) / white, g = DecodePQ(c[1]) / white, b = DecodePQ(c[2]) / white;
      c[0] = 1.660491f * r - 0.587641f * g - 0.072850f * b;
      c[1] = -0.124550f * r + 1.132900f * g - 0.008349f * b;
      c[2] = -0.018151f * r - 0.100579f * g + 1.118730f * b;
      for (int k = 0; k < 3; ++k)
        c[k] = EncodeSRGB(std::max(c[k], 0.0f));
    } else if (!depth && space != COLOR_SPACE_SRGB_NONLINEAR) {
      for (int k = 0; k < 3; ++k)
        c[k] = EncodeSRGB(std::max(c[k] / white_scale, 0.0f));
    }
    for (int k = 0; k < 4; ++k)
      preview[i * 4 + k] = uint8_t(std::lround(std::clamp(std::isfinite(c[k]) ? c[k] : 0.0f, 0.0f, 1.0f) * 255.0f));
  }
  auto filename = base;
  filename += ".png";
  std::ofstream png(filename, std::ios::binary);
  if (!png)
    return false;
  const auto write = [](void *context, void *data, int size) {
    static_cast<std::ofstream *>(context)->write(static_cast<const char *>(data), size);
  };
  const bool encoded = stbi_write_png_to_func(write, &png, int(desc.width), int(desc.height), 4, preview.data(),
                                              int(desc.width * 4)) != 0;
  png.close();
  if (!encoded || !png)
    return false;
  std::cout << "Saved last frame: " << filename.u8string() << " (raw pixels: .bin, metadata: .txt)\n";
  return true;
}
