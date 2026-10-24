#include "last_frame.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace rhi_demo {
using namespace grassland::rhi;

LastFrame::LastFrame(const std::string &directory, const std::string &name) : directory_(directory), name_(name) {
}

bool LastFrame::Record(ICommandContext *commands, IImage *image, IWindow *window) {
  if (directory_.empty())
    return true;
  download_ = commands->CmdDownloadImage(image);
  if (!download_)
    return false;
  desc_ = image->GetDesc();
  space_ = window ? window->GetColorSpace() : COLOR_SPACE_SRGB_NONLINEAR;
  white_scale_ = window ? window->GetHDRReferenceWhiteScale() : 1.0f;
  return true;
}

void LastFrame::Submitted(ISubmissionToken *completion) {
  if (!directory_.empty())
    completion_ = ObjPtr<ISubmissionToken>::Retain(completion);
}

namespace {
float EncodeSRGB(float x) {
  return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

float DecodePQ(float x) {
  const float p = std::pow(std::max(x, 0.0f), 1.0f / 78.84375f);
  return 10000.0f * std::pow(std::max(p - 0.8359375f, 0.0f) / (18.8515625f - 18.6875f * p), 1.0f / 0.1593017578125f);
}
}  // namespace

bool LastFrame::Save() {
  if (directory_.empty())
    return true;
  if (!download_) {
    std::cerr << name_ << ": no rendered frame to save\n";
    return false;
  }
  if (!completion_ || !completion_->WaitHost())
    return false;
  std::vector<uint8_t> pixels(download_->GetSize());
  if (!download_->CopyTo(pixels.data(), pixels.size()))
    return false;
  std::error_code error;
  const auto directory = std::filesystem::u8path(directory_);
  std::filesystem::create_directories(directory, error);
  if (error) {
    std::cerr << error.message() << '\n';
    return false;
  }
  const auto base = directory / name_;
  // Preserve exact HDR/10-bit/half-float values alongside the SDR PNG preview.
  auto raw_path = base;
  raw_path += ".bin";
  std::ofstream raw(raw_path, std::ios::binary);
  raw.write(reinterpret_cast<const char *>(pixels.data()), pixels.size());
  raw.close();
  auto metadata_path = base;
  metadata_path += ".txt";
  std::ofstream metadata(metadata_path);
  metadata << "width=" << desc_.width << "\nheight=" << desc_.height << "\nformat=" << ImageFormatName(desc_.format)
           << "\ncolor_space=" << ColorSpaceName(space_)
           << "\nrow_pitch=" << uint64_t(desc_.width) * ImageFormatSize(desc_.format)
           << "\nreference_white_scale=" << white_scale_
           << "\nlayout=top-to-bottom, tightly packed, native pixel encoding\n";
  metadata.close();
  if (!raw || !metadata)
    return false;
  std::vector<uint8_t> preview(size_t(desc_.width) * desc_.height * 4);
  const uint32_t stride = ImageFormatSize(desc_.format);
  for (size_t i = 0; i < size_t(desc_.width) * desc_.height; ++i) {
    const auto source = pixels.data() + i * stride;
    float c[4]{};
    if (desc_.format == IMAGE_FORMAT_RGBA16_FLOAT) {
      uint16_t halves[4];
      std::memcpy(halves, source, 8);
      for (int k = 0; k < 4; ++k)
        c[k] = glm::unpackHalf1x16(halves[k]);
    } else if (desc_.format == IMAGE_FORMAT_RGB10A2_UNORM || desc_.format == IMAGE_FORMAT_BGR10A2_UNORM) {
      uint32_t value;
      std::memcpy(&value, source, 4);
      for (int k = 0; k < 3; ++k)
        c[k] = float((value >> (10 * k)) & 1023) / 1023.0f;
      c[3] = float(value >> 30) / 3.0f;
      if (desc_.format == IMAGE_FORMAT_BGR10A2_UNORM)
        std::swap(c[0], c[2]);
    } else {
      for (int k = 0; k < 4; ++k)
        c[k] = source[k] / 255.0f;
      if (desc_.format == IMAGE_FORMAT_BGRA8_UNORM)
        std::swap(c[0], c[2]);
    }
    if (space_ == COLOR_SPACE_HDR10_ST2084) {
#if defined(_WIN32)
      const float white = 80.0f * white_scale_;
#elif defined(__APPLE__)
      const float white = 100.0f * white_scale_;
#else
      const float white = 203.0f * white_scale_;
#endif
      const float r = DecodePQ(c[0]) / white, g = DecodePQ(c[1]) / white, b = DecodePQ(c[2]) / white;
      c[0] = 1.660491f * r - 0.587641f * g - 0.072850f * b;
      c[1] = -0.124550f * r + 1.132900f * g - 0.008349f * b;
      c[2] = -0.018151f * r - 0.100579f * g + 1.118730f * b;
      for (int k = 0; k < 3; ++k)
        c[k] = EncodeSRGB(std::max(c[k], 0.0f));
    } else if (space_ != COLOR_SPACE_SRGB_NONLINEAR) {
      for (int k = 0; k < 3; ++k)
        c[k] = EncodeSRGB(std::max(c[k] / white_scale_, 0.0f));
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
  const bool encoded = stbi_write_png_to_func(write, &png, int(desc_.width), int(desc_.height), 4, preview.data(),
                                              int(desc_.width * 4)) != 0;
  png.close();
  if (!encoded || !png)
    return false;
  std::cout << "Saved last frame: " << filename.u8string() << " (raw pixels: .bin, metadata: .txt)\n";
  return true;
}
}  // namespace rhi_demo
