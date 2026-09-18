// HLSL resource types for the Sparkium CPU backend.
//
// The shaders address their inputs through ByteAddressBuffer, Texture2D and
// ConstantBuffer declarations. On the GPU those bind to device resources; here
// they are thin views over host memory the pipeline fills in before a frame.
//
// The byte-addressed views deliberately mirror the GPU's layouts. Materials
// upload their parameters at fixed offsets (material/light/sampler.hlsl reads
// `Load(12)`, `Load(16)`, ...), so the host side must preserve them exactly.
#pragma once

#include <cstdint>
#include <cstring>

#include "sparkium/pipelines/raytracing/cpu/shaders/hlsl_cpu_matrix.h"

namespace sparkium::cpu::hlsl {

// The generated shader graphs broadcast through SPARKIUM_SPLAT4; on the GPU
// common.hlsli defines it to the HLSL swizzle, here it maps to the helper in
// hlsl_cpu_ops.h.
#define SPARKIUM_SPLAT4(x) Splat4(x)

// NonUniformResourceIndex is a GPU scheduling hint; indexing host arrays needs
// no equivalent.
inline uint32_t NonUniformResourceIndex(uint32_t index) {
  return index;
}

// A ray as the shaders construct it (`RayDesc ray; ray.Origin = ...`). HLSL
// defines this type implicitly; the CPU side makes it explicit.
struct RayDesc {
  Float3 Origin;
  float TMin;
  Float3 Direction;
  float TMax;
};

class ByteAddressBuffer {
 public:
  ByteAddressBuffer() = default;
  ByteAddressBuffer(const void *data, size_t bytes) : data_(static_cast<const uint8_t *>(data)), size_(bytes) {
  }

  const uint8_t *Data() const {
    return data_;
  }
  size_t Size() const {
    return size_;
  }

  uint32_t Load(uint32_t offset) const {
    uint32_t value;
    std::memcpy(&value, data_ + offset, sizeof(value));
    return value;
  }
  Uint2 Load2(uint32_t offset) const {
    Uint2 value;
    std::memcpy(&value, data_ + offset, sizeof(value));
    return value;
  }
  Uint3 Load3(uint32_t offset) const {
    Uint3 value;
    std::memcpy(&value, data_ + offset, sizeof(value));
    return value;
  }
  Uint4 Load4(uint32_t offset) const {
    Uint4 value;
    std::memcpy(&value, data_ + offset, sizeof(value));
    return value;
  }
  // sizeof(T) is the HLSL size as long as T is built from the packed vector
  // types above, which the static asserts in software_pipeline.cpp pin down.
  template <class T>
  T Load(uint32_t offset) const {
    T value;
    std::memcpy(&value, data_ + offset, sizeof(T));
    return value;
  }

 protected:
  const uint8_t *data_{nullptr};
  size_t size_{0};
};

class RWByteAddressBuffer : public ByteAddressBuffer {
 public:
  RWByteAddressBuffer() = default;
  RWByteAddressBuffer(void *data, size_t bytes) : ByteAddressBuffer(data, bytes), writable_(static_cast<uint8_t *>(data)) {
  }

  void Store(uint32_t offset, uint32_t value) {
    std::memcpy(writable_ + offset, &value, sizeof(value));
  }
  void Store2(uint32_t offset, const Uint2 &value) {
    std::memcpy(writable_ + offset, &value, sizeof(value));
  }
  void Store3(uint32_t offset, const Uint3 &value) {
    std::memcpy(writable_ + offset, &value, sizeof(value));
  }
  void Store4(uint32_t offset, const Uint4 &value) {
    std::memcpy(writable_ + offset, &value, sizeof(value));
  }
  template <class T>
  void Store(uint32_t offset, const T &value) {
    std::memcpy(writable_ + offset, &value, sizeof(T));
  }

 private:
  uint8_t *writable_{nullptr};
};

// Samplers on this backend are all linear with repeat addressing and no mip
// chain, matching the SamplerInfo the raytracing scene creates.
struct SamplerState {};

// Host-side image data. The pipeline converts every registered image to RGBA
// float once, which keeps sampling free of format branches.
struct TextureData {
  float *pixels{nullptr};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t channels{4};
};

namespace detail {
// Bilinear fetch with repeat addressing, matching the GPU's linear filter.
inline void Bilinear(const TextureData &texture, float u, float v, float *out) {
  const int channels = static_cast<int>(texture.channels);
  const float fx = u * static_cast<float>(texture.width) - 0.5f;
  const float fy = v * static_cast<float>(texture.height) - 0.5f;
  const float x0 = std::floor(fx);
  const float y0 = std::floor(fy);
  const float tx = fx - x0;
  const float ty = fy - y0;
  const auto wrap = [](int value, uint32_t extent) {
    const int size = static_cast<int>(extent);
    value %= size;
    return value < 0 ? value + size : value;
  };
  const int ix0 = wrap(static_cast<int>(x0), texture.width);
  const int iy0 = wrap(static_cast<int>(y0), texture.height);
  const int ix1 = wrap(static_cast<int>(x0) + 1, texture.width);
  const int iy1 = wrap(static_cast<int>(y0) + 1, texture.height);
  const auto fetch = [&](int x, int y, int channel) {
    const uint32_t index = static_cast<uint32_t>(y) * texture.width + static_cast<uint32_t>(x);
    if (channel >= channels)
      return channel == 3 ? 1.0f : 0.0f;
    return texture.pixels[static_cast<size_t>(index) * channels + channel];
  };
  for (int channel = 0; channel < 4; ++channel) {
    const float top = fetch(ix0, iy0, channel) + (fetch(ix1, iy0, channel) - fetch(ix0, iy0, channel)) * tx;
    const float bottom = fetch(ix0, iy1, channel) + (fetch(ix1, iy1, channel) - fetch(ix0, iy1, channel)) * tx;
    out[channel] = top + (bottom - top) * ty;
  }
}
}  // namespace detail

template <class T>
class Texture2D {
 public:
  Texture2D() = default;
  explicit Texture2D(const TextureData *data) : data_(data) {
  }

  const TextureData *Data() const {
    return data_;
  }

  void GetDimensions(uint32_t &width, uint32_t &height) const {
    width = data_ ? data_->width : 0;
    height = data_ ? data_->height : 0;
  }

  T Load(const Int3 &coord) const {
    T value{};
    // Pixels are stored interleaved, so a pixel's offset advances by the
    // channel count, not by one float.
    const size_t index =
        (static_cast<size_t>(coord.y) * data_->width + static_cast<size_t>(coord.x)) * data_->channels;
    std::memcpy(&value, &data_->pixels[index], sizeof(T));
    return value;
  }

  Float4 SampleLevel(SamplerState, const Float2 &uv, float) const {
    Float4 result;
    detail::Bilinear(*data_, uv.x, uv.y, result.data);
    return result;
  }
  Float4 Sample(SamplerState sampler, const Float2 &uv) const {
    return SampleLevel(sampler, uv, 0.0f);
  }

 private:
  const TextureData *data_{nullptr};
};

// A writable image. The path tracer accumulates into accumulated_color and
// accumulated_samples through this.
template <class T>
class RWTexture2D {
 public:
  RWTexture2D() = default;
  explicit RWTexture2D(TextureData *data) : data_(data) {
  }

  TextureData *Data() const {
    return data_;
  }

  void GetDimensions(uint32_t &width, uint32_t &height) const {
    width = data_ ? data_->width : 0;
    height = data_ ? data_->height : 0;
  }

  T &operator[](const Uint2 &coord) {
    return reinterpret_cast<T *>(data_->pixels)[static_cast<size_t>(coord.y) * data_->width + coord.x];
  }
  const T &operator[](const Uint2 &coord) const {
    return reinterpret_cast<const T *>(data_->pixels)[static_cast<size_t>(coord.y) * data_->width + coord.x];
  }

 private:
  TextureData *data_{nullptr};
};

// ConstantBuffers are small and re-uploaded every frame, so the host keeps the
// value inline and deriving from T is enough for `cb.field` to work.
template <class T>
struct ConstantBuffer : T {};

}  // namespace sparkium::cpu::hlsl
