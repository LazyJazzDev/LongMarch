#pragma once
#include <cstring>
#include <stdexcept>

#include "grassland/graphics/backend/native/native_core.h"
#include "grassland/graphics/buffer.h"
#include "grassland/graphics/image.h"
#include "grassland/graphics/program.h"
#include "grassland/graphics/sampler.h"
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
#include <cuda.h>
#include <nvrtc.h>
#endif

namespace grassland::graphics::backend {
#ifdef LONGMARCH_NATIVE_CUDA_ENABLED
inline void CheckCUDA(CUresult result) {
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string("native CUDA: ") + (text ? text : "unknown driver error"));
  }
}
#endif
[[noreturn]] inline void NativeUnsupported() {
  throw std::runtime_error(
      "operation unavailable on this headless native backend; use a graphics backend for rasterization or presentation");
}

// Native target ABI: buffer/unsized array = pointer + byte size/element count.
struct NativeSpan {
  void *data{};
  size_t size{};
};
struct NativeImageBinding {
  NativeSpan pixels;
  uint32_t width{}, height{};
  uint32_t unorm{}, padding{};
};
static_assert(sizeof(NativeImageBinding) == 32);
class NativeMemory {
 public:
  NativeMemory(bool cuda, size_t size);
  ~NativeMemory();
  NativeMemory(const NativeMemory &) = delete;
  void Upload(const void *, size_t, size_t = 0) const;
  void Download(void *, size_t, size_t = 0) const;
  void *Data() const {
    return data_;
  }
  size_t Size() const {
    return size_;
  }
  bool IsCUDA() const {
    return cuda_;
  }

 private:
  bool cuda_;
  size_t size_;
  void *data_{};
};
class NativeBuffer final : public Buffer {
 public:
  NativeBuffer(bool cuda, size_t size, BufferType type);
  BufferType Type() const override {
    return type_;
  }
  size_t Size() const override {
    return memory->Size();
  }
  void Resize(size_t size) override;
  void UploadData(const void *p, size_t s, size_t o) override {
    memory->Upload(p, s, o);
  }
  void DownloadData(void *p, size_t s, size_t o) override {
    memory->Download(p, s, o);
  }
  std::unique_ptr<NativeMemory> memory;

 private:
  bool cuda_;
  BufferType type_;
};
class NativeImage final : public Image {
 public:
  NativeImage(bool cuda, int width, int height, ImageFormat format);
  Extent2D Extent() const override {
    return extent_;
  }
  ImageFormat Format() const override {
    return format_;
  }
  void UploadData(const void *data) const override {
    UploadData(data, {0, 0}, extent_);
  }
  void DownloadData(void *data) const override {
    DownloadData(data, {0, 0}, extent_);
  }
  void UploadData(const void *, const Offset2D &, const Extent2D &) const override;
  void DownloadData(void *, const Offset2D &, const Extent2D &) const override;
  void Clear(const ClearValue &value);
  NativeImageBinding Binding() const {
    return {{memory->Data(), memory->Size()}, extent_.width, extent_.height, uint32_t(unorm_), 0};
  }
  std::unique_ptr<NativeMemory> memory;

 private:
  Extent2D extent_;
  ImageFormat format_;
  size_t channels_, external_bytes_;
  bool unorm_;
};
class NativeSampler final : public Sampler {
 public:
  explicit NativeSampler(const SamplerInfo &info) : info(info) {
  }
  SamplerInfo info;
};
struct NativeBindings {
  std::map<int, std::vector<BufferRange>> buffers;
  std::map<int, std::vector<Image *>> images;
  std::map<int, std::vector<Sampler *>> samplers;
  std::map<int, AccelerationStructure *> acceleration_structures;
};
class NativeShader final : public Shader {
 public:
  NativeShader(bool cuda,
               const VirtualFileSystem &,
               const std::string &,
               const std::string &,
               const std::vector<std::string> &,
               OptixDevice *optix = nullptr);
  ~NativeShader() override;
  std::string EntryPoint() const override;
  void Dispatch(const NativeBindings &, uint32_t, uint32_t, uint32_t);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
class NativeProgram final : public ComputeProgram {
 public:
  explicit NativeProgram(NativeShader *shader) : shader(shader) {
  }
  void AddResourceBinding(ResourceType type, int count) override {
    resources.emplace_back(type, count);
  }
  void Finalize() override {
  }
  NativeShader *shader;
  std::vector<std::pair<ResourceType, int>> resources;
};
}  // namespace grassland::graphics::backend
