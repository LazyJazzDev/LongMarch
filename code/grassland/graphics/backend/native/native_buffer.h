#pragma once
#include "grassland/graphics/buffer.h"
#include "native_memory.h"

namespace grassland::graphics::backend {

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

}  // namespace grassland::graphics::backend
