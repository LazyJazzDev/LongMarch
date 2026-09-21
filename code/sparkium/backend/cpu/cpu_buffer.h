#pragma once
#include "grassland/graphics/buffer.h"
#include "sparkium/backend/cpu/cpu_memory.h"

namespace sparkium::backend::cpu {
using namespace grassland;
using namespace grassland::graphics;

class CpuBuffer final : public Buffer {
 public:
  CpuBuffer(size_t size, BufferType type);

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

  std::unique_ptr<CpuMemory> memory;

 private:
  BufferType type_;
};

}  // namespace sparkium::backend::cpu
