#pragma once
#include "grassland/rhi/rhi_object.h"

LONGMARCH_RHI_UUID(grassland::rhi::IBuffer, "57217c65-4df9-4f6c-b6a6-0d177729b168")

namespace grassland::rhi {
struct BufferDesc {
  uint64_t size{};
};

// Fixed physical allocation. GPU-only contents are initialized/updated with CmdUploadBuffer.
class IBuffer : public IObject {
 public:
  virtual BufferDesc GetDesc() const = 0;

 protected:
  ~IBuffer() override;
};
}  // namespace grassland::rhi
