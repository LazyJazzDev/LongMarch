#pragma once
#include "grassland/rhi/rhi_object.h"

LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::IBuffer)

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
