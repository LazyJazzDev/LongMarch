#pragma once
#include "grassland/rhi/rhi_interface_id.h"

namespace grassland::rhi::backend {
class Buffer;
}
LONGMARCH_RHI_DECLARE_UUID(grassland::rhi::backend::Buffer)
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_buffer.h"

namespace grassland::rhi::backend {
class Device;

class Buffer : public Object<IBuffer> {
 public:
  LONGMARCH_RHI_QUERY(Buffer, Object<IBuffer>)

  BufferDesc GetDesc() const override;
  virtual Device *Owner() const = 0;

 protected:
  explicit Buffer(const BufferDesc &desc);
  BufferDesc desc_;
};
}  // namespace grassland::rhi::backend
