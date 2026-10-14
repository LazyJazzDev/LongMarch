#pragma once
#include "grassland/rhi/rhi_util.h"

namespace grassland::rhi {

class Device {
 public:
  std::string Name() const;
  bool SupportRayTracingPipeline() const;
  bool SupportRayQuery() const;
  bool SupportGeometryShader() const;
  explicit operator bool() const noexcept;

 private:
  ObjectRef object_;
  friend struct backend::Access;
};

}  // namespace grassland::rhi
