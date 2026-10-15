#pragma once
#include <array>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;

class ICommandContext : public IObject {
 public:
  virtual bool CmdClearColor(IImage *image, const std::array<float, 4> &color) = 0;
  virtual bool CmdPresent(IImage *image) = 0;

 protected:
  ~ICommandContext() override;
};

}  // namespace grassland::rhi
