#pragma once
#include <array>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;

class IRecordedCommands : public IObject {
 protected:
  ~IRecordedCommands() override;
};

class ICommandContext : public IRecordedCommands {
 public:
  virtual bool ClearColor(IImage *image, const std::array<float, 4> &color) = 0;
  virtual bool Present(IImage *image) = 0;
  // Returns a new owned reference to sealed commands; does not release this
  // caller's context reference. Finished contexts cannot be recorded again.
  virtual IRecordedCommands *Finish() = 0;

 protected:
  ~ICommandContext() override;
};

}  // namespace grassland::rhi
