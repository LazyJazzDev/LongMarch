#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {

class ISubmissionToken : public IObject {
 public:
  virtual SubmissionStatus GetStatus() const = 0;
  virtual bool WaitHost() const = 0;

 protected:
  ~ISubmissionToken() override;
};

}  // namespace grassland::rhi
