#pragma once
#include <array>

#include "grassland/rhi/backend/backend_image.h"

namespace grassland::rhi::backend {

class ICommandContext : public IObject {
 public:
  enum class SubmitResult { Rejected, Submitted, FailedInFlight };
  ICommandContext(const void *identity, QueueType type);
  ~ICommandContext() override;
  bool Clear(const ObjectPtr<IImage> &image, const std::array<float, 4> &color);
  bool Present(const ObjectPtr<IImage> &image);
  bool Seal();
  const void *DeviceIdentity() const;
  QueueType Type() const;
  const ObjectPtr<IImage> &Frame() const;
  const std::array<float, 4> &Color() const;
  void MarkSubmitted();
  bool WasSubmitted() const;
  virtual SubmitResult SubmitNative() = 0;
  virtual SubmissionStatus WaitForProgress() = 0;

 private:
  const void *identity_;
  QueueType type_;
  ObjectPtr<IImage> image_;
  std::array<float, 4> color_{};
  bool present_{};
  bool sealed_{};
  bool failed_{};
  bool submitted_{};
};

}  // namespace grassland::rhi::backend
