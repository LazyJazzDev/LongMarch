#pragma once
#include <array>

#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/rhi_command_context.h"

namespace grassland::rhi::backend {
class Window;

class CommandContext : public Object<rhi::ICommandContext> {
 public:
  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

  CommandContext(const void *identity, QueueType type);
  ~CommandContext() override;
  bool CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) override;
  bool CmdPresent(rhi::IWindow *window) override;
  bool CmdClearColor(const ObjectPtr<Image> &image, const std::array<float, 4> &color);
  bool Seal();
  void AdvanceFrame();
  const void *DeviceIdentity() const;
  QueueType Type() const;
  const ObjectPtr<Image> &Frame() const;
  void MarkSubmitted();
  bool WasSubmitted() const;
  virtual SubmitResult SubmitNative() = 0;
  virtual SubmissionStatus WaitForProgress() = 0;

 protected:
  virtual bool RecordClear(Image *image, const std::array<float, 4> &color, bool first_use) = 0;
  virtual bool RecordPresent(Image *image) = 0;
  virtual bool EndRecording() = 0;

 private:
  Window *present_window_{};
  const void *identity_;
  QueueType type_;
  ObjectPtr<Image> image_;
  bool present_{};
  bool sealed_{};
  bool failed_{};
  bool submitted_{};
};

}  // namespace grassland::rhi::backend
