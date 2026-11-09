#pragma once
#include <array>
#include <unordered_map>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend {

class CommandContext : public Object<rhi::ICommandContext> {
 public:
  bool RecordWindowImGui(IWindow *window, IImage *target);
  bool RecordWindowPresentation(IWindow *window, IImage *frame_img, ColorSpace frame_color_space);

  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

  CommandContext(const void *identity);
  ~CommandContext() override;

  void BindDownloads(SubmissionToken *submission);
  void ReleaseDownloads();
  bool Seal();
  const void *DeviceIdentity() const;
  void MarkSubmitted();
  bool WasSubmitted() const;
  virtual SubmitResult SubmitNative() = 0;
  virtual SubmissionStatus WaitForProgress() = 0;

 protected:
  // Call before encoding each resource use. Retains each exact object once.
  IDownloadToken *RegisterDownload(ObjectPtr<DownloadToken> token);
  bool RetainResource(rhi::IObject *resource);
  virtual bool EndRecording() = 0;
  bool PrepareRendering(const RenderingInfo &info);
  bool RenderingMatches(const IProgram *program) const;
  RenderingInfo rendering_;
  bool recording_failed_{};
  bool sealed_{};

 private:
  const void *identity_;
  std::unordered_map<rhi::IObject *, ObjectPtr<rhi::IObject>> resources_;
  bool submitted_{};
  std::vector<ObjectPtr<DownloadToken>> downloads_;
};

}  // namespace grassland::rhi::backend
