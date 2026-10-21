#pragma once
#include <array>
#include <unordered_map>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_command_context.h"

namespace grassland::rhi::backend {

class CommandContext : public Object<rhi::ICommandContext> {
 public:
  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

  CommandContext(const void *identity);
  ~CommandContext() override;
  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override;
  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override;
  bool CmdBindGroup(uint32_t index, IBindingGroup *group) override;
  bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) override;
  bool CmdBindProgram(IProgram *program) override;
  bool CmdBeginRendering(const RenderingInfo &info) override;
  bool CmdEndRendering() override;
  bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0) override;
  bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0) override;
  bool CmdSetViewport(const Viewport &viewport) override;
  bool CmdSetScissor(const Scissor &scissor) override;
  bool CmdDraw(uint32_t vertex_count,
               uint32_t instance_count = 1,
               uint32_t first_vertex = 0,
               uint32_t first_instance = 0) override;
  bool CmdDrawIndexed(uint32_t index_count,
                      uint32_t instance_count = 1,
                      uint32_t first_index = 0,
                      int32_t vertex_offset = 0,
                      uint32_t first_instance = 0) override;
  bool Seal();
  const void *DeviceIdentity() const;
  void MarkSubmitted();
  bool WasSubmitted() const;
  virtual SubmitResult SubmitNative() = 0;
  virtual SubmissionStatus WaitForProgress() = 0;

 protected:
  // Call before encoding each resource use. Retains each exact object once.
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
};

}  // namespace grassland::rhi::backend
