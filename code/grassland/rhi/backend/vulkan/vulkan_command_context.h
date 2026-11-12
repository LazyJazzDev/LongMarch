#pragma once
#include <array>
#include <unordered_map>
#include <vector>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/vulkan/vulkan_device.h"
#include "grassland/rhi/backend/vulkan/vulkan_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend::vulkan {
class IVulkanCommandContext : public IObject {
 public:
  typedef enum {
    SUBMIT_RESULT_REJECTED,
    SUBMIT_RESULT_SUBMITTED,
    SUBMIT_RESULT_FAILED_IN_FLIGHT,
  } SubmitResult;

 public:
  virtual void BindDownloads(rhi::ISubmissionToken *submission) = 0;
  virtual void ReleaseDownloads() = 0;
  virtual bool Seal() = 0;
  virtual void MarkSubmitted() = 0;
  virtual bool WasSubmitted() const = 0;
  virtual void UpdatePresentedImages() = 0;
  virtual SubmitResult SubmitNative(const SubmitInfo &info, Event signal) = 0;
  virtual SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) = 0;
};

ICommandContext *CreateVulkanCommandContext(IVulkanQueue *queue);

}  // namespace grassland::rhi::backend::vulkan

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::vulkan::IVulkanCommandContext,
                        grassland::rhi::IObject,
                        "c66f4335-1f50-47ca-9a8d-754583fbdfea")

namespace grassland::rhi::backend::vulkan {
class VulkanCommandContextImpl final : public Object<rhi::ICommandContext, IVulkanCommandContext> {
 public:
  bool CmdBindRayTracingProgram(IRayTracingProgram *program) override;
  bool CmdDispatchRays(uint32_t width,
                       uint32_t height,
                       uint32_t depth = 1,
                       IShaderBindingTable *sbt = nullptr) override;
  bool CmdQueryCompactedSize(IAccelerationStructure *structure) override;
  bool CmdCompactAccelerationStructure(IAccelerationStructure *source, IAccelerationStructure *destination) override;
  bool CmdBuildBottomLevelAccelerationStructure(
      IBottomLevelAccelerationStructure *destination,
      const std::vector<AccelerationStructureGeometryBuffers> &geometries,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override;
  bool CmdBuildTopLevelAccelerationStructure(
      ITopLevelAccelerationStructure *destination,
      const std::vector<RayTracingInstance> &instances,
      AccelerationStructureBuildMode mode = ACCELERATION_STRUCTURE_BUILD) override;
  ~VulkanCommandContextImpl() override;
  static ICommandContext *Create(IVulkanQueue *queue);
  IDownloadToken *CmdDownloadBuffer(IBuffer *buffer, uint64_t offset, uint64_t size) override;
  IDownloadToken *CmdDownloadImage(IImage *image, const ImageDownloadRegion &region = {}) override;
  bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) override;
  bool CmdCopyBuffer(IBuffer *source,
                     uint64_t source_offset,
                     IBuffer *destination,
                     uint64_t destination_offset,
                     uint64_t size) override;
  bool CmdBindGroup(uint32_t index, IBindingGroup *group) override;
  bool CmdUploadImage(IImage *image, const void *data, uint64_t row_pitch) override;
  bool CmdBindComputeProgram(IComputeProgram *program) override;
  bool CmdDispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1) override;
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
  bool CmdClearImage(rhi::IImage *image, const ClearValue &value) override;
  bool CmdPresent(rhi::IWindow *window) override;
  void UpdatePresentedImages() override;
  SubmitResult SubmitNative(const SubmitInfo &info, Event signal) override;
  SubmissionStatus PollCompletion(Event signal, uint64_t completed, bool timeline_valid) override;

 private:
  const void *DeviceIdentity() const;
  IVulkanQueue *OwnerQueue() const;

  // Inputs and ranges have been validated by the recording entry point.
  bool CopyBuffer(IVulkanBuffer *source,
                  uint64_t source_offset,
                  IVulkanBuffer *destination,
                  uint64_t destination_offset,
                  uint64_t size);
  VulkanCommandContextImpl(IVulkanQueue *queue);
  bool ValidateGroups();
  bool BindGroupVersion(uint32_t index, IVulkanBindingVersion *version);
  std::vector<IVulkanBindingVersion *> groups_;
  IProgram *program_{};
  IVulkanProgramBase *native_program_{};
  std::unordered_map<IVulkanTopLevelAccelerationStructure *, std::vector<ObjectPtr<IBottomLevelAccelerationStructure>>>
      as_dependencies_;
  bool compute_active_{};
  IVulkanRayTracingProgram *ray_program_{};
  const PipelineLayoutDesc *binding_layout_{};
  IVulkanImage *render_target_{};
  std::vector<IVulkanBuffer *> vertex_buffers_;
  bool viewport_set_{}, scissor_set_{}, index_bound_{};
  bool EndRecording();
  bool TransitionBuffer(IVulkanBuffer *buffer, VkPipelineStageFlags stages, VkAccessFlags access);

  struct BufferState {
    VkPipelineStageFlags stages;
    VkAccessFlags access;
  };

  std::unordered_map<IVulkanBuffer *, BufferState> buffer_states_;
  bool TransitionImage(IVulkanImage *image, VkImageLayout layout, VkPipelineStageFlags stages, VkAccessFlags access);

  struct ImageState {
    VkImageLayout layout;
    VkPipelineStageFlags stages;
    VkAccessFlags access;
  };

  std::unordered_map<IVulkanImage *, ImageState> image_states_;
  std::vector<IVulkanWindow *> presents_;
  IVulkanDevice *device_;
  VkCommandPool pool_{};
  VkCommandBuffer command_{};
#if defined(LONGMARCH_CUDA_RUNTIME)
  VkCommandBuffer cuda_acquire_{};
#endif
  bool submission_failed_{};
  bool reusable_{};

 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override;

  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override;

 private:
  bool PrepareRendering(const RenderingInfo &info);

  bool RenderingMatches(const IProgram *program) const;

  void InitializeCommandContext(IVulkanQueue *queue);

 public:
  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override;

  void BindDownloads(rhi::ISubmissionToken *submission) override;

  void ReleaseDownloads() override;

 private:
  // Call before encoding each resource use. Retains each exact object once.
  IDownloadToken *RegisterDownload(ObjectPtr<IVulkanDownloadToken> token);

  template <typename T, std::enable_if_t<!std::is_convertible_v<T *, rhi::IObject *>, int> = 0>
  bool RetainResource(T *resource) {
    return RetainResource(rhi::QueryInterfacePtr<rhi::IObject>(resource).Get());
  }

  bool RetainResource(rhi::IObject *resource);

 public:
  bool Seal() override;

  void MarkSubmitted() override;

  bool WasSubmitted() const override;

 private:
  RenderingInfo rendering_{};
  bool recording_failed_{};
  bool sealed_{};
  IVulkanQueue *queue_{};
  std::unordered_map<rhi::IObject *, ObjectPtr<rhi::IObject>> resources_{};
  bool submitted_{};
  std::vector<ObjectPtr<IVulkanDownloadToken>> downloads_{};
};
}  // namespace grassland::rhi::backend::vulkan
