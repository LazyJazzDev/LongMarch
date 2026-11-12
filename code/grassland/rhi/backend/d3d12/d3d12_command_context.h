#pragma once
#include <array>
#include <unordered_map>
#include <vector>

#include "grassland/rhi/backend/backend_download_token.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_util.h"
#include "grassland/rhi/rhi_acceleration_structure.h"
#include "grassland/rhi/rhi_command_context.h"
#include "grassland/rhi/rhi_ray_tracing_program.h"

namespace grassland::rhi::backend::d3d12 {
class ID3D12CommandContext : public IObject {
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

ICommandContext *CreateD3D12CommandContext(ID3D12Queue *queue);

}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::d3d12::ID3D12CommandContext,
                        grassland::rhi::IObject,
                        "ef4ee783-d8b8-4f50-a4ca-711c708ba841")

namespace grassland::rhi::backend::d3d12 {
class D3D12CommandContextImpl final : public Object<rhi::ICommandContext, ID3D12CommandContext> {
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
  ~D3D12CommandContextImpl() override;
  static ICommandContext *Create(ID3D12Queue *queue);
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
  ID3D12Queue *OwnerQueue() const;

  // Inputs and ranges have been validated by the recording entry point.
  bool CopyBuffer(ID3D12Buffer *source,
                  uint64_t source_offset,
                  ID3D12Buffer *destination,
                  uint64_t destination_offset,
                  uint64_t size);
  D3D12CommandContextImpl(ID3D12Queue *queue);
  bool ClearColor(ID3D12Image *image, ImageFormat format, const ClearColorValue &value);
  bool PrepareGroups();

  struct BoundGroup {
    ID3D12BindingVersion *version{};
    std::array<D3D12DescriptorPool::Lease *, 2> sources{};
    std::array<D3D12_GPU_DESCRIPTOR_HANDLE, 2> tables{};
    std::array<bool, 2> dirty{};
  };

  std::vector<BoundGroup> groups_;
  std::array<D3D12TransientDescriptorPool::Lease *, 2> descriptor_pages_{};
  std::array<uint32_t, 2> descriptor_used_{};
  bool tables_dirty_{};
  IProgram *program_{};
  ID3D12ProgramBase *native_program_{};
  std::unordered_map<ID3D12TopLevelAccelerationStructure *, std::vector<ObjectPtr<IBottomLevelAccelerationStructure>>>
      as_dependencies_;
  bool compute_active_{};
  ID3D12RayTracingProgram *ray_program_{};
  const PipelineLayoutDesc *binding_layout_{};
  bool rendering_active_{};
  std::vector<ID3D12Buffer *> vertex_buffers_;
  ID3D12Buffer *index_buffer_{};
  bool PrepareDrawBuffers(bool indexed);
  bool viewport_set_{}, scissor_set_{}, index_bound_{};
  bool EndRecording();
  bool TransitionResource(ID3D12Resource *resource, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_STATES default_state);

  struct ResourceState {
    D3D12_RESOURCE_STATES current;
    D3D12_RESOURCE_STATES default_state;
  };

  std::unordered_map<ID3D12Resource *, ResourceState> resource_states_;
  std::vector<std::pair<ID3D12Window *, ID3D12SwapChainImage *>> presents_;
  ID3D12Device *device_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> list_;
  ComPtr<ID3D12GraphicsCommandList4> ray_list_;
  bool submission_failed_{};
  bool reusable_{};
  bool closed_{};

 public:
  bool CmdDrawImGui(IWindow *window, IImage *target) override;

  bool CmdPresentImage(IWindow *window, IImage *frame_img, ColorSpace frame_color_space) override;

 private:
  bool PrepareRendering(const RenderingInfo &info);

  bool RenderingMatches(const IProgram *program) const;

  void InitializeCommandContext(ID3D12Queue *queue);

 public:
  rhi::ISubmissionToken *Submit(const rhi::SubmitInfo &info = {}) override;

  void BindDownloads(rhi::ISubmissionToken *submission) override;

  void ReleaseDownloads() override;

 private:
  // Call before encoding each resource use. Retains each exact object once.
  IDownloadToken *RegisterDownload(ObjectPtr<ID3D12DownloadToken> token);

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
  ID3D12Queue *queue_{};
  std::unordered_map<rhi::IObject *, ObjectPtr<rhi::IObject>> resources_{};
  bool submitted_{};
  std::vector<ObjectPtr<ID3D12DownloadToken>> downloads_{};
};
}  // namespace grassland::rhi::backend::d3d12
