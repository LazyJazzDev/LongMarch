#pragma once
#include "grassland/rhi/backend/backend_command_context_state.h"
#include "grassland/rhi/backend/d3d12/d3d12_command_context.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12CommandContextImpl final : public Object<ID3D12CommandContext> {
 public:
  LONGMARCH_RHI_FINAL_QUERY(rhi::ICommandContext, ID3D12CommandContext, CommandContext, rhi::ICommandContext)
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
  static ID3D12CommandContext *Create(ID3D12Queue *queue);
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
  D3D12CommandContextImpl(ID3D12Queue *queue);
  bool ClearColor(D3D12Image *image, const ClearColorValue &value);
  bool PrepareGroups();

  struct BoundGroup {
    D3D12BindingGroup::Version *version{};
    std::array<D3D12DescriptorPool::Lease *, 2> sources{};
    std::array<D3D12_GPU_DESCRIPTOR_HANDLE, 2> tables{};
    std::array<bool, 2> dirty{};
  };

  std::vector<BoundGroup> groups_;
  std::array<D3D12TransientDescriptorPool::Lease *, 2> descriptor_pages_{};
  std::array<uint32_t, 2> descriptor_used_{};
  bool tables_dirty_{};
  ID3D12Program *program_{};
  D3D12ProgramBase *native_program_{};
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
  bool EndRecording() override;
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

 private:
  CommandContextState state_CommandContext_;

  CommandContextState &StateCommandContext() override {
    return state_CommandContext_;
  }

  const CommandContextState &StateCommandContext() const override {
    return state_CommandContext_;
  }
};
}  // namespace grassland::rhi::backend::d3d12
