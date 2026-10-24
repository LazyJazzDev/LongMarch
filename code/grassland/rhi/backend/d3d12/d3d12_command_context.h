#pragma once
#include <vector>

#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_binding_group.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Image;
class D3D12Buffer;
class D3D12Program;
class D3D12RayTracingProgram;
class D3D12ProgramBase;
class D3D12Window;
class D3D12TopLevelAccelerationStructure;
class D3D12SwapChainImage;

class D3D12CommandContext final : public CommandContext {
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
  ~D3D12CommandContext() override;
  static D3D12CommandContext *Create(D3D12Device *device);
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
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  D3D12CommandContext(D3D12Device *device);

  bool ClearColor(D3D12Image *image, const ClearValue &value);
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
  D3D12Program *program_{};
  D3D12ProgramBase *native_program_{};
  std::unordered_map<D3D12TopLevelAccelerationStructure *, std::vector<ObjectPtr<IBottomLevelAccelerationStructure>>>
      as_dependencies_;
  bool compute_active_{};
  D3D12RayTracingProgram *ray_program_{};
  const PipelineLayoutDesc *binding_layout_{};
  bool rendering_active_{};
  std::vector<D3D12Buffer *> vertex_buffers_;
  D3D12Buffer *index_buffer_{};
  bool PrepareDrawBuffers(bool indexed);
  bool viewport_set_{}, scissor_set_{}, index_bound_{};

  bool EndRecording() override;
  bool TransitionResource(ID3D12Resource *resource, D3D12_RESOURCE_STATES state, D3D12_RESOURCE_STATES default_state);

  struct ResourceState {
    D3D12_RESOURCE_STATES current;
    D3D12_RESOURCE_STATES default_state;
  };

  std::unordered_map<ID3D12Resource *, ResourceState> resource_states_;
  std::vector<std::pair<D3D12Window *, D3D12SwapChainImage *>> presents_;
  D3D12Device *device_;
  ComPtr<ID3D12CommandAllocator> allocator_;
  ComPtr<ID3D12GraphicsCommandList> list_;
  ComPtr<ID3D12GraphicsCommandList4> ray_list_;
  HANDLE event_{};
  uint64_t value_{};
  bool submission_failed_{};
  bool reusable_{};
  bool closed_{};
};

}  // namespace grassland::rhi::backend::d3d12
