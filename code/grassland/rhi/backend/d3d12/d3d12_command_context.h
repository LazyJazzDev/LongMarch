#pragma once
#include <vector>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_command_context.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
class D3D12Image;
class D3D12Buffer;
class D3D12Program;
class D3D12Window;
class D3D12SwapChainImage;

class D3D12CommandContext final : public CommandContext {
 public:
  ~D3D12CommandContext() override;
  static D3D12CommandContext *Create(D3D12Device *device);
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
  bool CmdClearColor(rhi::IImage *image, const std::array<float, 4> &color) override;
  bool CmdPresent(rhi::IWindow *window) override;
  SubmitResult SubmitNative() override;
  SubmissionStatus WaitForProgress() override;

 private:
  D3D12CommandContext(D3D12Device *device);

  bool PrepareGroups();
  std::vector<BindingGroup::Version *> groups_;
  bool groups_dirty_{true};
  D3D12Program *program_{};
  D3D12Image *render_target_{};
  AttachmentStoreOp store_op_{ATTACHMENT_STORE_OP_STORE};
  std::vector<D3D12Buffer *> vertex_buffers_;
  D3D12Buffer *index_buffer_{};
  bool PrepareDrawBuffers(bool indexed);
  bool viewport_set_{}, scissor_set_{}, index_bound_{};

  bool EndRecording() override;
  bool RenderTarget(D3D12Image *image, D3D12_CPU_DESCRIPTOR_HANDLE &handle);
  std::unordered_map<D3D12Image *, D3D12_CPU_DESCRIPTOR_HANDLE> rtv_leases_;
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
  HANDLE event_{};
  uint64_t value_{};
  bool submission_failed_{};
  bool reusable_{};
  bool closed_{};
};

}  // namespace grassland::rhi::backend::d3d12
