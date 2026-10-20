#pragma once
#include <array>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IBuffer;
class IProgram;
class IImage;

struct RenderingInfo {
  IImage *color_attachment{};
  AttachmentLoadOp load_op{ATTACHMENT_LOAD_OP_LOAD};
  AttachmentStoreOp store_op{ATTACHMENT_STORE_OP_STORE};
  std::array<float, 4> clear_color{};
};

struct Viewport {
  float x{}, y{}, width{}, height{}, min_depth{}, max_depth{1};
};

struct Scissor {
  int32_t x{}, y{};
  uint32_t width{}, height{};
};
class IWindow;

class ICommandContext : public IObject {
 public:
  // Cmd calls encode native commands immediately and retain the exact resources
  // they use. Parameter values are consumed during the call, not during Submit.
  // Callers must not use an image concurrently from multiple command contexts.
  // Submit restores every used resource to its backend default state and ends
  // recording; a context cannot be recorded or submitted again.
  // Upload snapshots the CPU bytes now; staging storage is leased through GPU completion.
  // Bind a finalized Program before vertex buffers; rebind buffers after changing its layout.
  // Begin/End delimit an explicit, non-nested rendering scope. Draw requires a scope.
  // Upload, copy, standalone clear, Present and Submit require no active scope.
  // Begin consumes attachment operations; backend attachment leases remain automatic.
  virtual bool CmdUploadBuffer(IBuffer *buffer, uint64_t offset, const void *data, uint64_t size) = 0;
  virtual bool CmdCopyBuffer(IBuffer *source,
                             uint64_t source_offset,
                             IBuffer *destination,
                             uint64_t destination_offset,
                             uint64_t size) = 0;
  virtual bool CmdBindProgram(IProgram *program) = 0;
  virtual bool CmdBeginRendering(const RenderingInfo &info) = 0;
  virtual bool CmdEndRendering() = 0;
  virtual bool CmdBindVertexBuffer(uint32_t binding, IBuffer *buffer, uint64_t offset = 0) = 0;
  virtual bool CmdBindIndexBuffer(IBuffer *buffer, IndexFormat format = INDEX_FORMAT_UINT32, uint64_t offset = 0) = 0;
  virtual bool CmdSetViewport(const Viewport &viewport) = 0;
  virtual bool CmdSetScissor(const Scissor &scissor) = 0;
  virtual bool CmdDraw(uint32_t vertex_count,
                       uint32_t instance_count = 1,
                       uint32_t first_vertex = 0,
                       uint32_t first_instance = 0) = 0;
  virtual bool CmdDrawIndexed(uint32_t index_count,
                              uint32_t instance_count = 1,
                              uint32_t first_index = 0,
                              int32_t vertex_offset = 0,
                              uint32_t first_instance = 0) = 0;
  virtual bool CmdClearColor(IImage *image, const std::array<float, 4> &color) = 0;
  // Presents the target window's next image fixed at this call. Multiple windows
  // may be presented by one context; a preceding clear is not required.
  virtual bool CmdPresent(IWindow *window) = 0;

 protected:
  ~ICommandContext() override;
};

}  // namespace grassland::rhi
