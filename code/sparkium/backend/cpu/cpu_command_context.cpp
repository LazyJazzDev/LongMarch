#include "sparkium/backend/cpu/cpu_command_context.h"

#include "sparkium/backend/cpu/cpu_image.h"
#include "sparkium/backend/cpu/cpu_program.h"
#include "sparkium/backend/cpu/cpu_util.h"
#include "sparkium/backend/device.h"

namespace sparkium::backend::cpu {

CpuCommandContext::CpuCommandContext(Device *core) : core_(core) {
}

Core *CpuCommandContext::GetCore() const {
  return core_->GraphicsCore();
}

void CpuCommandContext::CmdBindComputeProgram(ComputeProgram *p) {
  program_ = dynamic_cast<CpuProgram *>(p);
  if (!program_)
    throw std::runtime_error("foreign compute compute program");
}

void CpuCommandContext::CmdBindResources(int s, const std::vector<BufferRange> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.buffers[s] = v;
}

void CpuCommandContext::CmdBindResources(int s, const std::vector<Image *> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.images[s] = v;
}

void CpuCommandContext::CmdBindResources(int s, const std::vector<Sampler *> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.samplers[s] = v;
}

void CpuCommandContext::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (!program_)
    throw std::runtime_error("compute dispatch without program");
  auto bindings = bindings_;
  auto *shader = program_->shader;
  commands.emplace_back([=] { shader->Dispatch(bindings, x, y, z); });
}

void CpuCommandContext::CmdClearImage(Image *image, const ClearValue &value) {
  auto *compute = dynamic_cast<CpuImage *>(image);
  if (!compute)
    throw std::runtime_error("foreign compute image");
  commands.emplace_back([=] { compute->Clear(value); });
}

void CpuCommandContext::CmdCopyBuffer(Buffer *d, Buffer *s, uint64_t n, uint64_t od, uint64_t os) {
  commands.emplace_back([=] {
    std::vector<uint8_t> bytes(n);
    s->DownloadData(bytes.data(), n, os);
    d->UploadData(bytes.data(), n, od);
  });
}

void CpuCommandContext::CmdBindProgram(Program *) {
  CpuUnsupported();
}

void CpuCommandContext::CmdBindRayTracingProgram(RayTracingProgram *) {
  CpuUnsupported();
}

void CpuCommandContext::CmdBindVertexBuffers(uint32_t, const std::vector<Buffer *> &, const std::vector<uint64_t> &) {
  CpuUnsupported();
}

void CpuCommandContext::CmdBindIndexBuffer(Buffer *, uint64_t) {
  CpuUnsupported();
}

void CpuCommandContext::CmdBindResources(int slot, AccelerationStructure *as, BindPoint point) {
  Check(point);
  CpuUnsupported();
}

void CpuCommandContext::CmdBeginRendering(const std::vector<Image *> &, Image *) {
  CpuUnsupported();
}

void CpuCommandContext::CmdEndRendering() {
  CpuUnsupported();
}

void CpuCommandContext::CmdSetViewport(const Viewport &) {
  CpuUnsupported();
}

void CpuCommandContext::CmdSetScissor(const Scissor &) {
  CpuUnsupported();
}

void CpuCommandContext::CmdSetPrimitiveTopology(PrimitiveTopology) {
  CpuUnsupported();
}

void CpuCommandContext::CmdDraw(uint32_t, uint32_t, int32_t, uint32_t) {
  CpuUnsupported();
}

void CpuCommandContext::CmdDrawIndexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) {
  CpuUnsupported();
}

void CpuCommandContext::CmdPresent(Window *, Image *) {
  CpuUnsupported();
}

void CpuCommandContext::CmdDispatchRays(uint32_t, uint32_t, uint32_t) {
  CpuUnsupported();
}

void CpuCommandContext::Check(BindPoint b) {
  if (b != BIND_POINT_COMPUTE)
    CpuUnsupported();
}

void CpuCommandContext::ClearSlot(int s) {
  bindings_.buffers.erase(s);
  bindings_.images.erase(s);
  bindings_.samplers.erase(s);
}

}  // namespace sparkium::backend::cpu
