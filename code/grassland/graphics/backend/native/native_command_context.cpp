#include "native_command_context.h"

#include "native_core.h"
#include "native_image.h"
#include "native_program.h"
#include "native_util.h"
#ifdef LONGMARCH_OPTIX_ENABLED
#include "native_acceleration_structure.h"
#endif

namespace grassland::graphics::backend {

NativeCommandContext::NativeCommandContext(NativeCore *core) : core_(core) {
}

Core *NativeCommandContext::GetCore() const {
  return core_;
}

void NativeCommandContext::CmdBindComputeProgram(ComputeProgram *p) {
  program_ = dynamic_cast<NativeProgram *>(p);
  if (!program_)
    throw std::runtime_error("foreign native compute program");
}

void NativeCommandContext::CmdBindResources(int s, const std::vector<BufferRange> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.buffers[s] = v;
}

void NativeCommandContext::CmdBindResources(int s, const std::vector<Image *> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.images[s] = v;
}

void NativeCommandContext::CmdBindResources(int s, const std::vector<Sampler *> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.samplers[s] = v;
}

void NativeCommandContext::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (!program_)
    throw std::runtime_error("native dispatch without program");
  auto bindings = bindings_;
  auto *shader = program_->shader;
  commands.emplace_back([=] { shader->Dispatch(bindings, x, y, z); });
}

void NativeCommandContext::CmdClearImage(Image *image, const ClearValue &value) {
  auto *native = dynamic_cast<NativeImage *>(image);
  if (!native)
    throw std::runtime_error("foreign native image");
  commands.emplace_back([=] { native->Clear(value); });
}

void NativeCommandContext::CmdCopyBuffer(Buffer *d, Buffer *s, uint64_t n, uint64_t od, uint64_t os) {
  commands.emplace_back([=] {
    std::vector<uint8_t> bytes(n);
    s->DownloadData(bytes.data(), n, os);
    d->UploadData(bytes.data(), n, od);
  });
}

void NativeCommandContext::CmdBindProgram(Program *) {
  NativeUnsupported();
}

void NativeCommandContext::CmdBindRayTracingProgram(RayTracingProgram *) {
  NativeUnsupported();
}

void NativeCommandContext::CmdBindVertexBuffers(uint32_t,
                                                const std::vector<Buffer *> &,
                                                const std::vector<uint64_t> &) {
  NativeUnsupported();
}

void NativeCommandContext::CmdBindIndexBuffer(Buffer *, uint64_t) {
  NativeUnsupported();
}

void NativeCommandContext::CmdBindResources(int slot, AccelerationStructure *as, BindPoint point) {
  Check(point);
#ifdef LONGMARCH_OPTIX_ENABLED
  if (!core_->DeviceRayTracingSupport() || !dynamic_cast<OptixAccelerationStructure *>(as))
    throw std::invalid_argument("OptiX requires a native acceleration structure");
  ClearSlot(slot);
  bindings_.acceleration_structures[slot] = as;
#else
  NativeUnsupported();
#endif
}

void NativeCommandContext::CmdBeginRendering(const std::vector<Image *> &, Image *) {
  NativeUnsupported();
}

void NativeCommandContext::CmdEndRendering() {
  NativeUnsupported();
}

void NativeCommandContext::CmdSetViewport(const Viewport &) {
  NativeUnsupported();
}

void NativeCommandContext::CmdSetScissor(const Scissor &) {
  NativeUnsupported();
}

void NativeCommandContext::CmdSetPrimitiveTopology(PrimitiveTopology) {
  NativeUnsupported();
}

void NativeCommandContext::CmdDraw(uint32_t, uint32_t, int32_t, uint32_t) {
  NativeUnsupported();
}

void NativeCommandContext::CmdDrawIndexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) {
  NativeUnsupported();
}

void NativeCommandContext::CmdPresent(Window *, Image *) {
  NativeUnsupported();
}

void NativeCommandContext::CmdDispatchRays(uint32_t, uint32_t, uint32_t) {
  NativeUnsupported();
}

void NativeCommandContext::Check(BindPoint b) {
  if (b != BIND_POINT_COMPUTE)
    NativeUnsupported();
}

void NativeCommandContext::ClearSlot(int s) {
  bindings_.buffers.erase(s);
  bindings_.images.erase(s);
  bindings_.samplers.erase(s);
  bindings_.acceleration_structures.erase(s);
}

}  // namespace grassland::graphics::backend
