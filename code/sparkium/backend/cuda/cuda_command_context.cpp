#include "sparkium/backend/cuda/cuda_command_context.h"

#include "sparkium/backend/cuda/cuda_image.h"
#include "sparkium/backend/cuda/cuda_program.h"
#include "sparkium/backend/cuda/cuda_util.h"
#include "sparkium/backend/device.h"
#ifdef SPARKIUM_OPTIX_ENABLED
#include "sparkium/backend/cuda/optix_acceleration_structure.h"
#endif

namespace sparkium::backend::cuda {

CudaCommandContext::CudaCommandContext(Device *core) : core_(core) {
}

Core *CudaCommandContext::GetCore() const {
  return core_->GraphicsCore();
}

void CudaCommandContext::CmdBindComputeProgram(ComputeProgram *p) {
  program_ = dynamic_cast<CudaProgram *>(p);
  if (!program_)
    throw std::runtime_error("foreign compute compute program");
}

void CudaCommandContext::CmdBindResources(int s, const std::vector<BufferRange> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.buffers[s] = v;
}

void CudaCommandContext::CmdBindResources(int s, const std::vector<Image *> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.images[s] = v;
}

void CudaCommandContext::CmdBindResources(int s, const std::vector<Sampler *> &v, BindPoint b) {
  Check(b);
  ClearSlot(s);
  bindings_.samplers[s] = v;
}

void CudaCommandContext::CmdDispatch(uint32_t x, uint32_t y, uint32_t z) {
  if (!program_)
    throw std::runtime_error("compute dispatch without program");
  auto bindings = bindings_;
  auto *shader = program_->shader;
  commands.emplace_back([=] { shader->Dispatch(bindings, x, y, z); });
}

void CudaCommandContext::CmdClearImage(Image *image, const ClearValue &value) {
  auto *compute = dynamic_cast<CudaImage *>(image);
  if (!compute)
    throw std::runtime_error("foreign compute image");
  commands.emplace_back([=] { compute->Clear(value); });
}

void CudaCommandContext::CmdCopyBuffer(Buffer *d, Buffer *s, uint64_t n, uint64_t od, uint64_t os) {
  commands.emplace_back([=] {
    std::vector<uint8_t> bytes(n);
    s->DownloadData(bytes.data(), n, os);
    d->UploadData(bytes.data(), n, od);
  });
}

void CudaCommandContext::CmdBindProgram(Program *) {
  CudaUnsupported();
}

void CudaCommandContext::CmdBindRayTracingProgram(RayTracingProgram *) {
  CudaUnsupported();
}

void CudaCommandContext::CmdBindVertexBuffers(uint32_t, const std::vector<Buffer *> &, const std::vector<uint64_t> &) {
  CudaUnsupported();
}

void CudaCommandContext::CmdBindIndexBuffer(Buffer *, uint64_t) {
  CudaUnsupported();
}

void CudaCommandContext::CmdBindResources(int slot, AccelerationStructure *as, BindPoint point) {
  Check(point);
#ifdef SPARKIUM_OPTIX_ENABLED
  if (!core_->DeviceRayTracingSupport() || !dynamic_cast<OptixAccelerationStructure *>(as))
    throw std::invalid_argument("OptiX requires an OptiX acceleration structure");
  ClearSlot(slot);
  bindings_.acceleration_structures[slot] = as;
#else
  CudaUnsupported();
#endif
}

void CudaCommandContext::CmdBeginRendering(const std::vector<Image *> &, Image *) {
  CudaUnsupported();
}

void CudaCommandContext::CmdEndRendering() {
  CudaUnsupported();
}

void CudaCommandContext::CmdSetViewport(const Viewport &) {
  CudaUnsupported();
}

void CudaCommandContext::CmdSetScissor(const Scissor &) {
  CudaUnsupported();
}

void CudaCommandContext::CmdSetPrimitiveTopology(PrimitiveTopology) {
  CudaUnsupported();
}

void CudaCommandContext::CmdDraw(uint32_t, uint32_t, int32_t, uint32_t) {
  CudaUnsupported();
}

void CudaCommandContext::CmdDrawIndexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) {
  CudaUnsupported();
}

void CudaCommandContext::CmdPresent(Window *, Image *) {
  CudaUnsupported();
}

void CudaCommandContext::CmdDispatchRays(uint32_t, uint32_t, uint32_t) {
  CudaUnsupported();
}

void CudaCommandContext::Check(BindPoint b) {
  if (b != BIND_POINT_COMPUTE)
    CudaUnsupported();
}

void CudaCommandContext::ClearSlot(int s) {
  bindings_.buffers.erase(s);
  bindings_.images.erase(s);
  bindings_.samplers.erase(s);
  bindings_.acceleration_structures.erase(s);
}

}  // namespace sparkium::backend::cuda
