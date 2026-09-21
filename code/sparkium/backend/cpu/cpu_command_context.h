#pragma once
#include "grassland/graphics/command_context.h"
#include "sparkium/backend/cpu/cpu_bindings.h"

namespace sparkium::backend {
class Device;
}

namespace sparkium::backend::cpu {
using namespace grassland;
using namespace grassland::graphics;

class CpuProgram;

class CpuCommandContext final : public CommandContext {
 public:
  explicit CpuCommandContext(Device *core);

  Core *GetCore() const override;

  // Like the graphics backends, retain compatible resource bindings across
  // program switches. The shared hierarchical prefix scan relies on this.
  void CmdBindComputeProgram(ComputeProgram *p) override;

  void CmdBindResources(int s, const std::vector<BufferRange> &v, BindPoint b) override;

  void CmdBindResources(int s, const std::vector<Image *> &v, BindPoint b) override;

  void CmdBindResources(int s, const std::vector<Sampler *> &v, BindPoint b) override;

  void CmdDispatch(uint32_t x, uint32_t y, uint32_t z) override;

  void CmdClearImage(Image *image, const ClearValue &value) override;

  void CmdCopyBuffer(Buffer *d, Buffer *s, uint64_t n, uint64_t od, uint64_t os) override;

  void CmdBindProgram(Program *) override;

  void CmdBindRayTracingProgram(RayTracingProgram *) override;

  void CmdBindVertexBuffers(uint32_t, const std::vector<Buffer *> &, const std::vector<uint64_t> &) override;

  void CmdBindIndexBuffer(Buffer *, uint64_t) override;

  void CmdBindResources(int slot, AccelerationStructure *as, BindPoint point) override;

  void CmdBeginRendering(const std::vector<Image *> &, Image *) override;

  void CmdEndRendering() override;

  void CmdSetViewport(const Viewport &) override;

  void CmdSetScissor(const Scissor &) override;

  void CmdSetPrimitiveTopology(PrimitiveTopology) override;

  void CmdDraw(uint32_t, uint32_t, int32_t, uint32_t) override;

  void CmdDrawIndexed(uint32_t, uint32_t, uint32_t, int32_t, uint32_t) override;

  void CmdPresent(Window *, Image *) override;

  void CmdDispatchRays(uint32_t, uint32_t, uint32_t) override;

  std::vector<std::function<void()>> commands;

 private:
  void Check(BindPoint b);

  void ClearSlot(int s);

  Device *core_;
  CpuProgram *program_{};
  CpuBindings bindings_;
};

}  // namespace sparkium::backend::cpu
