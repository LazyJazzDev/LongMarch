#include "grassland/rhi/backend/metal/metal_compute_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"

namespace grassland::rhi::backend::metal {
MetalComputeProgram::MetalComputeProgram(MetalDevice *device, const ComputeProgramDesc &desc)
    : ComputeProgram(device, desc),
      MetalProgramBase(device->NativeDevice(), desc_.pipeline_layout) {
}

MTL::ComputePipelineState *MetalComputeProgram::NativePipeline() const {
  return pipeline_.get();
}

MTL::Size MetalComputeProgram::ThreadGroupSize() const {
  return threads_;
}

MetalComputeProgram *MetalComputeProgram::Create(MetalDevice *owner_device, const ComputeProgramDesc &desc) {
  auto object = std::unique_ptr<MetalComputeProgram>(new MetalComputeProgram(owner_device, desc));
  if (!object->ValidateLayout())
    return nullptr;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::array<uint32_t, 3> threads{};
  auto function = object->CompileFunction(ShaderCompileDesc(object->desc_.compute_shader), {}, &threads);
  if (!function)
    return nullptr;
  auto device = owner_device->NativeDevice();
  const auto limits = device->maxThreadsPerThreadgroup();
  if (!threads[0] || !threads[1] || !threads[2] || threads[0] > limits.width || threads[1] > limits.height ||
      threads[2] > limits.depth)
    return nullptr;
  NS::Error *error = nullptr;
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  if (!pipeline) {
    ReportMetalError(error, "Metal compute pipeline creation failed");
    return nullptr;
  }
  if (uint64_t(threads[0]) * threads[1] * threads[2] > pipeline->maxTotalThreadsPerThreadgroup() ||
      pipeline->staticThreadgroupMemoryLength() > device->maxThreadgroupMemoryLength())
    return nullptr;
  object->threads_ = MTL::Size::Make(threads[0], threads[1], threads[2]);
  object->pipeline_ = std::move(pipeline);
  return object.release();
}
}  // namespace grassland::rhi::backend::metal
