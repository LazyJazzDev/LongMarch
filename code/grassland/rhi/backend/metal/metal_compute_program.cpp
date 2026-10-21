#include "grassland/rhi/backend/metal/metal_compute_program.h"

#include "grassland/rhi/backend/metal/metal_device.h"

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

bool MetalComputeProgram::Finalize() {
  if (finalized_)
    return true;
  if (!ValidateLayout())
    return false;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::array<uint32_t, 3> threads{};
  auto function = CompileFunction(desc_.compute_shader->GetDesc(), {}, &threads);
  if (!function)
    return false;
  auto device = static_cast<MetalDevice *>(device_)->NativeDevice();
  const auto limits = device->maxThreadsPerThreadgroup();
  if (!threads[0] || !threads[1] || !threads[2] || threads[0] > limits.width || threads[1] > limits.height ||
      threads[2] > limits.depth)
    return false;
  NS::Error *error = nullptr;
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  if (!pipeline) {
    ReportError(-1, error ? error->localizedDescription()->utf8String() : "Metal compute pipeline creation failed.");
    return false;
  }
  if (uint64_t(threads[0]) * threads[1] * threads[2] > pipeline->maxTotalThreadsPerThreadgroup() ||
      pipeline->staticThreadgroupMemoryLength() > device->maxThreadgroupMemoryLength())
    return false;
  threads_ = MTL::Size::Make(threads[0], threads[1], threads[2]);
  pipeline_ = std::move(pipeline);
  finalized_ = true;
  return true;
}
}  // namespace grassland::rhi::backend::metal
