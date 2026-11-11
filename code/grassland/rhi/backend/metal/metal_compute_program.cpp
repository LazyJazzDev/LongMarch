#include "grassland/rhi/backend/metal/metal_compute_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_compute_program_impl.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"

namespace grassland::rhi::backend::metal {
MetalComputeProgramImpl::MetalComputeProgramImpl(IMetalDevice *device, const ComputeProgramDesc &desc) {
  InitializeComputeProgram(device, desc);
  InitializeMetalProgramBase(device->NativeDevice(), StateComputeProgram().desc_.pipeline_layout);
}

MTL::ComputePipelineState *MetalComputeProgramImpl::NativePipeline() const {
  return pipeline_.get();
}

MTL::Size MetalComputeProgramImpl::ThreadGroupSize() const {
  return threads_;
}

IMetalComputeProgram *MetalComputeProgramImpl::Create(IMetalDevice *owner_device, const ComputeProgramDesc &desc) {
  auto object = std::unique_ptr<MetalComputeProgramImpl>(new MetalComputeProgramImpl(owner_device, desc));
  if (!object->ValidateLayout())
    return nullptr;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::array<uint32_t, 3> threads{};
  auto function = object->CompileFunction(object->StateComputeProgram().desc_.compute_shader, {}, &threads);
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

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalComputeProgram, "848b485d-2208-4d1d-86ae-cadee6182bfe")

namespace grassland::rhi::backend::metal {
IMetalComputeProgram *IMetalComputeProgram::Create(IMetalDevice *device, const ComputeProgramDesc &desc) {
  return MetalComputeProgramImpl::Create(device, desc);
}
}  // namespace grassland::rhi::backend::metal
