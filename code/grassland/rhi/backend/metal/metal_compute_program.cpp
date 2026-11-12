#include "grassland/rhi/backend/metal/metal_compute_program.h"

#include <cstring>
#include <spirv_msl.hpp>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_compute_program_impl.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"

namespace grassland::rhi::backend::metal {
MetalComputeProgramImpl::MetalComputeProgramImpl(IMetalDevice *device, const ComputeProgramDesc &desc) {
  InitializeComputeProgram(device, desc);
  InitializeMetalProgramBase(device->NativeDevice());
}

MTL::ComputePipelineState *MetalComputeProgramImpl::NativePipeline() const {
  return pipeline_.get();
}

MTL::Size MetalComputeProgramImpl::ThreadGroupSize() const {
  return threads_;
}

rhi::IComputeProgram *MetalComputeProgramImpl::Create(IMetalDevice *owner_device, const ComputeProgramDesc &desc) {
  auto object = std::unique_ptr<MetalComputeProgramImpl>(new MetalComputeProgramImpl(owner_device, desc));
  if (!object->ValidateLayout())
    return nullptr;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  std::array<uint32_t, 3> threads{};
  auto function = object->CompileFunction(object->desc_.compute_shader, {}, &threads);
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

rhi::IComputeProgram *IMetalComputeProgram::Create(IMetalDevice *device, const ComputeProgramDesc &desc) {
  return MetalComputeProgramImpl::Create(device, desc);
}

void MetalComputeProgramImpl::InitializeComputeProgram(IMetalDevice *device, const ComputeProgramDesc &desc) {
  device_ = device;
  desc_ = desc;
  shader_ = ObjectPtr<IShader>::Retain(desc.compute_shader);
}

const ComputeProgramDesc &MetalComputeProgramImpl::GetDesc() const {
  return desc_;
}

IMetalDevice *MetalComputeProgramImpl::Owner() const {
  return device_;
}

bool MetalComputeProgramImpl::ValidateLayout() const {
  if (!desc_.compute_shader || desc_.compute_shader->GetStage() != SHADER_STAGE_COMPUTE ||
      desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}

void MetalComputeProgramImpl::InitializeMetalProgramBase(MTL::Device *device) {
  native_device_ = device;
}

const PipelineLayoutDesc &MetalComputeProgramImpl::BindingLayout() const {
  return desc_.pipeline_layout;
}

NS::SharedPtr<MTL::Function> MetalComputeProgramImpl::CompileFunction(const IShader *shader,
                                                                      const std::vector<InputAttributeDesc> &attributes,
                                                                      std::array<uint32_t, 3> *threads) const {
  for (const auto &group : (desc_.pipeline_layout).groups)
    if (!ValidateMetalBindingLayout(native_device_, group))
      return {};
  std::string code, entry;
  if (!CompileMetal(shader, desc_.pipeline_layout, attributes, code, entry, threads))
    return {};
  NS::Error *error = nullptr;
  auto library = NS::TransferPtr(
      native_device_->newLibrary(NS::String::string(code.c_str(), NS::UTF8StringEncoding), nullptr, &error));
  if (!library) {
    ReportMetalError(error, "Metal library compilation failed");
    return {};
  }
  auto function = NS::TransferPtr(library->newFunction(NS::String::string(entry.c_str(), NS::UTF8StringEncoding)));
  if (!function)
    ReportError(-1, "Metal shader entry point not found.");
  return function;
}
}  // namespace grassland::rhi::backend::metal
