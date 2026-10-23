#include "grassland/rhi/backend/metal/metal_program.h"

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
MetalProgram::MetalProgram(MetalDevice *device, const ProgramDesc &desc)
    : Program(device, desc),
      MetalProgramBase(device->NativeDevice(), desc_.pipeline_layout) {
}

MTL::RenderPipelineState *MetalProgram::NativePipeline() const {
  return pipeline_.get();
}

MTL::DepthStencilState *MetalProgram::NativeDepthStencil() const {
  return depth_stencil_.get();
}

MetalProgram *MetalProgram::Create(MetalDevice *owner_device, const ProgramDesc &desc) {
  auto object = std::unique_ptr<MetalProgram>(new MetalProgram(owner_device, desc));

  if (!object->ValidateLayout() || object->desc_.vertex_bindings.size() > 31 - VertexBufferBase) {
    ReportError(-1, "Invalid Metal graphics program layout.");
    return nullptr;
  }
  for (const auto &binding : object->desc_.vertex_bindings) {
    if (!binding.stride || binding.stride > 2048 || binding.stride % 4) {
      ReportError(-1, "Metal vertex stride must be a multiple of four in [4, 2048].");
      return nullptr;
    }
  }
  for (const auto &attribute : object->desc_.vertex_attributes) {
    if (attribute.location >= 31 || attribute.offset % 4) {
      ReportError(-1, "Invalid Metal vertex attribute location or alignment.");
      return nullptr;
    }
  }
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = owner_device->NativeDevice();
  NS::SharedPtr<MTL::Function> functions[2];
  const IShader *shaders[] = {object->desc_.vertex_shader, object->desc_.fragment_shader};
  for (int i = 0; i < 2; ++i) {
    functions[i] = object->CompileFunction(ShaderCompileDesc(shaders[i]), object->desc_.vertex_attributes);
    if (!functions[i])
      return nullptr;
  }
  auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
  descriptor->setVertexFunction(functions[0].get());
  descriptor->setFragmentFunction(functions[1].get());
  for (size_t i = 0; i < object->desc_.color_formats.size(); ++i) {
    auto attachment = descriptor->colorAttachments()->object(i);
    attachment->setPixelFormat(NativeImageFormat(object->desc_.color_formats[i]));
    const auto value = object->desc_.blend_states.empty() ? BlendState{} : object->desc_.blend_states[i];
    attachment->setBlendingEnabled(value.blend_enable);
    attachment->setSourceRGBBlendFactor(NativeBlendFactor(value.src_color));
    attachment->setDestinationRGBBlendFactor(NativeBlendFactor(value.dst_color));
    attachment->setRgbBlendOperation(NativeBlendOp(value.color_op));
    attachment->setSourceAlphaBlendFactor(NativeBlendFactor(value.src_alpha, true));
    attachment->setDestinationAlphaBlendFactor(NativeBlendFactor(value.dst_alpha, true));
    attachment->setAlphaBlendOperation(NativeBlendOp(value.alpha_op));
  }
  const auto &ds = object->desc_.depth_stencil;
  descriptor->setDepthAttachmentPixelFormat(NativeImageFormat(ds.format));
  if (HasStencil(ds.format))
    descriptor->setStencilAttachmentPixelFormat(NativeImageFormat(ds.format));
  auto depth = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
  depth->setDepthCompareFunction(ds.depth_test ? static_cast<MTL::CompareFunction>(ds.depth_compare)
                                               : MTL::CompareFunctionAlways);
  depth->setDepthWriteEnabled(ds.depth_test && ds.depth_write);
  if (ds.stencil_test) {
    auto setup = [&](const StencilFaceDesc &face) {
      auto d = MTL::StencilDescriptor::alloc()->init();
      d->setStencilCompareFunction(static_cast<MTL::CompareFunction>(face.compare));
      d->setStencilFailureOperation(static_cast<MTL::StencilOperation>(face.fail_op));
      d->setDepthFailureOperation(static_cast<MTL::StencilOperation>(face.depth_fail_op));
      d->setDepthStencilPassOperation(static_cast<MTL::StencilOperation>(face.pass_op));
      d->setReadMask(ds.stencil_read_mask);
      d->setWriteMask(ds.stencil_write_mask);
      return NS::TransferPtr(d);
    };
    auto front = setup(ds.front), back = setup(ds.back);
    depth->setFrontFaceStencil(front.get());
    depth->setBackFaceStencil(back.get());
  }
  object->depth_stencil_ = NS::TransferPtr(device->newDepthStencilState(depth.get()));
  if (!object->depth_stencil_)
    return nullptr;
  auto vertex = MTL::VertexDescriptor::vertexDescriptor();
  for (size_t i = 0; i < object->desc_.vertex_bindings.size(); ++i) {
    auto layout = vertex->layouts()->object(i + VertexBufferBase);
    layout->setStride(object->desc_.vertex_bindings[i].stride);
    layout->setStepFunction(object->desc_.vertex_bindings[i].per_instance ? MTL::VertexStepFunctionPerInstance
                                                                          : MTL::VertexStepFunctionPerVertex);
    layout->setStepRate(1);
  }
  const MTL::VertexFormat formats[] = {MTL::VertexFormatFloat, MTL::VertexFormatFloat2, MTL::VertexFormatFloat3,
                                       MTL::VertexFormatFloat4};
  for (const auto &a : object->desc_.vertex_attributes) {
    auto attribute = vertex->attributes()->object(a.location);
    attribute->setFormat(formats[a.format]);
    attribute->setBufferIndex(a.binding + VertexBufferBase);
    attribute->setOffset(a.offset);
  }
  descriptor->setVertexDescriptor(vertex);
  NS::Error *error = nullptr;
  object->pipeline_ = NS::TransferPtr(device->newRenderPipelineState(descriptor.get(), &error));
  if (!object->pipeline_) {
    ReportMetalError(error, "Metal pipeline creation failed");
    return nullptr;
  }
  return object.release();
}
}  // namespace grassland::rhi::backend::metal
