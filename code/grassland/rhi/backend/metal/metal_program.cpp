#include "grassland/rhi/backend/metal/metal_program.h"

#include <cstring>
#include <spirv_msl.hpp>
#include <unordered_set>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/backend/metal/metal_program_impl.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
MetalProgramImpl::MetalProgramImpl(IMetalDevice *device, const ProgramDesc &desc) {
  InitializeProgram(device, desc);
  InitializeMetalProgramBase(device->NativeDevice());
}

MTL::RenderPipelineState *MetalProgramImpl::NativePipeline() const {
  return pipeline_.get();
}

MTL::DepthStencilState *MetalProgramImpl::NativeDepthStencil() const {
  return depth_stencil_.get();
}

rhi::IProgram *MetalProgramImpl::Create(IMetalDevice *owner_device, const ProgramDesc &desc) {
  auto object = std::unique_ptr<MetalProgramImpl>(new MetalProgramImpl(owner_device, desc));

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
    functions[i] = object->CompileFunction(shaders[i], object->desc_.vertex_attributes);
    if (!functions[i])
      return nullptr;
  }
  auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
  descriptor->setVertexFunction(functions[0].get());
  descriptor->setInputPrimitiveTopology(
      desc.topology == PRIMITIVE_TOPOLOGY_POINT_LIST ? MTL::PrimitiveTopologyClassPoint
      : (desc.topology == PRIMITIVE_TOPOLOGY_LINE_LIST || desc.topology == PRIMITIVE_TOPOLOGY_LINE_STRIP)
          ? MTL::PrimitiveTopologyClassLine
          : MTL::PrimitiveTopologyClassTriangle);
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
  const MTL::VertexFormat formats[] = {MTL::VertexFormatFloat,  MTL::VertexFormatFloat2, MTL::VertexFormatFloat3,
                                       MTL::VertexFormatFloat4, MTL::VertexFormatInt,    MTL::VertexFormatInt2,
                                       MTL::VertexFormatInt3,   MTL::VertexFormatInt4,   MTL::VertexFormatUInt,
                                       MTL::VertexFormatUInt2,  MTL::VertexFormatUInt3,  MTL::VertexFormatUInt4};
  for (const auto &a : object->desc_.vertex_attributes) {
    auto attribute = vertex->attributes()->object(a.location);
    attribute->setFormat(formats[a.type]);
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

rhi::IProgram *IMetalProgram::Create(IMetalDevice *device, const ProgramDesc &desc) {
  return MetalProgramImpl::Create(device, desc);
}

void MetalProgramImpl::InitializeProgram(IBackendDevice *device, const ProgramDesc &desc) {
  device_ = device;
  desc_ = desc;
  vertex_shader_ = ObjectPtr<IShader>::Retain(desc.vertex_shader);
  fragment_shader_ = ObjectPtr<IShader>::Retain(desc.fragment_shader);
}

const ProgramDesc &MetalProgramImpl::GetDesc() const {
  return desc_;
}

IBackendDevice *MetalProgramImpl::Owner() const {
  return device_;
}

bool MetalProgramImpl::ValidateLayout() const {
  if (desc_.topology < PRIMITIVE_TOPOLOGY_TRIANGLE_LIST || desc_.topology > PRIMITIVE_TOPOLOGY_POINT_LIST ||
      desc_.rasterizer.cull_mode < CULL_MODE_NONE || desc_.rasterizer.cull_mode > CULL_MODE_BACK ||
      desc_.rasterizer.front_face < FRONT_FACE_COUNTER_CLOCKWISE ||
      desc_.rasterizer.front_face > FRONT_FACE_CLOCKWISE || desc_.rasterizer.fill_mode < FILL_MODE_SOLID ||
      desc_.rasterizer.fill_mode > FILL_MODE_WIREFRAME)
    return false;
  if (desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  if (!desc_.vertex_shader || !desc_.fragment_shader)
    return false;
  if (desc_.vertex_shader->GetStage() != SHADER_STAGE_VERTEX)
    return false;
  if (desc_.fragment_shader->GetStage() != SHADER_STAGE_FRAGMENT)
    return false;
  if (desc_.color_formats.size() > 8 || (desc_.color_formats.empty() && !IsDepthFormat(desc_.depth_stencil.format)))
    return false;
  for (auto format : desc_.color_formats)
    if (!IsColorFormat(format))
      return false;
  if (!desc_.blend_states.empty() && desc_.blend_states.size() != desc_.color_formats.size())
    return false;
  for (size_t i = 0; i < desc_.blend_states.size(); ++i) {
    if (desc_.blend_states[i].blend_enable &&
        (IsSignedIntegerFormat(desc_.color_formats[i]) || IsUnsignedIntegerFormat(desc_.color_formats[i])))
      return false;
  }
  for (const auto &blend : desc_.blend_states) {
    for (auto factor : {blend.src_color, blend.dst_color, blend.src_alpha, blend.dst_alpha})
      if (factor < BLEND_FACTOR_ZERO || factor > BLEND_FACTOR_ONE_MINUS_DST_ALPHA)
        return false;
    for (auto op : {blend.color_op, blend.alpha_op})
      if (op < BLEND_OP_ADD || op > BLEND_OP_MAX)
        return false;
  }
  const auto &depth = desc_.depth_stencil;
  if (depth.format != IMAGE_FORMAT_UNKNOWN && !IsDepthFormat(depth.format))
    return false;
  if ((depth.depth_test || depth.depth_write || depth.stencil_test) && !IsDepthFormat(depth.format))
    return false;
  if (depth.stencil_test && !HasStencil(depth.format))
    return false;
  if (depth.depth_compare < COMPARE_NEVER || depth.depth_compare > COMPARE_ALWAYS)
    return false;
  for (auto face : {depth.front, depth.back}) {
    if (face.compare < COMPARE_NEVER || face.compare > COMPARE_ALWAYS)
      return false;
    for (auto op : {face.fail_op, face.depth_fail_op, face.pass_op})
      if (op < STENCIL_KEEP || op > STENCIL_DECREMENT_WRAP)
        return false;
  }
  std::unordered_set<uint32_t> locations;
  for (const auto &attribute : desc_.vertex_attributes) {
    if (attribute.binding >= desc_.vertex_bindings.size())
      return false;
    auto size = InputTypeSize(attribute.type);
    auto stride = desc_.vertex_bindings[attribute.binding].stride;
    if (!size || attribute.offset > stride || size > stride - attribute.offset)
      return false;
    if (!locations.insert(attribute.location).second)
      return false;
  }
  return true;
}

void MetalProgramImpl::InitializeMetalProgramBase(MTL::Device *device) {
  native_device_ = device;
}

const PipelineLayoutDesc &MetalProgramImpl::BindingLayout() const {
  return desc_.pipeline_layout;
}

NS::SharedPtr<MTL::Function> MetalProgramImpl::CompileFunction(const IShader *shader,
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

LONGMARCH_RHI_UUID(grassland::rhi::backend::metal::IMetalProgram, "26dfc710-560c-4dff-bfbd-2adc53cd3739")
