#include "grassland/graphics/backend/webgpu/webgpu_program.h"

#include <stdexcept>

#include "grassland/graphics/backend/webgpu/webgpu_core.h"
#include "grassland/graphics/backend/webgpu/webgpu_shader.h"

namespace grassland::graphics::backend {

namespace {
void AddBinding(std::vector<WebGPUBinding> &bindings, ResourceType type, int count) {
  if (count != 1)
    throw std::invalid_argument("WebGPU supports one resource per binding slot");
  if (type == RESOURCE_TYPE_ACCELERATION_STRUCTURE || type == RESOURCE_TYPE_WRITABLE_IMAGE)
    throw std::invalid_argument("WebGPU backend does not support this resource type");
  bindings.push_back({type, count});
}

wgpu::BlendFactor Factor(BlendFactor factor) {
  static const wgpu::BlendFactor factors[] = {wgpu::BlendFactor::Zero,     wgpu::BlendFactor::One,
                                              wgpu::BlendFactor::Src,      wgpu::BlendFactor::OneMinusSrc,
                                              wgpu::BlendFactor::Dst,      wgpu::BlendFactor::OneMinusDst,
                                              wgpu::BlendFactor::SrcAlpha, wgpu::BlendFactor::OneMinusSrcAlpha,
                                              wgpu::BlendFactor::DstAlpha, wgpu::BlendFactor::OneMinusDstAlpha};
  return factors[factor];
}

wgpu::BlendOperation Operation(BlendOp op) {
  static const wgpu::BlendOperation ops[] = {wgpu::BlendOperation::Add, wgpu::BlendOperation::Subtract,
                                             wgpu::BlendOperation::ReverseSubtract, wgpu::BlendOperation::Min,
                                             wgpu::BlendOperation::Max};
  return ops[op];
}

wgpu::VertexFormat Format(InputType type) {
  static const wgpu::VertexFormat formats[] = {
      wgpu::VertexFormat::Uint32,   wgpu::VertexFormat::Sint32,   wgpu::VertexFormat::Float32,
      wgpu::VertexFormat::Uint32x2, wgpu::VertexFormat::Sint32x2, wgpu::VertexFormat::Float32x2,
      wgpu::VertexFormat::Uint32x3, wgpu::VertexFormat::Sint32x3, wgpu::VertexFormat::Float32x3,
      wgpu::VertexFormat::Uint32x4, wgpu::VertexFormat::Sint32x4, wgpu::VertexFormat::Float32x4};
  return formats[type];
}

wgpu::PrimitiveTopology Topology(PrimitiveTopology topology) {
  static const wgpu::PrimitiveTopology topologies[] = {
      wgpu::PrimitiveTopology::TriangleList, wgpu::PrimitiveTopology::TriangleStrip, wgpu::PrimitiveTopology::LineList,
      wgpu::PrimitiveTopology::LineStrip, wgpu::PrimitiveTopology::PointList};
  return topologies[topology];
}

wgpu::PipelineLayout CreatePipelineLayout(WebGPUCore *core, const std::vector<wgpu::BindGroupLayout> &layouts) {
  wgpu::PipelineLayoutDescriptor descriptor{};
  descriptor.bindGroupLayoutCount = layouts.size();
  descriptor.bindGroupLayouts = layouts.data();
  return core->Device().CreatePipelineLayout(&descriptor);
}
}  // namespace

std::vector<wgpu::BindGroupLayout> CreateBindGroupLayouts(WebGPUCore *core,
                                                          const std::vector<WebGPUBinding> &bindings,
                                                          wgpu::ShaderStage visibility) {
  // Textures pair with filtering samplers only when a program declares samplers;
  // otherwise shaders read texels, which float32 formats also permit.
  bool samplers = false;
  for (const auto &binding : bindings)
    samplers |= binding.type == RESOURCE_TYPE_SAMPLER;
  std::vector<wgpu::BindGroupLayout> layouts;
  for (const auto &binding : bindings) {
    wgpu::BindGroupLayoutEntry entry{};
    entry.binding = 0;
    entry.visibility = visibility;
    switch (binding.type) {
      case RESOURCE_TYPE_UNIFORM_BUFFER:
        entry.buffer.type = wgpu::BufferBindingType::Uniform;
        break;
      case RESOURCE_TYPE_STORAGE_BUFFER:
        entry.buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;
        break;
      case RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER:
        entry.buffer.type = wgpu::BufferBindingType::Storage;
        // Vertex shaders cannot write storage buffers.
        entry.visibility = visibility & ~wgpu::ShaderStage::Vertex;
        break;
      case RESOURCE_TYPE_IMAGE:
        entry.texture.sampleType =
            samplers ? wgpu::TextureSampleType::Float : wgpu::TextureSampleType::UnfilterableFloat;
        entry.texture.viewDimension = wgpu::TextureViewDimension::e2D;
        break;
      case RESOURCE_TYPE_SAMPLER:
        entry.sampler.type = wgpu::SamplerBindingType::Filtering;
        break;
      default:
        throw std::invalid_argument("WebGPU backend does not support this resource type");
    }
    wgpu::BindGroupLayoutDescriptor descriptor{};
    descriptor.entryCount = 1;
    descriptor.entries = &entry;
    layouts.push_back(core->Device().CreateBindGroupLayout(&descriptor));
  }
  return layouts;
}

void WebGPUComputeProgram::AddResourceBinding(ResourceType type, int count) {
  AddBinding(bindings_, type, count);
}

void WebGPUComputeProgram::Finalize() {
  if (!shader_)
    throw std::invalid_argument("missing WebGPU compute shader");
  layouts_ = CreateBindGroupLayouts(core_, bindings_, wgpu::ShaderStage::Compute);
  wgpu::ComputePipelineDescriptor descriptor{};
  descriptor.layout = CreatePipelineLayout(core_, layouts_);
  const auto entry = shader_->EntryPoint();
  descriptor.compute.module = shader_->Module();
  descriptor.compute.entryPoint = {entry.data(), entry.size()};
  pipeline_ = core_->Device().CreateComputePipeline(&descriptor);
}

WebGPUProgram::WebGPUProgram(WebGPUCore *core, const std::vector<ImageFormat> &colors, ImageFormat depth)
    : core_(core),
      colors_(colors),
      depth_(depth) {
}

void WebGPUProgram::AddInputBinding(uint32_t stride, bool per_instance) {
  vertex_bindings_.push_back({stride, per_instance, {}});
}

void WebGPUProgram::AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) {
  if (binding >= vertex_bindings_.size() || type < 0 || type > INPUT_TYPE_FLOAT4)
    throw std::out_of_range("WebGPU vertex attribute");
  // Slang assigns @location in declaration order, as the attribute index on other backends.
  vertex_bindings_[binding].attributes.push_back({nullptr, Format(type), offset, attributes_++});
}

void WebGPUProgram::AddResourceBinding(ResourceType type, int count) {
  AddBinding(bindings_, type, count);
}

void WebGPUProgram::SetBlendState(int target, const BlendState &state) {
  if (target < 0 || target >= int(colors_.size()))
    throw std::out_of_range("WebGPU blend target");
  blends_[target] = state;
}

void WebGPUProgram::BindShader(Shader *shader, ShaderType type) {
  auto webgpu = dynamic_cast<WebGPUShader *>(shader);
  if (!webgpu)
    throw std::invalid_argument("expected WebGPUShader");
  if (type == SHADER_TYPE_VERTEX)
    vertex_ = webgpu;
  else if (type == SHADER_TYPE_PIXEL)
    fragment_ = webgpu;
  else
    throw std::runtime_error("WebGPU does not support geometry shaders");
}

void WebGPUProgram::Finalize() {
  if (!vertex_)
    throw std::invalid_argument("missing WebGPU vertex shader");
  layouts_ = CreateBindGroupLayouts(core_, bindings_, wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment);
  layout_ = CreatePipelineLayout(core_, layouts_);
  pipelines_.clear();
  // Build the common topology now so shader and layout errors surface at Finalize.
  Pipeline(PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
}

const wgpu::RenderPipeline &WebGPUProgram::Pipeline(PrimitiveTopology topology) {
  if (topology < 0 || topology > PRIMITIVE_TOPOLOGY_POINT_LIST)
    throw std::invalid_argument("primitive topology");
  auto found = pipelines_.find(topology);
  if (found != pipelines_.end())
    return found->second;
  std::vector<wgpu::VertexBufferLayout> buffers;
  for (const auto &binding : vertex_bindings_) {
    wgpu::VertexBufferLayout layout{};
    layout.arrayStride = binding.stride;
    layout.stepMode = binding.per_instance ? wgpu::VertexStepMode::Instance : wgpu::VertexStepMode::Vertex;
    layout.attributeCount = binding.attributes.size();
    layout.attributes = binding.attributes.data();
    buffers.push_back(layout);
  }
  std::vector<wgpu::BlendState> blends(colors_.size());
  std::vector<wgpu::ColorTargetState> targets(colors_.size());
  for (size_t i = 0; i < colors_.size(); ++i) {
    targets[i].format = WebGPUFormat(colors_[i]);
    auto blend = blends_.find(int(i));
    if (blend != blends_.end() && blend->second.blend_enable) {
      const auto &state = blend->second;
      blends[i].color = {Operation(state.color_op), Factor(state.src_color), Factor(state.dst_color)};
      blends[i].alpha = {Operation(state.alpha_op), Factor(state.src_alpha), Factor(state.dst_alpha)};
      targets[i].blend = &blends[i];
    }
  }
  const auto vertex_entry = vertex_->EntryPoint();
  const auto fragment_entry = fragment_ ? fragment_->EntryPoint() : std::string();
  wgpu::FragmentState fragment{};
  if (fragment_) {
    fragment.module = fragment_->Module();
    fragment.entryPoint = {fragment_entry.data(), fragment_entry.size()};
    fragment.targetCount = targets.size();
    fragment.targets = targets.data();
  }
  wgpu::DepthStencilState depth{};
  depth.format = WebGPUFormat(depth_);
  depth.depthWriteEnabled = wgpu::OptionalBool::True;
  depth.depthCompare = wgpu::CompareFunction::Less;

  wgpu::RenderPipelineDescriptor descriptor{};
  descriptor.layout = layout_;
  descriptor.vertex.module = vertex_->Module();
  descriptor.vertex.entryPoint = {vertex_entry.data(), vertex_entry.size()};
  descriptor.vertex.bufferCount = buffers.size();
  descriptor.vertex.buffers = buffers.data();
  descriptor.primitive.topology = Topology(topology);
  if (topology == PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP || topology == PRIMITIVE_TOPOLOGY_LINE_STRIP)
    descriptor.primitive.stripIndexFormat = wgpu::IndexFormat::Uint32;
  descriptor.primitive.frontFace = wgpu::FrontFace::CCW;
  descriptor.primitive.cullMode = cull_ == CULL_MODE_NONE    ? wgpu::CullMode::None
                                  : cull_ == CULL_MODE_FRONT ? wgpu::CullMode::Front
                                                             : wgpu::CullMode::Back;
  descriptor.depthStencil = depth_ == IMAGE_FORMAT_UNDEFINED ? nullptr : &depth;
  descriptor.fragment = fragment_ ? &fragment : nullptr;
  return pipelines_[topology] = core_->Device().CreateRenderPipeline(&descriptor);
}

}  // namespace grassland::graphics::backend
