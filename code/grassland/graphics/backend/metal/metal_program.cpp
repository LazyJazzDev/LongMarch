#include "grassland/graphics/backend/metal/metal_program.h"

#include <stdexcept>

#include "grassland/graphics/backend/metal/metal_core.h"

namespace grassland::graphics::backend {

namespace {
void AddBinding(std::vector<MetalBinding> &bindings, ResourceType type, int count) {
  if (count <= 0 || (type == RESOURCE_TYPE_ACCELERATION_STRUCTURE && count != 1))
    throw std::invalid_argument("unsupported Metal resource binding");
  bindings.push_back({type, count});
}
}  // namespace

void MetalComputeProgram::AddResourceBinding(ResourceType type, int count) {
  AddBinding(bindings, type, count);
}

void MetalComputeProgram::Finalize() {
  MetalPool pool;
  stage = CompileMetalStage(core, shader, bindings);
  NS::Error *error = nullptr;
  pipeline = NS::TransferPtr(core->Device()->newComputePipelineState(stage.function.get(), &error));
  MetalCheck(pipeline.get(), error, "newComputePipelineState");
  if (stage.threads.width * stage.threads.height * stage.threads.depth > pipeline->maxTotalThreadsPerThreadgroup())
    throw std::runtime_error("shader threadgroup exceeds Metal pipeline limit");
}

MetalProgram::MetalProgram(MetalCore *core, const std::vector<ImageFormat> &colors, ImageFormat depth) : core(core) {
  MetalPool pool;
  descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
  if (colors.size() > 8)
    throw std::invalid_argument("Metal supports at most 8 color attachments");
  for (size_t i = 0; i < colors.size(); ++i)
    descriptor->colorAttachments()->object(i)->setPixelFormat(MetalFormat(colors[i]));
  descriptor->setDepthAttachmentPixelFormat(MetalFormat(depth));
  descriptor->setVertexDescriptor(MTL::VertexDescriptor::vertexDescriptor());
  auto depth_descriptor = NS::TransferPtr(MTL::DepthStencilDescriptor::alloc()->init());
  depth_descriptor->setDepthCompareFunction(depth == IMAGE_FORMAT_UNDEFINED ? MTL::CompareFunctionAlways
                                                                            : MTL::CompareFunctionLess);
  depth_descriptor->setDepthWriteEnabled(depth != IMAGE_FORMAT_UNDEFINED);
  depth_state = NS::TransferPtr(core->Device()->newDepthStencilState(depth_descriptor.get()));
}

void MetalProgram::AddInputBinding(uint32_t stride, bool per_instance) {
  if (input_bindings >= 15)
    throw std::out_of_range("Metal vertex binding slots");
  auto layout = descriptor->vertexDescriptor()->layouts()->object(16 + input_bindings++);
  layout->setStride(stride);
  layout->setStepFunction(per_instance ? MTL::VertexStepFunctionPerInstance : MTL::VertexStepFunctionPerVertex);
  layout->setStepRate(1);
}

void MetalProgram::AddInputAttribute(uint32_t binding, InputType type, uint32_t offset) {
  static const MTL::VertexFormat formats[] = {MTL::VertexFormatUInt,  MTL::VertexFormatInt,  MTL::VertexFormatFloat,
                                              MTL::VertexFormatUInt2, MTL::VertexFormatInt2, MTL::VertexFormatFloat2,
                                              MTL::VertexFormatUInt3, MTL::VertexFormatInt3, MTL::VertexFormatFloat3,
                                              MTL::VertexFormatUInt4, MTL::VertexFormatInt4, MTL::VertexFormatFloat4};
  if (type < 0 || type > INPUT_TYPE_FLOAT4 || attributes >= 31 || binding >= input_bindings)
    throw std::out_of_range("Metal vertex attribute");
  auto attribute = descriptor->vertexDescriptor()->attributes()->object(attributes++);
  attribute->setFormat(formats[type]);
  attribute->setOffset(offset);
  attribute->setBufferIndex(16 + binding);
}

void MetalProgram::AddResourceBinding(ResourceType type, int count) {
  AddBinding(bindings, type, count);
}

void MetalProgram::SetBlendState(int target, const BlendState &state) {
  if (target < 0 || target >= 8)
    throw std::out_of_range("Metal blend target");
  static const MTL::BlendFactor factors[] = {MTL::BlendFactorZero,
                                             MTL::BlendFactorOne,
                                             MTL::BlendFactorSourceColor,
                                             MTL::BlendFactorOneMinusSourceColor,
                                             MTL::BlendFactorDestinationColor,
                                             MTL::BlendFactorOneMinusDestinationColor,
                                             MTL::BlendFactorSourceAlpha,
                                             MTL::BlendFactorOneMinusSourceAlpha,
                                             MTL::BlendFactorDestinationAlpha,
                                             MTL::BlendFactorOneMinusDestinationAlpha};
  auto attachment = descriptor->colorAttachments()->object(target);
  attachment->setBlendingEnabled(state.blend_enable);
  attachment->setSourceRGBBlendFactor(factors[state.src_color]);
  attachment->setDestinationRGBBlendFactor(factors[state.dst_color]);
  attachment->setSourceAlphaBlendFactor(factors[state.src_alpha]);
  attachment->setDestinationAlphaBlendFactor(factors[state.dst_alpha]);
  attachment->setRgbBlendOperation(static_cast<MTL::BlendOperation>(state.color_op));
  attachment->setAlphaBlendOperation(static_cast<MTL::BlendOperation>(state.alpha_op));
}

void MetalProgram::BindShader(Shader *shader, ShaderType type) {
  auto metal = dynamic_cast<MetalShader *>(shader);
  if (!metal)
    throw std::invalid_argument("expected MetalShader");
  if (type == SHADER_TYPE_VERTEX)
    vertex = metal;
  else if (type == SHADER_TYPE_PIXEL)
    fragment = metal;
  else
    throw std::runtime_error("Metal does not support geometry shaders");
}

void MetalProgram::Finalize() {
  MetalPool pool;
  vertex_stage = CompileMetalStage(core, vertex, bindings);
  descriptor->setVertexFunction(vertex_stage.function.get());
  if (fragment) {
    fragment_stage = CompileMetalStage(core, fragment, bindings);
    descriptor->setFragmentFunction(fragment_stage.function.get());
  }

  NS::Error *error = nullptr;
  pipeline = NS::TransferPtr(core->Device()->newRenderPipelineState(descriptor.get(), &error));
  MetalCheck(pipeline.get(), error, "newRenderPipelineState");
}

}  // namespace grassland::graphics::backend
