#include "grassland/rhi/backend/metal/metal_program.h"

#include <cstring>
#include <spirv_msl.hpp>

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_device.h"
#include "grassland/rhi/backend/metal/metal_util.h"

namespace grassland::rhi::backend::metal {
MetalProgram::MetalProgram(MetalDevice *device, const ProgramDesc &desc) : Program(device, desc) {
}

MTL::RenderPipelineState *MetalProgram::NativePipeline() const {
  return pipeline_.get();
}

MTL::DepthStencilState *MetalProgram::NativeDepthStencil() const {
  return depth_stencil_.get();
}

// Both backends consume the same DX-layout SPIR-V. Argument IDs come from the
// complete C++ layout, so unused declarations never shift a captured version.
static bool CompileMetal(const ShaderDesc &shader, const ProgramDesc &desc, std::string &source, std::string &entry) {
  std::vector<uint8_t> code;
  if (!CompileSlang(shader, ShaderMacroHeader(desc, BACKEND_API_VULKAN), BACKEND_API_VULKAN, code,
                    &desc.pipeline_layout))
    return false;
  std::vector<uint32_t> words(code.size() / 4);
  std::memcpy(words.data(), code.data(), code.size());
  try {
    spirv_cross::CompilerMSL compiler(std::move(words));
    auto stage = shader.stage == SHADER_STAGE_VERTEX ? spv::ExecutionModelVertex : spv::ExecutionModelFragment;
    compiler.set_entry_point(shader.entry_point, stage);
    auto options = compiler.get_msl_options();
    options.set_msl_version(3, 0);
    options.argument_buffers = true;
    options.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    options.pad_argument_buffer_resources = true;
    compiler.set_msl_options(options);
    for (uint32_t g = 0; g < desc.pipeline_layout.groups.size(); ++g) {
      auto layout = desc.pipeline_layout.groups[g];
      std::sort(layout.bindings.begin(), layout.bindings.end(), [](auto a, auto b) { return a.binding < b.binding; });
      uint32_t index = 0;
      for (auto b : layout.bindings) {
        spirv_cross::MSLResourceBinding binding;
        binding.stage = stage;
        binding.desc_set = g;
        binding.binding = b.binding;
        binding.count = b.count;
        binding.msl_buffer = binding.msl_texture = binding.msl_sampler = index;
        binding.basetype = b.type == BINDING_SAMPLER     ? spirv_cross::SPIRType::Sampler
                           : b.type >= BINDING_TEXTURE2D ? spirv_cross::SPIRType::Image
                                                         : spirv_cross::SPIRType::Void;
        compiler.add_msl_resource_binding(binding);
        index += b.count;
      }
      spirv_cross::MSLResourceBinding binding;
      binding.stage = stage;
      binding.desc_set = g;
      binding.binding = spirv_cross::kBufferSizeBufferBinding;
      binding.count = 1;
      binding.basetype = spirv_cross::SPIRType::UInt;
      binding.msl_buffer = index;
      compiler.add_msl_resource_binding(binding);
      binding.binding = spirv_cross::kArgumentBufferBinding;
      binding.msl_buffer = g;
      options.pad_argument_buffer_resources = false;
      compiler.set_msl_options(options);
      compiler.add_msl_resource_binding(binding);
      options.pad_argument_buffer_resources = true;
      compiler.set_msl_options(options);
    }
    source = compiler.compile();
    entry = compiler.get_cleansed_entry_point_name(shader.entry_point, stage);
    return true;
  } catch (const spirv_cross::CompilerError &e) {
    ReportError(-1, e.what());
    return false;
  }
}

bool MetalProgram::Finalize() {
  if (finalized_)
    return true;
  try {
    if (!ValidateLayout() || desc_.vertex_bindings.size() > 31 - VertexBufferBase) {
      ReportError(-1, "Invalid Metal graphics program layout.");
      return false;
    }
    for (const auto &binding : desc_.vertex_bindings) {
      if (!binding.stride || binding.stride > 2048 || binding.stride % 4) {
        ReportError(-1, "Metal vertex stride must be a multiple of four in [4, 2048].");
        return false;
      }
    }
    for (const auto &attribute : desc_.vertex_attributes) {
      if (attribute.location >= 31 || attribute.offset % 4) {
        ReportError(-1, "Invalid Metal vertex attribute location or alignment.");
        return false;
      }
    }
    auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto device = static_cast<MetalDevice *>(device_)->NativeDevice();
    NS::SharedPtr<MTL::Function> functions[2];
    const IShader *shaders[] = {desc_.vertex_shader, desc_.fragment_shader};
    for (const auto &group : desc_.pipeline_layout.groups)
      if (!ValidateMetalBindingLayout(device, group))
        return false;
    for (int i = 0; i < 2; ++i) {
      std::string code, entry;
      if (!CompileMetal(shaders[i]->GetDesc(), desc_, code, entry))
        return false;
      NS::Error *error = nullptr;
      auto library = NS::TransferPtr(
          device->newLibrary(NS::String::string(code.c_str(), NS::UTF8StringEncoding), nullptr, &error));
      if (!library) {
        ReportError(-1, error ? error->localizedDescription()->utf8String() : "Metal library compilation failed.");
        return false;
      }
      functions[i] = NS::TransferPtr(library->newFunction(NS::String::string(entry.c_str(), NS::UTF8StringEncoding)));
      if (!functions[i]) {
        ReportError(-1, "Metal shader entry point not found.");
        return false;
      }
    }
    auto descriptor = NS::TransferPtr(MTL::RenderPipelineDescriptor::alloc()->init());
    descriptor->setVertexFunction(functions[0].get());
    descriptor->setFragmentFunction(functions[1].get());
    for (size_t i = 0; i < desc_.color_formats.size(); ++i)
      descriptor->colorAttachments()->object(i)->setPixelFormat(NativeImageFormat(desc_.color_formats[i]));
    const auto &ds = desc_.depth_stencil;
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
    depth_stencil_ = NS::TransferPtr(device->newDepthStencilState(depth.get()));
    if (!depth_stencil_)
      return false;
    auto vertex = MTL::VertexDescriptor::vertexDescriptor();
    for (size_t i = 0; i < desc_.vertex_bindings.size(); ++i) {
      auto layout = vertex->layouts()->object(i + VertexBufferBase);
      layout->setStride(desc_.vertex_bindings[i].stride);
      layout->setStepFunction(desc_.vertex_bindings[i].per_instance ? MTL::VertexStepFunctionPerInstance
                                                                    : MTL::VertexStepFunctionPerVertex);
      layout->setStepRate(1);
    }
    const MTL::VertexFormat formats[] = {MTL::VertexFormatFloat, MTL::VertexFormatFloat2, MTL::VertexFormatFloat3,
                                         MTL::VertexFormatFloat4};
    for (const auto &a : desc_.vertex_attributes) {
      auto attribute = vertex->attributes()->object(a.location);
      attribute->setFormat(formats[a.format]);
      attribute->setBufferIndex(a.binding + VertexBufferBase);
      attribute->setOffset(a.offset);
    }
    descriptor->setVertexDescriptor(vertex);
    NS::Error *error = nullptr;
    pipeline_ = NS::TransferPtr(device->newRenderPipelineState(descriptor.get(), &error));
    if (!pipeline_) {
      ReportError(-1, error ? error->localizedDescription()->utf8String() : "Metal pipeline creation failed.");
      return false;
    }
    finalized_ = true;
    return true;
  } catch (const std::bad_alloc &) {
    ReportError(-1, "Metal program allocation failed.");
    return false;
  }
}
}  // namespace grassland::rhi::backend::metal
