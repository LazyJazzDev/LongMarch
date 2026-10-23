#include "grassland/rhi/backend/metal/metal_program_base.h"

#include <cstring>
#include <spirv_msl.hpp>

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"

namespace grassland::rhi::backend::metal {
// Both backends consume the same DX-layout SPIR-V. Argument IDs come from the
// complete C++ layout, so unused declarations never shift a captured version.
static bool CompileMetal(const ShaderDesc &shader,
                         const PipelineLayoutDesc &layout,
                         const std::vector<VertexAttributeDesc> &attributes,
                         std::string &source,
                         std::string &entry,
                         std::array<uint32_t, 3> *threads) {
  std::vector<uint8_t> code;
  if (!CompileSlang(shader, ShaderMacroHeader(layout, BACKEND_API_VULKAN, attributes), BACKEND_API_VULKAN, code,
                    &layout, threads))
    return false;
  std::vector<uint32_t> words(code.size() / 4);
  std::memcpy(words.data(), code.data(), code.size());
  try {
    spirv_cross::CompilerMSL compiler(std::move(words));
    auto stage = shader.stage == SHADER_STAGE_COMPUTE  ? spv::ExecutionModelGLCompute
                 : shader.stage == SHADER_STAGE_VERTEX ? spv::ExecutionModelVertex
                                                       : spv::ExecutionModelFragment;
    compiler.set_entry_point(shader.entry_point, stage);
    auto options = compiler.get_msl_options();
    options.set_msl_version(3, 0);
    options.argument_buffers = true;
    options.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    options.pad_argument_buffer_resources = true;
    compiler.set_msl_options(options);
    for (uint32_t g = 0; g < layout.groups.size(); ++g) {
      auto group_layout = layout.groups[g];
      std::sort(group_layout.bindings.begin(), group_layout.bindings.end(),
                [](auto a, auto b) { return a.binding < b.binding; });
      uint32_t index = 0;
      for (auto b : group_layout.bindings) {
        spirv_cross::MSLResourceBinding binding;
        binding.stage = stage;
        binding.desc_set = g;
        binding.binding = b.binding;
        binding.count = b.count;
        binding.msl_buffer = binding.msl_texture = binding.msl_sampler = index;
        // Unused AS slots pad as 64-bit buffer handles; active AS types come from SPIR-V.
        binding.basetype = b.type == BINDING_ACCELERATION_STRUCTURE ? spirv_cross::SPIRType::Void
                           : b.type == BINDING_SAMPLER              ? spirv_cross::SPIRType::Sampler
                           : b.type >= BINDING_TEXTURE2D            ? spirv_cross::SPIRType::Image
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

MetalProgramBase::MetalProgramBase(MTL::Device *device, const PipelineLayoutDesc &layout)
    : native_device_(device),
      layout_(layout) {
}

const PipelineLayoutDesc &MetalProgramBase::BindingLayout() const {
  return layout_;
}

NS::SharedPtr<MTL::Function> MetalProgramBase::CompileFunction(const ShaderDesc &shader,
                                                               const std::vector<VertexAttributeDesc> &attributes,
                                                               std::array<uint32_t, 3> *threads) const {
  for (const auto &group : layout_.groups)
    if (!ValidateMetalBindingLayout(native_device_, group))
      return {};
  std::string code, entry;
  if (!CompileMetal(shader, layout_, attributes, code, entry, threads))
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
