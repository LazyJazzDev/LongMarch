#include "grassland/rhi/backend/metal/metal_program_base.h"

#include <cstring>
#include <set>
#ifndef LONGMARCH_OFFLINE_SHADERS
#include <spirv_msl.hpp>
#endif
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/metal/metal_binding_group.h"
#include "grassland/rhi/backend/metal/metal_diagnostics.h"
#include "grassland/rhi/shader_cache.h"

namespace grassland::rhi::backend::metal {
// Both backends consume the same DX-layout SPIR-V. Argument IDs come from the
// complete C++ layout, so unused declarations never shift a captured version.
static bool CompileMetal(const IShader *shader,
                         const PipelineLayoutDesc &layout,
                         const std::vector<InputAttributeDesc> &attributes,
                         std::string &source,
                         std::string &entry,
                         std::array<uint32_t, 3> *threads,
                         bool tier1) {
  const std::string key =
      "msl-" +
      ShaderCacheKey({"rhi-msl-v5", LONGMARCH_SHADER_CACHE_ABI, shader->GetPreprocessedCode(), shader->GetEntry(),
                      std::to_string(shader->GetStage()), ShaderMacroHeader(layout, BACKEND_API_VULKAN, attributes),
                      GetShaderCacheSettings().ios ? "ios-simdgroup" : (tier1 ? "macos-tier1" : "macos")});
  std::vector<uint8_t> cached;
  if (GetShaderCacheSettings().read_only && ReadShaderCache(key, cached)) {
    auto split = std::find(cached.begin(), cached.end(), uint8_t{0});
    if (split != cached.end() && (!threads || ReadShaderThreadGroup(key, *threads))) {
      entry.assign(cached.begin(), split);
      source.assign(split + 1, cached.end());
      return true;
    }
  }
  if (!GetShaderCacheSettings().directory.empty() && GetShaderCacheSettings().read_only)
    return false;
#ifdef LONGMARCH_OFFLINE_SHADERS
  ReportError(-1, "Offline Metal compilation requires a prepared bundle");
  return false;
#else
  std::vector<uint8_t> code;
  if (!CompileSlang(shader, ShaderMacroHeader(layout, BACKEND_API_VULKAN, attributes), BACKEND_API_VULKAN, code,
                    &layout, threads))
    return false;
  std::vector<uint32_t> words(code.size() / 4);
  std::memcpy(words.data(), code.data(), code.size());
  try {
    spirv_cross::CompilerMSL compiler(std::move(words));
    auto stage = shader->GetStage() == SHADER_STAGE_COMPUTE  ? spv::ExecutionModelGLCompute
                 : shader->GetStage() == SHADER_STAGE_VERTEX ? spv::ExecutionModelVertex
                                                             : spv::ExecutionModelFragment;
    compiler.set_entry_point(shader->GetEntry(), stage);
    auto options = compiler.get_msl_options();
    options.set_msl_version(3, 0);
    options.ios_support_base_vertex_instance = true;
    if (GetShaderCacheSettings().ios) {
      options.platform = spirv_cross::CompilerMSL::Options::iOS;
      options.ios_use_simdgroup_functions = true;
    }
    options.argument_buffers = true;
    options.argument_buffers_tier = tier1 ? spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier1
                                          : spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
    options.pad_argument_buffer_resources = true;
    compiler.set_msl_options(options);

    std::set<uint32_t> writable_texture_slots;
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
        // Tier 1 emits storage images as discrete textures. The argument-ID
        // mapping also determines these slots; reject collisions across groups.
        if (tier1 && b.type == BINDING_RW_TEXTURE2D) {
          for (uint32_t element = 0; element < b.count; ++element)
            if (index + element >= 31 || !writable_texture_slots.insert(index + element).second) {
              ReportError(-1, "Metal Tier 1 storage texture slots overlap or exceed the native limit");
              return false;
            }
        }
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
    entry = compiler.get_cleansed_entry_point_name(shader->GetEntry(), stage);
    cached.assign(entry.begin(), entry.end());
    cached.push_back(0);
    cached.insert(cached.end(), source.begin(), source.end());
    return (!threads || WriteShaderThreadGroup(key, *threads)) && WriteShaderCache(key, cached);
  } catch (const spirv_cross::CompilerError &e) {
    ReportError(-1, e.what());
    return false;
  }
#endif
}

NS::SharedPtr<MTL::Function> CompileMetalFunction(MTL::Device *device,
                                                  const PipelineLayoutDesc &layout,
                                                  const IShader *shader,
                                                  const std::vector<InputAttributeDesc> &attributes,
                                                  std::array<uint32_t, 3> *threads) {
  for (const auto &group : layout.groups)
    if (!ValidateMetalBindingLayout(device, group))
      return {};
  std::string code, entry;
  if (!CompileMetal(shader, layout, attributes, code, entry, threads,
                    GetShaderCacheSettings().ios || device->argumentBuffersSupport() == MTL::ArgumentBuffersTier1))
    return {};
  NS::Error *error = nullptr;
  auto library =
      NS::TransferPtr(device->newLibrary(NS::String::string(code.c_str(), NS::UTF8StringEncoding), nullptr, &error));
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
