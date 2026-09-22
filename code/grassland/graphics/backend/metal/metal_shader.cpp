#include "grassland/graphics/backend/metal/metal_shader.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <spirv_cross/spirv_msl.hpp>
#include <stdexcept>

#include "grassland/graphics/backend/metal/metal_core.h"

namespace grassland::graphics::backend {

MetalStage CompileMetalStage(MetalCore *core,
                             MetalShader *shader,
                             const std::vector<MetalBinding> &bindings,
                             bool packed) {
  if (!shader || shader->blob.data.empty())
    throw std::invalid_argument("missing Metal shader");
  MetalPool pool;
  std::vector<uint32_t> words(shader->blob.data.size() / 4);
  std::memcpy(words.data(), shader->blob.data.data(), words.size() * 4);
  spirv_cross::CompilerMSL compiler(std::move(words));
  auto entry = compiler.get_entry_points_and_stages()[0];
  compiler.set_entry_point(entry.name, entry.execution_model);
  auto options = compiler.get_msl_options();
  options.set_msl_version(3, 0);
  options.argument_buffers = true;
  options.argument_buffers_tier = spirv_cross::CompilerMSL::Options::ArgumentBuffersTier::Tier2;
  compiler.set_msl_options(options);
  if (!packed && bindings.size() > spirv_cross::kMaxArgumentBuffers)
    throw std::runtime_error(
        "legacy Metal resource layout exceeds the SPIRV-Cross argument-buffer limit; use ShaderCode to pack resource slots");
  uint32_t argument_index = 0;
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    spirv_cross::MSLResourceBinding binding;
    binding.stage = entry.execution_model;
    binding.desc_set = packed ? 0 : i;
    binding.binding = packed ? i : 0;
    binding.count = bindings[i].count;
    binding.msl_buffer = binding.msl_texture = binding.msl_sampler = packed ? argument_index : 0;
    argument_index += bindings[i].count;
    compiler.add_msl_resource_binding(binding);
    binding.binding = spirv_cross::kArgumentBufferBinding;
    binding.msl_buffer = packed ? 0 : i;
    compiler.add_msl_resource_binding(binding);
  }

  auto source = compiler.compile();
  // Opt-in shader dumps make translator/compiler failures reproducible.
  if (const char *directory = std::getenv("LONGMARCH_METAL_SHADER_DUMP")) {
    std::filesystem::create_directories(directory);
    std::ofstream(std::filesystem::path(directory) /
                  (entry.name + "-" + std::to_string(std::hash<std::string>{}(source)) + ".metal"))
        << source;
  }

  NS::Error *error = nullptr;
  auto compile_options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  compile_options->setLanguageVersion(MTL::LanguageVersion3_0);
  compile_options->setFastMathEnabled(true);
  auto library = NS::TransferPtr(core->Device()->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding),
                                                            compile_options.get(), &error));
  MetalCheck(library.get(), error, "compile MSL");
  MetalStage result;
  result.packed = packed;
  auto name = compiler.get_cleansed_entry_point_name(entry.name, entry.execution_model);
  result.function = NS::TransferPtr(library->newFunction(NS::String::string(name.c_str(), NS::UTF8StringEncoding)));
  MetalCheck(result.function.get(), nullptr, "Metal shader entry point");
  argument_index = 0;
  for (uint32_t i = 0; i < bindings.size(); ++i) {
    if (compiler.is_msl_resource_binding_used(entry.execution_model, packed ? 0 : i, packed ? i : 0)) {
      result.resource_indices.emplace(i, packed ? argument_index : 0);
      auto buffer_slot = packed ? 0 : i;
      if (!result.arguments.count(buffer_slot)) {
        auto encoder = NS::TransferPtr(result.function->newArgumentEncoder(buffer_slot));
        MetalCheck(encoder.get(), nullptr, "Metal argument encoder");
        result.arguments.emplace(buffer_slot, std::move(encoder));
      }
    }
    argument_index += bindings[i].count;
  }
  if (entry.execution_model == spv::ExecutionModelGLCompute) {
    const auto &wg = compiler.get_entry_point(entry.name, entry.execution_model).workgroup_size;
    result.threads = MTL::Size(wg.x, wg.y, wg.z);
  }
  return result;
}

}  // namespace grassland::graphics::backend
