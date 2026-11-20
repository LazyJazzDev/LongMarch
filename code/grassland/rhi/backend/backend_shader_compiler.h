#pragma once
#include <array>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
bool PreprocessShader(const ShaderDesc &desc, BackendAPI api, std::string &source);

// The C++ input layout owns locations. Slang expands this prefix; no source parsing.
std::string ShaderMacroHeader(const PipelineLayoutDesc &layout,
                              BackendAPI api,
                              const std::vector<InputAttributeDesc> &attributes = {});
// Defines RHI_PLATFORM_WINDOWS/MACOS/LINUX/IOS as 0 or 1, also for imports.
// Normally follows the execution platform; ShaderCacheSettings::ios selects iOS
// when preparing its bundle on a host. Independent of api/intermediate format.
bool CompileSlang(const IShader *shader,
                  const std::string &prefix,
                  BackendAPI api,
                  std::vector<uint8_t> &code,
                  const PipelineLayoutDesc *layout = nullptr,
                  std::array<uint32_t, 3> *thread_group_size = nullptr);
}  // namespace grassland::rhi::backend
