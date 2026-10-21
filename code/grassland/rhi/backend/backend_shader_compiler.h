#pragma once
#include <array>

#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
// The C++ input layout owns locations. Slang expands this prefix; no source parsing.
std::string ShaderMacroHeader(const PipelineLayoutDesc &layout,
                              BackendAPI api,
                              const std::vector<VertexAttributeDesc> &attributes = {});
bool CompileSlang(const ShaderDesc &shader,
                  const std::string &prefix,
                  BackendAPI api,
                  std::vector<uint8_t> &code,
                  const PipelineLayoutDesc *layout = nullptr,
                  std::array<uint32_t, 3> *thread_group_size = nullptr);
}  // namespace grassland::rhi::backend
