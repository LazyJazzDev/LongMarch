#pragma once
#include "grassland/rhi/rhi_program.h"

namespace grassland::rhi::backend {
// The C++ input layout owns locations. Slang expands this prefix; no source parsing.
std::string ShaderMacroHeader(const ProgramDesc &desc, BackendAPI api);
bool CompileSlang(const ShaderDesc &shader,
                  const std::string &prefix,
                  BackendAPI api,
                  std::vector<uint8_t> &code,
                  const PipelineLayoutDesc *layout = nullptr);
}  // namespace grassland::rhi::backend
