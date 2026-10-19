#include "grassland/rhi/backend/backend_shader_compiler.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <cstring>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
std::string ShaderMacroHeader(const ProgramDesc &desc, BackendAPI api) {
  std::string header = "#define RHI_INPUT_I(L,T,N) RHI_INPUT_##L(T,N)\n#define RHI_INPUT(L,T,N) RHI_INPUT_I(L,T,N)\n";
  for (const auto &attribute : desc.vertex_attributes) {
    auto location = std::to_string(attribute.location);
    header += "#define RHI_INPUT_" + location + "(T,N) ";
    if (api == BACKEND_API_VULKAN)
      header += "[[vk::location(" + location + ")]] ";
    header += "T N : TEXCOORD" + location + "\n";
  }
  // Resource bindings deliberately have no mapping until a C++ binding layout is supplied.
  header += "#line 1 \"shader.slang\"\n";
  return header;
}

bool CompileSlang(const ShaderDesc &shader, const std::string &prefix, BackendAPI api, std::vector<uint8_t> &code) {
  code.clear();
  thread_local Slang::ComPtr<slang::IGlobalSession> global;
  if (!global) {
    if (SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) {
      ReportError(-1, "Create Slang session failed.");
      return false;
    }
  }
  std::vector<const char *> args{"-profile",
                                 "sm_6_0",
                                 "-matrix-layout-column-major",
                                 "-O3",
                                 "-target",
                                 api == BACKEND_API_VULKAN ? "spirv" : "dxil",
                                 "-emit-spirv-directly",
                                 "-fvk-use-entrypoint-name"};
  slang::SessionDesc desc{};
  Slang::ComPtr<ISlangUnknown> allocation;
  if (SLANG_FAILED(
          global->parseCommandLineArguments(static_cast<int>(args.size()), args.data(), &desc, allocation.writeRef())))
    return false;
  Slang::ComPtr<slang::ISession> session;
  if (SLANG_FAILED(global->createSession(desc, session.writeRef())))
    return false;
  Slang::ComPtr<ISlangBlob> diagnostics;
  auto report = [&]() {
    if (diagnostics)
      ReportError(
          -1, std::string(static_cast<const char *>(diagnostics->getBufferPointer()), diagnostics->getBufferSize()));
  };
  const size_t start = shader.source.compare(0, 3, "\xef\xbb\xbf") == 0 ? 3 : 0;
  const auto source = prefix + shader.source.substr(start);
  auto module =
      session->loadModuleFromSourceString("rhi_shader", "shader.slang", source.c_str(), diagnostics.writeRef());
  if (!module) {
    report();
    return false;
  }
  Slang::ComPtr<slang::IEntryPoint> entry;
  auto stage = shader.stage == SHADER_STAGE_VERTEX ? SLANG_STAGE_VERTEX : SLANG_STAGE_FRAGMENT;
  if (SLANG_FAILED(module->findAndCheckEntryPoint(shader.entry_point.c_str(), stage, entry.writeRef(),
                                                  diagnostics.writeRef()))) {
    report();
    return false;
  }
  slang::IComponentType *components[] = {module, entry};
  Slang::ComPtr<slang::IComponentType> composite, linked;
  if (SLANG_FAILED(
          session->createCompositeComponentType(components, 2, composite.writeRef(), diagnostics.writeRef()))) {
    report();
    return false;
  }
  if (SLANG_FAILED(composite->link(linked.writeRef(), diagnostics.writeRef()))) {
    report();
    return false;
  }
  auto layout = linked->getLayout(0, diagnostics.writeRef());
  if (!layout) {
    report();
    return false;
  }
  if (layout->getParameterCount() != 0) {
    ReportError(
        -1, "Shader resource bindings require a binding layout, which is outside the current graphics program API.");
    return false;
  }
  Slang::ComPtr<ISlangBlob> blob;
  if (SLANG_FAILED(linked->getEntryPointCode(0, 0, blob.writeRef(), diagnostics.writeRef()))) {
    report();
    return false;
  }
  code.resize(blob->getBufferSize());
  std::memcpy(code.data(), blob->getBufferPointer(), code.size());
  return true;
}
}  // namespace grassland::rhi::backend
