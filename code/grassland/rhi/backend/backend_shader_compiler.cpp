#include "grassland/rhi/backend/backend_shader_compiler.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <cstring>
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend {
std::string ShaderMacroHeader(const PipelineLayoutDesc &layout,
                              BackendAPI api,
                              const std::vector<VertexAttributeDesc> &attributes) {
  std::string header = "#define RHI_INPUT_I(L,T,N) RHI_INPUT_##L(T,N)\n#define RHI_INPUT(L,T,N) RHI_INPUT_I(L,T,N)\n";
  for (const auto &attribute : attributes) {
    auto location = std::to_string(attribute.location);
    header += "#define RHI_INPUT_" + location + "(T,N) ";
    if (api == BACKEND_API_VULKAN || api == BACKEND_API_METAL)
      header += "[[vk::location(" + location + ")]] ";
    header += "T N : TEXCOORD" + location + "\n";
  }
  const char *macros[] = {"UNIFORM",
                          "STRUCTURED_BUFFER",
                          "RW_STRUCTURED_BUFFER",
                          "BYTE_ADDRESS_BUFFER",
                          "RW_BYTE_ADDRESS_BUFFER",
                          "TEXTURE2D",
                          "RW_TEXTURE2D",
                          "SAMPLER",
                          "ACCELERATION_STRUCTURE"};
  const char *types[] = {"ConstantBuffer<T>", "StructuredBuffer<T>", "RWStructuredBuffer<T>",
                         "ByteAddressBuffer", "RWByteAddressBuffer", "Texture2D<T>",
                         "RWTexture2D<T>",    "SamplerState",        "RaytracingAccelerationStructure"};
  for (auto macro : macros) {
    for (const auto &suffix : {std::string{}, std::string{"_ARRAY"}}) {
      auto name = std::string(macro) + suffix;
      header += "#define RHI_" + name + "_I(G,B) RHI_" + name + "_##G##_##B\n";
      header += "#define RHI_" + name + "_D(G,B) RHI_" + name + "_I(G,B)\n";
      bool untyped = std::string(macro) == "SAMPLER" || std::string(macro) == "ACCELERATION_STRUCTURE" ||
                     std::string(macro).find("BYTE_ADDRESS_BUFFER") != std::string::npos;
      header += "#define RHI_" + name + (untyped ? "(G,B,N) " : "(G,B,T,N) ") + "RHI_" + name + "_D(G,B)" +
                (untyped ? "(N)\n" : "(T,N)\n");
    }
  }
  for (auto mapped : MapBindings(layout, api)) {
    auto b = mapped.desc;
    std::string base = macros[b.type];
    for (bool array : {false, true}) {
      if (!array && b.count != 1)
        continue;
      auto name = base + (array ? "_ARRAY" : "");
      header += "#define RHI_" + name + "_" + std::to_string(mapped.group) + "_" + std::to_string(b.binding) +
                ((b.type == BINDING_ACCELERATION_STRUCTURE || b.type == BINDING_SAMPLER ||
                  b.type == BINDING_BYTE_ADDRESS_BUFFER || b.type == BINDING_RW_BYTE_ADDRESS_BUFFER)
                     ? "(N) "
                     : "(T,N) ");
      if (api == BACKEND_API_VULKAN)
        header += "[[vk::binding(" + std::to_string(b.binding) + "," + std::to_string(mapped.group) + ")]] ";
      header += std::string(types[b.type]) + " N" + (array ? "[" + std::to_string(b.count) + "]" : "");
      if (api != BACKEND_API_VULKAN) {
        const char reg = b.type == BINDING_UNIFORM_BUFFER ? 'b'
                         : b.type == BINDING_SAMPLER      ? 's'
                         : (b.type == BINDING_RW_STRUCTURED_BUFFER || b.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
                            b.type == BINDING_RW_TEXTURE2D)
                             ? 'u'
                             : 't';
        header += " : register(" + std::string(1, reg) + std::to_string(mapped.shader_register) + ", space" +
                  std::to_string(api == BACKEND_API_METAL ? 0 : mapped.group) + ")";
      }
      header += "\n";
    }
  }
  header += "#line 1 \"shader.slang\"\n";
  return header;
}

bool CompileSlang(const ShaderDesc &shader,
                  const std::string &prefix,
                  BackendAPI api,
                  std::vector<uint8_t> &code,
                  const PipelineLayoutDesc *binding_layout,
                  std::array<uint32_t, 3> *thread_group_size) {
  code.clear();
  thread_local Slang::ComPtr<slang::IGlobalSession> global;
  if (!global) {
    if (SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) {
      ReportError(-1, "Create Slang session failed.");
      return false;
    }
  }
  bool ray_tracing = shader.stage >= SHADER_STAGE_RAY_GENERATION;
  if (binding_layout)
    for (const auto &group : binding_layout->groups)
      for (auto binding : group.bindings)
        if (binding.type == BINDING_ACCELERATION_STRUCTURE)
          ray_tracing = true;
  std::vector<const char *> args{"-profile",
                                 ray_tracing ? "sm_6_5" : "sm_6_0",
                                 "-matrix-layout-column-major",
                                 "-O3",
                                 "-target",
                                 api == BACKEND_API_METAL    ? "metal"
                                 : api == BACKEND_API_VULKAN ? "spirv"
                                                             : "dxil",
                                 "-fvk-use-entrypoint-name"};
  // Execution platform, independent of the intermediate shader target (Metal uses SPIR-V).
#ifdef _WIN32
  args.push_back("-DRHI_PLATFORM_WINDOWS=1");
#else
  args.push_back("-DRHI_PLATFORM_WINDOWS=0");
#endif
#if defined(__APPLE__) && TARGET_OS_OSX
  args.push_back("-DRHI_PLATFORM_MACOS=1");
#else
  args.push_back("-DRHI_PLATFORM_MACOS=0");
#endif
#ifdef __linux__
  args.push_back("-DRHI_PLATFORM_LINUX=1");
#else
  args.push_back("-DRHI_PLATFORM_LINUX=0");
#endif
  if (api == BACKEND_API_VULKAN) {
    args.push_back("-emit-spirv-directly");
    args.push_back("-fvk-use-dx-layout");
  }
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
  constexpr SlangStage stages[] = {SLANG_STAGE_VERTEX,         SLANG_STAGE_FRAGMENT,     SLANG_STAGE_COMPUTE,
                                   SLANG_STAGE_RAY_GENERATION, SLANG_STAGE_MISS,         SLANG_STAGE_CLOSEST_HIT,
                                   SLANG_STAGE_ANY_HIT,        SLANG_STAGE_INTERSECTION, SLANG_STAGE_CALLABLE};
  const auto stage = stages[shader.stage];
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
  if (layout->getParameterCount() != 0 && (!binding_layout || binding_layout->groups.empty())) {
    ReportError(
        -1, "Shader resource bindings require a binding layout, which is outside the current graphics program API.");
    return false;
  }
  Slang::ComPtr<ISlangBlob> blob;
  if (SLANG_FAILED(linked->getEntryPointCode(0, 0, blob.writeRef(), diagnostics.writeRef()))) {
    report();
    return false;
  }

  Slang::ComPtr<slang::IMetadata> metadata;
  if (SLANG_FAILED(linked->getEntryPointMetadata(0, 0, metadata.writeRef(), diagnostics.writeRef()))) {
    report();
    return false;
  }
  const auto bindings = binding_layout ? MapBindings(*binding_layout, api) : std::vector<MappedBinding>{};
  for (unsigned p = 0; p < layout->getParameterCount(); ++p) {
    auto parameter = layout->getParameterByIndex(p);
    auto type_layout = parameter->getTypeLayout();
    auto type = parameter->getType();
    size_t count = 1;
    while (type->getKind() == slang::TypeReflection::Kind::Array) {
      count *= type->getElementCount();
      type = type->getElementType();
      type_layout = type_layout->getElementTypeLayout();
    }
    int resource = -1;
    if (type->getKind() == slang::TypeReflection::Kind::ConstantBuffer)
      resource = BINDING_UNIFORM_BUFFER;
    else if (type->getKind() == slang::TypeReflection::Kind::SamplerState)
      resource = BINDING_SAMPLER;
    else if (type->getKind() == slang::TypeReflection::Kind::Resource) {
      const bool rw = type->getResourceAccess() == SLANG_RESOURCE_ACCESS_READ_WRITE;
      if (type->getResourceShape() == SLANG_ACCELERATION_STRUCTURE)
        resource = BINDING_ACCELERATION_STRUCTURE;
      else if (type->getResourceShape() == SLANG_STRUCTURED_BUFFER)
        resource = rw ? BINDING_RW_STRUCTURED_BUFFER : BINDING_STRUCTURED_BUFFER;
      else if (type->getResourceShape() == SLANG_BYTE_ADDRESS_BUFFER)
        resource = rw ? BINDING_RW_BYTE_ADDRESS_BUFFER : BINDING_BYTE_ADDRESS_BUFFER;
      else if (type->getResourceShape() == SLANG_TEXTURE_2D)
        resource = rw ? BINDING_RW_TEXTURE2D : BINDING_TEXTURE2D;
    }
    const auto reg = parameter->getBindingIndex(), space = parameter->getBindingSpace();
    const auto category = parameter->getCategory();
    bool used = true;
    if (metadata)
      metadata->isParameterLocationUsed(static_cast<SlangParameterCategory>(category), space, reg, used);
    if (!used)
      continue;
    const MappedBinding *match = nullptr;
    for (const auto &m : bindings) {
      const auto native_space = api == BACKEND_API_METAL ? 0 : m.group;
      if (m.shader_register == reg && native_space == space && m.desc.type == resource) {
        match = &m;
        break;
      }
    }
    const uint32_t stage_bit = 1u << shader.stage;
    if (!match || count != match->desc.count || !(match->desc.visibility & stage_bit)) {
      ReportError(-1, std::string("Shader resource does not match the C++ binding layout: ") + parameter->getName());
      return false;
    }
    if (resource == BINDING_STRUCTURED_BUFFER || resource == BINDING_RW_STRUCTURED_BUFFER) {
      // Reflect the element in its structured-buffer layout for this target,
      // not the standalone struct or constant-buffer layout.
      auto element = type_layout->getElementTypeLayout();
      const auto stride = element ? element->getStride() : 0;
      if (!stride || stride > UINT32_MAX || (api == BACKEND_API_D3D12 && (stride > 2048 || stride % 4))) {
        ReportError(-1, "Unsupported reflected structured-buffer stride.");
        return false;
      }
      if (stride != match->desc.stride) {
        ReportError(-1, std::string("Structured-buffer stride does not match BindingDesc: ") + parameter->getName() +
                            " (layout " + std::to_string(match->desc.stride) + ", shader " + std::to_string(stride) +
                            ")");
        return false;
      }
    }
  }

  if (thread_group_size) {
    SlangUInt sizes[3]{};
    layout->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, sizes);
    for (int i = 0; i < 3; ++i)
      (*thread_group_size)[i] = uint32_t(sizes[i]);
  }
  code.resize(blob->getBufferSize());
  std::memcpy(code.data(), blob->getBufferPointer(), code.size());
  return true;
}
}  // namespace grassland::rhi::backend
