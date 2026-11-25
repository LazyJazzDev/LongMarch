#include "grassland/rhi/backend/backend_shader_compiler.h"

#ifndef LONGMARCH_OFFLINE_SHADERS
#include <slang-com-ptr.h>
#include <slang-deprecated.h>
#include <slang.h>
#endif
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "grassland/rhi/backend/backend_shader_cache.h"
#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/util/log.h"

namespace grassland::rhi::backend {
namespace {
char ShaderRegisterPrefix(BindingType type) {
  switch (type) {
    case BINDING_UNIFORM_BUFFER:
      return 'b';
    case BINDING_SAMPLER:
      return 's';
    case BINDING_RW_STRUCTURED_BUFFER:
    case BINDING_RW_BYTE_ADDRESS_BUFFER:
    case BINDING_RW_TEXTURE2D:
      return 'u';
    default:
      return 't';
  }
}

std::vector<std::pair<std::string, std::string>> PlatformDefines() {
  const auto target = GetShaderCacheSettings().target_platform;
  return {{"RHI_PLATFORM_WINDOWS", target == ShaderTargetPlatform::Windows ? "1" : "0"},
          {"RHI_PLATFORM_MACOS", target == ShaderTargetPlatform::MacOS ? "1" : "0"},
          {"RHI_PLATFORM_LINUX", target == ShaderTargetPlatform::Linux ? "1" : "0"},
          {"RHI_PLATFORM_IOS", target == ShaderTargetPlatform::IOS ? "1" : "0"},
          {"RHI_PLATFORM_OHOS", target == ShaderTargetPlatform::OHOS ? "1" : "0"}};
}

#ifndef LONGMARCH_OFFLINE_SHADERS
bool SameUUID(const SlangUUID &a, const SlangUUID &b) {
  return std::memcmp(&a, &b, sizeof(a)) == 0;
}

class SourceBlob final : public ISlangBlob {
 public:
  explicit SourceBlob(std::vector<uint8_t> data) : data_(std::move(data)) {
  }

  SlangResult SLANG_MCALL queryInterface(const SlangUUID &id, void **out) noexcept override {
    *out = nullptr;
    if (!SameUUID(id, ISlangUnknown::getTypeGuid()) && !SameUUID(id, ISlangBlob::getTypeGuid()))
      return SLANG_E_NO_INTERFACE;
    *out = static_cast<ISlangBlob *>(this);
    addRef();
    return SLANG_OK;
  }

  uint32_t SLANG_MCALL addRef() noexcept override {
    return ++references_;
  }

  uint32_t SLANG_MCALL release() noexcept override {
    auto count = --references_;
    if (!count)
      delete this;
    return count;
  }

  const void *SLANG_MCALL getBufferPointer() noexcept override {
    return data_.data();
  }

  size_t SLANG_MCALL getBufferSize() noexcept override {
    return data_.size();
  }

 private:
  std::atomic<uint32_t> references_{1};
  std::vector<uint8_t> data_;
};

// Slang owns include guards and include resolution. Do not suppress repeated
// includes here: generated materials legitimately include files in namespaces.
class ShaderFileSystem final : public ISlangFileSystem {
 public:
  explicit ShaderFileSystem(const FileSystem &vfs) : vfs_(vfs) {
  }

  void *SLANG_MCALL castAs(const SlangUUID &id) noexcept override {
    if (SameUUID(id, ISlangUnknown::getTypeGuid()) || SameUUID(id, ISlangCastable::getTypeGuid()) ||
        SameUUID(id, ISlangFileSystem::getTypeGuid()))
      return static_cast<ISlangFileSystem *>(this);
    return nullptr;
  }

  SlangResult SLANG_MCALL queryInterface(const SlangUUID &id, void **out) noexcept override {
    *out = castAs(id);
    if (!*out)
      return SLANG_E_NO_INTERFACE;
    addRef();
    return SLANG_OK;
  }

  uint32_t SLANG_MCALL addRef() noexcept override {
    return ++references_;
  }

  uint32_t SLANG_MCALL release() noexcept override {
    auto count = --references_;
    if (!count)
      delete this;
    return count;
  }

  SlangResult SLANG_MCALL loadFile(const char *path, ISlangBlob **out) noexcept override {
    *out = nullptr;
    try {
      std::vector<uint8_t> data;
      if (vfs_.ReadFile(std::filesystem::path(path).lexically_normal().generic_string(), data))
        return SLANG_E_NOT_FOUND;
      *out = new SourceBlob(std::move(data));
      return SLANG_OK;
    } catch (...) {
      return SLANG_FAIL;
    }
  }

 private:
  std::atomic<uint32_t> references_{1};
  const FileSystem &vfs_;
};

class TextWriter final : public ISlangWriter {
 public:
  SlangResult SLANG_MCALL queryInterface(const SlangUUID &id, void **out) noexcept override {
    *out = nullptr;
    if (!SameUUID(id, ISlangUnknown::getTypeGuid()) && !SameUUID(id, ISlangWriter::getTypeGuid()))
      return SLANG_E_NO_INTERFACE;
    *out = static_cast<ISlangWriter *>(this);
    addRef();
    return SLANG_OK;
  }

  uint32_t SLANG_MCALL addRef() noexcept override {
    return ++references_;
  }

  uint32_t SLANG_MCALL release() noexcept override {
    auto count = --references_;
    if (!count)
      delete this;
    return count;
  }

  char *SLANG_MCALL beginAppendBuffer(size_t size) noexcept override {
    buffer_.resize(size);
    return buffer_.data();
  }

  SlangResult SLANG_MCALL endAppendBuffer(char *buffer, size_t size) noexcept override {
    return write(buffer, size);
  }

  SlangResult SLANG_MCALL write(const char *data, size_t size) noexcept override {
    text_.append(data, size);
    return SLANG_OK;
  }

  void SLANG_MCALL flush() noexcept override {
  }

  SlangBool SLANG_MCALL isConsole() noexcept override {
    return false;
  }

  SlangResult SLANG_MCALL setMode(SlangWriterMode) noexcept override {
    return SLANG_OK;
  }

  const std::string &Text() const {
    return text_;
  }

 private:
  std::atomic<uint32_t> references_{1};
  std::vector<char> buffer_;
  std::string text_;
};
#endif
}  // namespace

bool PreprocessShader(const ShaderDesc &desc, BackendAPI api, std::string &source) {
  std::vector<uint8_t> original;
  std::string key;
  if (!GetShaderCacheSettings().directory.empty()) {
    // Bundle keys identify source requests; regenerate the bundle after changing includes.
    if (desc.file_system) {
      if (desc.file_system->ReadFile(desc.filename, original))
        return false;
    } else {
      std::ifstream file(desc.filename, std::ios::binary);
      if (!file)
        return false;
      original.assign(std::istreambuf_iterator<char>(file), {});
      if (file.bad())
        return false;
    }
    std::vector<std::string> parts{"rhi-preprocess-v2",
                                   LONGMARCH_SHADER_CACHE_ABI,
                                   std::to_string(api),
                                   GetShaderCacheSettings().ios ? "ios" : "native",
                                   desc.filename,
                                   desc.entry_point,
                                   std::to_string(desc.stage),
                                   std::string(original.begin(), original.end())};
    for (const auto &define : PlatformDefines()) {
      parts.push_back(define.first);
      parts.push_back(define.second);
    }
    parts.insert(parts.end(), desc.include_paths.begin(), desc.include_paths.end());
    for (auto &d : desc.defines) {
      parts.push_back(d.first);
      parts.push_back(d.second);
    }
    key = "pre-" + ShaderCacheKey(parts);
    if (GetShaderCacheSettings().read_only) {
      std::vector<uint8_t> cached;
      if (!ReadShaderCache(key, cached))
        return false;
      source.assign(cached.begin(), cached.end());
      return true;
    }
  }
#ifdef LONGMARCH_OFFLINE_SHADERS
  ReportError(-1, "Offline shader preprocessing requires a prepared bundle");
  return false;
#else
  source.clear();
  thread_local Slang::ComPtr<slang::IGlobalSession> global;
  if (!global && SLANG_FAILED(slang::createGlobalSession(global.writeRef())))
    return false;
  Slang::ComPtr<slang::ICompileRequest> request;
  if (SLANG_FAILED(global->createCompileRequest(request.writeRef())))
    return false;
  // Metal's program compiler consumes SPIR-V before translating to MSL.
  const char *args[] = {"-E", "-target", api == BACKEND_API_D3D12 ? "dxil" : "spirv", "-profile",
                        desc.stage >= SHADER_STAGE_RAY_GENERATION ? "sm_6_5" : "sm_6_0"};
  if (SLANG_FAILED(request->processCommandLineArguments(args, 5)))
    return false;
  for (const auto &define : PlatformDefines())
    request->addPreprocessorDefine(define.first.c_str(), define.second.c_str());
  for (const auto &define : desc.defines)
    request->addPreprocessorDefine(define.first.c_str(), define.second.c_str());
  for (const auto &path : desc.include_paths)
    request->addSearchPath(path.c_str());
  Slang::ComPtr<ISlangFileSystem> fs;
  if (desc.file_system) {
    fs.attach(new ShaderFileSystem(*desc.file_system));
    request->setFileSystem(fs);
  }
  Slang::ComPtr<TextWriter> output;
  output.attach(new TextWriter);
  request->setWriter(SLANG_WRITER_CHANNEL_STD_OUTPUT, output);
  const auto unit = request->addTranslationUnit(SLANG_SOURCE_LANGUAGE_SLANG, "rhi_shader");
  request->addTranslationUnitSourceFile(unit, desc.filename.c_str());
  const auto result = request->compile();
  const char *diagnostics = request->getDiagnosticOutput();
  // Slang's preprocessing-only request can return success even after errors.
  // Recognize error headers in both legacy and current Slang diagnostics; do
  // not reject warnings or inspect shader source to decide validity.
  bool diagnostic_error = false;
  if (diagnostics) {
    std::istringstream lines(diagnostics);
    std::string line;
    while (std::getline(lines, line)) {
      if (line.rfind("error[", 0) == 0 || line.rfind("fatal error[", 0) == 0 || line.rfind("error ", 0) == 0 ||
          line.rfind("fatal error ", 0) == 0 || line.find("): error ") != std::string::npos ||
          line.find("): fatal error ") != std::string::npos)
        diagnostic_error = true;
    }
  }
  if (diagnostics && *diagnostics) {
    if (SLANG_FAILED(result) || diagnostic_error)
      ReportError(-1, diagnostics);
    else
      LogWarning(diagnostics);
  }
  if (SLANG_FAILED(result) || diagnostic_error)
    return false;
  source = output->Text();
  return key.empty() || WriteShaderCache(key, {source.begin(), source.end()});
#endif
}

std::string ShaderMacroHeader(const PipelineLayoutDesc &layout,
                              BackendAPI api,
                              const std::vector<InputAttributeDesc> &attributes) {
  std::string header = "#define RHI_INPUT_I(L,T,N) RHI_INPUT_##L(T,N)\n#define RHI_INPUT(L,T,N) RHI_INPUT_I(L,T,N)\n";
  for (const auto &attribute : attributes) {
    auto location = std::to_string(attribute.location);
    header += "#define RHI_INPUT_" + location + "(T,N) ";
    if (api == BACKEND_API_VULKAN || api == BACKEND_API_METAL)
      header += "[[vk::location(" + location + ")]] ";
    header += "T N : TEXCOORD" + location + "\n";
  }

  struct ResourceMacro {
    BindingType binding_type;
    const char *name;
    const char *shader_type;
    bool typed;
  };

  // Both sampler declarations use the same binding type and generation path.
  const ResourceMacro macros[] = {
      {BINDING_UNIFORM_BUFFER, "UNIFORM", "ConstantBuffer<T>", true},
      {BINDING_STRUCTURED_BUFFER, "STRUCTURED_BUFFER", "StructuredBuffer<T>", true},
      {BINDING_RW_STRUCTURED_BUFFER, "RW_STRUCTURED_BUFFER", "RWStructuredBuffer<T>", true},
      {BINDING_BYTE_ADDRESS_BUFFER, "BYTE_ADDRESS_BUFFER", "ByteAddressBuffer", false},
      {BINDING_RW_BYTE_ADDRESS_BUFFER, "RW_BYTE_ADDRESS_BUFFER", "RWByteAddressBuffer", false},
      {BINDING_TEXTURE2D, "TEXTURE2D", "Texture2D<T>", true},
      {BINDING_RW_TEXTURE2D, "RW_TEXTURE2D", "RWTexture2D<T>", true},
      {BINDING_SAMPLER, "SAMPLER", "SamplerState", false},
      {BINDING_SAMPLER, "COMPARISON_SAMPLER", "SamplerComparisonState", false},
      {BINDING_ACCELERATION_STRUCTURE, "ACCELERATION_STRUCTURE", "RaytracingAccelerationStructure", false}};
  for (const auto &macro : macros) {
    for (bool array : {false, true}) {
      const auto name = std::string(macro.name) + (array ? "_ARRAY" : "");
      header += "#define RHI_" + name + "_I(G,B) RHI_" + name + "_##G##_##B\n";
      header += "#define RHI_" + name + (macro.typed ? "(G,B,T,N) " : "(G,B,N) ") + "RHI_" + name + "_I(G,B)" +
                (macro.typed ? "(T,N)\n" : "(N)\n");
    }
  }
  for (auto mapped : MapBindings(layout, api)) {
    const auto &b = mapped.desc;
    for (const auto &macro : macros) {
      if (macro.binding_type != b.type)
        continue;
      for (bool array : {false, true}) {
        if (!array && b.count != 1)
          continue;
        const auto name = std::string(macro.name) + (array ? "_ARRAY" : "");
        header += "#define RHI_" + name + "_" + std::to_string(mapped.group) + "_" + std::to_string(b.binding) +
                  (macro.typed ? "(T,N) " : "(N) ");
        if (api == BACKEND_API_VULKAN)
          header += "[[vk::binding(" + std::to_string(b.binding) + "," + std::to_string(mapped.group) + ")]] ";
        header += std::string(macro.shader_type) + " N" + (array ? "[" + std::to_string(b.count) + "]" : "");
        if (api != BACKEND_API_VULKAN) {
          const char reg = ShaderRegisterPrefix(b.type);
          header += " : register(" + std::string(1, reg) + std::to_string(mapped.shader_register) + ", space" +
                    std::to_string(api == BACKEND_API_METAL ? 0 : mapped.group) + ")";
        }
        header += "\n";
      }
    }
  }
  header += "#line 1 \"shader.slang\"\n";
  return header;
}

std::string ShaderBindingLayoutKey(const PipelineLayoutDesc &layout) {
  std::vector<std::string> parts{"rhi-binding-layout-v1", std::to_string(layout.groups.size())};
  for (size_t group = 0; group < layout.groups.size(); ++group) {
    auto bindings = layout.groups[group].bindings;
    std::sort(bindings.begin(), bindings.end(), [](const auto &a, const auto &b) { return a.binding < b.binding; });
    parts.push_back(std::to_string(group));
    parts.push_back(std::to_string(bindings.size()));
    for (const auto &binding : bindings) {
      parts.push_back(std::to_string(binding.binding));
      parts.push_back(std::to_string(binding.type));
      parts.push_back(std::to_string(binding.count));
      parts.push_back(std::to_string(binding.visibility));
      parts.push_back(std::to_string(binding.stride));
    }
  }
  return ShaderCacheKey(parts);
}

bool CompileSlang(const IShader *shader_object,
                  const std::string &prefix,
                  BackendAPI api,
                  std::vector<uint8_t> &code,
                  const PipelineLayoutDesc *binding_layout,
                  std::array<uint32_t, 3> *thread_group_size) {
  const std::string key =
      "slang-" + ShaderCacheKey({"rhi-spirv-v3", LONGMARCH_SHADER_CACHE_ABI, prefix,
                                 ShaderBindingLayoutKey(binding_layout ? *binding_layout : PipelineLayoutDesc{}),
                                 shader_object->GetPreprocessedCode(), shader_object->GetEntry(),
                                 std::to_string(shader_object->GetStage()), std::to_string(api),
                                 std::to_string(static_cast<int>(GetShaderCacheSettings().target_platform))});
  if (GetShaderCacheSettings().read_only && ReadShaderCache(key, code)) {
    if (!thread_group_size || ReadShaderThreadGroup(key, *thread_group_size))
      return true;
  }
  if (!GetShaderCacheSettings().directory.empty() && GetShaderCacheSettings().read_only)
    return false;
#ifdef LONGMARCH_OFFLINE_SHADERS
  ReportError(-1, "Offline compilation requires a prepared bundle");
  return false;
#else
  const auto &preprocessed = shader_object->GetPreprocessedCode();
  code.clear();
  thread_local Slang::ComPtr<slang::IGlobalSession> global;
  if (!global) {
    if (SLANG_FAILED(slang::createGlobalSession(global.writeRef()))) {
      ReportError(-1, "Create Slang session failed.");
      return false;
    }
  }
  bool ray_tracing = shader_object->GetStage() >= SHADER_STAGE_RAY_GENERATION;
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
  std::vector<std::string> platform_args;
  for (const auto &define : PlatformDefines())
    platform_args.push_back("-D" + define.first + "=" + define.second);
  for (const auto &arg : platform_args)
    args.push_back(arg.c_str());
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
  const auto source = prefix + preprocessed;
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
  const auto stage = stages[shader_object->GetStage()];
  if (SLANG_FAILED(module->findAndCheckEntryPoint(shader_object->GetEntry().c_str(), stage, entry.writeRef(),
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
    const uint32_t stage_bit = 1u << shader_object->GetStage();
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

  if (shader_object->GetStage() == SHADER_STAGE_COMPUTE) {
    SlangUInt sizes[3]{};
    std::array<uint32_t, 3> group{};
    layout->getEntryPointByIndex(0)->getComputeThreadGroupSize(3, sizes);
    for (int i = 0; i < 3; ++i)
      group[i] = uint32_t(sizes[i]);
    if (thread_group_size)
      *thread_group_size = group;
    if (!WriteShaderThreadGroup(key, group))
      return false;
  }
  code.resize(blob->getBufferSize());
  std::memcpy(code.data(), blob->getBufferPointer(), code.size());
  return WriteShaderCache(key, code);
#endif
}
}  // namespace grassland::rhi::backend
