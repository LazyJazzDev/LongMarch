#include "grassland/graphics/shader.h"

#include <slang-com-ptr.h>
#include <slang.h>

#include <atomic>

namespace grassland::graphics {
namespace {
#include "built_in_shaders.inl"

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
  explicit ShaderFileSystem(const VirtualFileSystem &vfs) : vfs_(vfs) {
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
  const VirtualFileSystem &vfs_;
};

void Diagnostics(ISlangBlob *blob) {
  if (blob && blob->getBufferSize())
    LogInfo("{}", std::string(static_cast<const char *>(blob->getBufferPointer()), blob->getBufferSize()));
}

SlangStage ShaderStage(const std::string &profile) {
  const auto stage = profile.substr(0, profile.find('_'));
  if (stage == "vs")
    return SLANG_STAGE_VERTEX;
  if (stage == "ps")
    return SLANG_STAGE_FRAGMENT;
  if (stage == "cs")
    return SLANG_STAGE_COMPUTE;
  if (stage == "gs")
    return SLANG_STAGE_GEOMETRY;
  if (stage == "hs")
    return SLANG_STAGE_HULL;
  if (stage == "ds")
    return SLANG_STAGE_DOMAIN;
  if (stage == "ms")
    return SLANG_STAGE_MESH;
  if (stage == "as")
    return SLANG_STAGE_AMPLIFICATION;
  if (stage == "lib")
    return SLANG_STAGE_NONE;  // [shader(...)] identifies ray tracing entry points.
  throw std::invalid_argument("Unsupported shader profile: " + profile);
}
}  // namespace

#if defined(LONGMARCH_PYTHON_ENABLED)
void Shader::PybindClassRegistration(py::classh<Shader> &c) {
  c.def("entry_point", &Shader::EntryPoint, "Get the shader entry point");
  c.def("__repr__", [](Shader *shader) { return py::str("Shader(entry_point='{}')").format(shader->EntryPoint()); });
}
#endif

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args) {
  VirtualFileSystem vfs;
  vfs.WriteFile("shader.slang", source_code);
  return CompileShader(vfs, "shader.slang", entry_point, target, args);
}

CompiledShaderBlob CompileShader(const VirtualFileSystem &vfs,
                                 const std::string &source_file,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args) {
  CompiledShaderBlob result;
  result.entry_point = entry_point;
  // Global sessions cache Slang's standard library. Separate sessions per thread
  // avoid concurrent access to the compiler's mutable state.
  thread_local Slang::ComPtr<slang::IGlobalSession> global;
  if (!global && SLANG_FAILED(slang::createGlobalSession(global.writeRef())))
    throw std::runtime_error("Cannot initialize Slang compiler");
  const auto stage = ShaderStage(target);
  const auto suffix = target.find('_');
  std::vector<std::string> options{"-profile", "sm" + target.substr(suffix), "-matrix-layout-column-major", "-O3",
                                   "-I.",      "-fvk-use-entrypoint-name",   "-emit-spirv-directly"};
  if (std::find(args.begin(), args.end(), "-target") == args.end())
    options.insert(options.begin(), {"-target", "dxil"});
  // Preserve the public register/space binding contract on SPIR-V targets.
  for (const char *kind : {"b", "t", "u", "s"}) {
    options.push_back(std::string("-fvk-") + kind + "-shift");
    options.insert(options.end(), {"0", "all"});
  }
#ifndef NDEBUG
  options.insert(options.end(), {"-g", "-DDEBUG_SHADER"});
#endif
  options.insert(options.end(), args.begin(), args.end());
  std::vector<const char *> arguments;
  for (const auto &option : options)
    arguments.push_back(option.c_str());
  slang::SessionDesc desc{};
  Slang::ComPtr<ISlangUnknown> allocation;
  if (SLANG_FAILED(global->parseCommandLineArguments(static_cast<int>(arguments.size()), arguments.data(), &desc,
                                                     allocation.writeRef()))) {
    LogError("Invalid Slang compiler options for {}", source_file);
    return result;
  }
  Slang::ComPtr<ISlangFileSystem> files;
  files.attach(new ShaderFileSystem(vfs));
  desc.fileSystem = files;
  Slang::ComPtr<slang::ISession> session;
  if (SLANG_FAILED(global->createSession(desc, session.writeRef())))
    return result;
  std::vector<uint8_t> source;
  if (vfs.ReadFile(source_file, source)) {
    LogError("Missing shader source: {}", source_file);
    return result;
  }
  source.push_back(0);
  Slang::ComPtr<ISlangBlob> diagnostics;
  auto module = session->loadModuleFromSourceString(
      "longmarch_shader", source_file.c_str(), reinterpret_cast<const char *>(source.data()), diagnostics.writeRef());
  Diagnostics(diagnostics);
  if (!module)
    return result;
  Slang::ComPtr<slang::IEntryPoint> entry;
  SlangResult status;
  if (stage == SLANG_STAGE_NONE)
    status = module->findEntryPointByName(entry_point.c_str(), entry.writeRef());
  else
    status = module->findAndCheckEntryPoint(entry_point.c_str(), stage, entry.writeRef(), diagnostics.writeRef());
  Diagnostics(diagnostics);
  if (SLANG_FAILED(status))
    return result;
  slang::IComponentType *components[] = {module, entry};
  Slang::ComPtr<slang::IComponentType> program, linked;
  status = session->createCompositeComponentType(components, 2, program.writeRef(), diagnostics.writeRef());
  Diagnostics(diagnostics);
  if (SLANG_FAILED(status))
    return result;
  status = program->link(linked.writeRef(), diagnostics.writeRef());
  Diagnostics(diagnostics);
  if (SLANG_FAILED(status))
    return result;
  Slang::ComPtr<ISlangBlob> code;
  status = linked->getEntryPointCode(0, 0, code.writeRef(), diagnostics.writeRef());
  Diagnostics(diagnostics);
  if (SLANG_FAILED(status) || !code)
    return result;
  result.data.resize(code->getBufferSize());
  std::memcpy(result.data.data(), code->getBufferPointer(), result.data.size());
  return result;
}
}  // namespace grassland::graphics
