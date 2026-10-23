#include <slang-com-ptr.h>
#include <slang-deprecated.h>
#include <slang.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <sstream>

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/util/log.h"

namespace grassland::rhi::backend {
namespace {
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
}  // namespace

bool PreprocessShader(const ShaderDesc &desc, BackendAPI api, std::string &source) {
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
#ifdef _WIN32
  request->addPreprocessorDefine("RHI_PLATFORM_WINDOWS", "1");
#else
  request->addPreprocessorDefine("RHI_PLATFORM_WINDOWS", "0");
#endif
#ifdef __APPLE__
  request->addPreprocessorDefine("RHI_PLATFORM_MACOS", "1");
#else
  request->addPreprocessorDefine("RHI_PLATFORM_MACOS", "0");
#endif
#ifdef __linux__
  request->addPreprocessorDefine("RHI_PLATFORM_LINUX", "1");
#else
  request->addPreprocessorDefine("RHI_PLATFORM_LINUX", "0");
#endif
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
  if (desc.source.empty()) {
    request->addTranslationUnitSourceFile(unit, desc.filename.c_str());
  } else {
    const size_t offset = desc.source.compare(0, 3, "\xef\xbb\xbf") == 0 ? 3 : 0;
    request->addTranslationUnitSourceString(unit, desc.filename.empty() ? "main.slang" : desc.filename.c_str(),
                                            desc.source.c_str() + offset);
  }
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
  return true;
}
}  // namespace grassland::rhi::backend
