#pragma once
#include <mutex>

#include "sparkium/backend/cpu/cpu_shader.h"
#include "sparkium/backend/cpu/slang_compiler.h"

namespace sparkium::backend::cpu {
// Ordinary exported function: a half-open range of workgroups and grid dimensions.
using HostFunction = void (*)(uint64_t, uint64_t, uint32_t, uint32_t);
using ContextFunction = void (*)(void *, uint64_t, uint64_t, uint32_t, uint32_t);

struct alignas(16) ComputeContext {
  uint64_t slots[256]{};
};

struct ConstantField {
  std::string name;
  size_t offset, size;
  void *address{};
};

struct Parameter {
  int slot;
  size_t offset, size;
  bool array;
  SlangTypeKind kind;
  std::string name;
  void *address{};
  std::vector<ConstantField> constants;
  size_t constant_size{};
};

struct CpuShader::Impl {
  std::string entry;
  std::vector<Parameter> parameters;
  size_t global_size{};
  uint32_t threads[3]{};
  Slang::ComPtr<ISlangSharedLibrary> library;
  HostFunction host{};
  ContextFunction context_host{};
  bool explicit_context{};
  std::mutex dispatch_mutex;

  ~Impl();
  void CompileCPU(const std::filesystem::path &directory,
                  const std::string &source,
                  const std::string &compute_entry,
                  const std::string &call_arguments,
                  const std::vector<std::string> &args);
  void DispatchCPU(const std::vector<uint8_t> &globals, uint32_t x, uint32_t y, uint32_t z);
};
}  // namespace sparkium::backend::cpu
