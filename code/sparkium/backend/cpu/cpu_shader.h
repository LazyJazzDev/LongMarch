#pragma once
#include "grassland/graphics/shader.h"
#include "sparkium/backend/cpu/cpu_bindings.h"

namespace sparkium::backend::cpu {
using namespace grassland;
using namespace grassland::graphics;

class CpuShader final : public Shader {
 public:
  CpuShader(const VirtualFileSystem &, const std::string &, const std::string &, const std::vector<std::string> &);
  ~CpuShader() override;
  std::string EntryPoint() const override;
  void Dispatch(const CpuBindings &, uint32_t, uint32_t, uint32_t);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace sparkium::backend::cpu
