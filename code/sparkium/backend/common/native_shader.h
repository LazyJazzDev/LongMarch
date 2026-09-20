#pragma once
#include "grassland/graphics/shader.h"
#include "sparkium/backend/common/native_bindings.h"

namespace sparkium::backend {
using namespace grassland;
using namespace grassland::graphics;

class OptixDevice;

class NativeShader final : public Shader {
 public:
  NativeShader(bool cuda,
               const VirtualFileSystem &,
               const std::string &,
               const std::string &,
               const std::vector<std::string> &,
               OptixDevice *optix = nullptr);
  ~NativeShader() override;
  std::string EntryPoint() const override;
  void Dispatch(const NativeBindings &, uint32_t, uint32_t, uint32_t);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace sparkium::backend
