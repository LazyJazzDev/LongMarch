#pragma once
#include "grassland/graphics/shader.h"
#include "sparkium/backend/cuda/cuda_bindings.h"

namespace sparkium::backend {
using namespace cuda;
class OptixDevice;
}  // namespace sparkium::backend

namespace sparkium::backend::cuda {
using namespace grassland;
using namespace grassland::graphics;

class CudaShader final : public Shader {
 public:
  CudaShader(const VirtualFileSystem &,
             const std::string &,
             const std::string &,
             const std::vector<std::string> &,
             OptixDevice *optix = nullptr);
  ~CudaShader() override;
  std::string EntryPoint() const override;
  void Dispatch(const CudaBindings &, uint32_t, uint32_t, uint32_t);

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace sparkium::backend::cuda
