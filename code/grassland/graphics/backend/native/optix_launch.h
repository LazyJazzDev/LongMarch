#pragma once
#include <memory>
#include <string>

#include "native_memory.h"
#include "optix_device.h"

namespace grassland::graphics::backend {

class OptixLaunch {
 public:
  OptixLaunch(OptixDevice *, const std::string &ptx, const std::string &raygen_entry, size_t params_size);
  ~OptixLaunch();
  OptixLaunch(const OptixLaunch &) = delete;
  OptixLaunch &operator=(const OptixLaunch &) = delete;
  void Dispatch(const void *params, size_t size, uint32_t x, uint32_t y, uint32_t z);

 private:
  void Destroy() noexcept;
  OptixModule module_{};
  OptixProgramGroup groups_[3]{};
  OptixPipeline pipeline_{};
  OptixShaderBindingTable sbt_{};
  std::unique_ptr<NativeMemory> records_, params_;
};

}  // namespace grassland::graphics::backend
