#pragma once
#include <optix.h>

#include "grassland/graphics/acceleration_structure.h"
#include "native_internal.h"

namespace grassland::graphics::backend {
class OptixDevice {
 public:
  explicit OptixDevice(CUcontext context, bool debug);
  ~OptixDevice();
  OptixDevice(const OptixDevice &) = delete;
  OptixDevice &operator=(const OptixDevice &) = delete;
  OptixDeviceContext Context() const {
    return context_;
  }

 private:
  OptixDeviceContext context_{};
};

class OptixAccelerationStructure final : public AccelerationStructure {
 public:
  OptixAccelerationStructure(OptixDevice *,
                            BufferRange vertices,
                            BufferRange indices,
                            uint32_t vertex_count,
                            uint32_t stride,
                            uint32_t primitive_count,
                            RayTracingGeometryFlag flags);
  OptixAccelerationStructure(OptixDevice *, const std::vector<RayTracingInstance> &);
  int UpdateInstances(const std::vector<RayTracingInstance> &) override;
  OptixTraversableHandle Handle() const {
    return handle_;
  }
  OptixDevice *Device() const {
    return device_;
  }
  bool IsTopLevel() const {
    return top_level_;
  }

 private:
  void Build(const OptixBuildInput &);
  OptixDevice *device_;
  bool top_level_{};
  OptixTraversableHandle handle_{};
  std::unique_ptr<NativeMemory> output_, instances_;
  std::vector<OptixInstance> previous_instances_;
};

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
