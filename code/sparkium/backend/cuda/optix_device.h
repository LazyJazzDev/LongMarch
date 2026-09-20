#pragma once
#include <cuda.h>
#include <optix.h>

namespace sparkium::backend {

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

}  // namespace sparkium::backend
