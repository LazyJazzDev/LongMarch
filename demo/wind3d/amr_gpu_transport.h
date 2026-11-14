#pragma once
#include <memory>
#include <string>

#include "amr_mesh.h"

namespace wind3d::amr {
// Persistent single-device sparse MAC transport. Topology changes require Build.
class GpuTransport {
 public:
  GpuTransport();
  ~GpuTransport();
  GpuTransport(const GpuTransport &) = delete;
  GpuTransport &operator=(const GpuTransport &) = delete;
  bool Build(int device, const Mesh &mesh, const Topology &tree, double root_width, std::array<double, 3> origin);
  // Default-stream operation on caller-owned buffers on the Build device.
  // Applies two-layer slip extension before tracing. Input/output may alias.
  // Numerical failure leaves destination unchanged; CUDA error contents are unspecified.
  bool AdvectDevice(const double *velocity,
                    double *destination,
                    size_t faces,
                    double dt,
                    bool cubic = true,
                    bool corrected = true);
  std::string Error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string build_error_;
};
}  // namespace wind3d::amr
