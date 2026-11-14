#pragma once
#include <memory>
#include <string>

#include "amr_advance.h"

namespace wind3d::amr {
// Persistent single-device sparse MAC transport. Topology changes require Build.
class GpuProjection;

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
  // Projection must be built for the same device and mesh. Velocity and
  // physical pressure are disjoint caller-owned buffers; only successful
  // steps commit them. Default stream, as for AdvectDevice.
  AdvanceResult AdvanceDevice(GpuProjection &projection,
                              double *velocity,
                              size_t faces,
                              double *pressure,
                              size_t cells,
                              double dt,
                              double speed,
                              bool cubic = true,
                              double density = 1.225,
                              double wall_viscosity = 9.2e-7,
                              double viscosity = 1.5e-5,
                              double sponge_length = 2);
  std::string Error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string build_error_;
};
}  // namespace wind3d::amr
