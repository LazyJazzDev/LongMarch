#pragma once
#include "amr_mesh.h"

namespace wind3d::amr {
// CPU reference sampler for the sparse MAC field. Native face samples are
// retained in uniform regions; virtual faces across levels use a bounded
// linear reconstruction. Geometry must remain unchanged during sampling.
class FaceField {
 public:
  FaceField(const Mesh &mesh, const Topology &tree, double root_width, std::array<double, 3> origin);
  bool ExtendSlip(std::vector<double> &velocity) const;
  bool Update(const std::vector<double> &velocity);
  std::array<double, 3> Sample(std::array<double, 3> point) const;
  double SampleComponent(std::array<double, 3> point, int axis) const;
  bool Advect(double dt, std::vector<double> &destination, bool cubic = false) const;
  // Both transport passes trace with the original velocity, as in the uniform
  // solver. Both passes and the limiter retain the target face spacing across
  // level boundaries. Out-of-stencil corrections revert to the first-order result.
  bool AdvectCorrected(double dt, std::vector<double> &destination, bool cubic = false) const;
  bool DiffuseAndSponge(double dt,
                        double viscosity,
                        double sponge_length,
                        double speed,
                        std::vector<double> &destination) const;

 private:
  int CellAt(std::array<double, 3> point) const;
  double VirtualFace(std::array<double, 3> point, int axis) const;
  double Component(std::array<double, 3> point, int axis, bool cubic = false, double spacing = 0) const;
  int NativeFace(std::array<double, 3> point, int axis) const;
  const Mesh &mesh_;
  const Topology &tree_;
  double root_width_;
  std::array<double, 3> origin_;
  std::vector<double> velocity_;
  std::vector<std::array<double, 3>> centres_;
  std::vector<std::array<double, 9>> slopes_;
  std::vector<std::array<double, 6>> limits_;
};
}  // namespace wind3d::amr
