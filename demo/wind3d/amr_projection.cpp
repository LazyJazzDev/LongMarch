#include "amr_projection.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace wind3d::amr {
namespace {
double Dot(const std::vector<double> &a, const std::vector<double> &b) {
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}
}  // namespace

ProjectionResult ProjectReference(const Mesh &mesh,
                                  const FaceGradient &g,
                                  std::vector<double> &velocity,
                                  std::vector<double> &potential,
                                  double tolerance,
                                  int max_iterations) {
  ProjectionResult result;
  const auto &faces = mesh.Faces();
  const size_t n = mesh.Cells().size();
  if (!n || velocity.size() != faces.size() || g.Offsets().size() != faces.size() + 1 || !(tolerance > 0) ||
      !std::isfinite(tolerance) || max_iterations <= 0)
    return result;
  std::vector<double> face_gradient(faces.size()), rhs(n), diagonal(n), x(n);
  auto gradient = [&](const std::vector<double> &p) {
    for (size_t f = 0; f < faces.size(); ++f) {
      double value = 0;
      for (int k = g.Offsets()[f]; k < g.Offsets()[f + 1]; ++k)
        value += g.Weights()[k] * p[g.Columns()[k]];
      if (faces[f].aperture > 0 && faces[f].axis == 0 && faces[f].high < 0)
        value = -p[faces[f].low] / faces[f].distance;
      face_gradient[f] = value;
    }
  };
  auto negative_divergence = [&](const std::vector<double> &u, std::vector<double> &out) {
    for (size_t i = 0; i < n; ++i) {
      double sum = 0;
      for (int k = mesh.Offsets()[i]; k < mesh.Offsets()[i + 1]; ++k) {
        int signed_face = mesh.Incidence()[k], f = std::abs(signed_face) - 1;
        sum += (signed_face > 0 ? 1 : -1) * faces[f].area * faces[f].aperture * u[f];
      }
      out[i] = -sum;
    }
  };
  auto apply = [&](const std::vector<double> &p, std::vector<double> &out) {
    gradient(p);
    negative_divergence(face_gradient, out);
    for (size_t i = 0; i < n; ++i)
      if (!mesh.Cells()[i].active)
        out[i] = p[i];
  };
  negative_divergence(velocity, rhs);
  double norm = std::sqrt(Dot(rhs, rhs));
  if (!std::isfinite(norm))
    return result;
  if (norm == 0) {
    for (size_t f = 0; f < faces.size(); ++f)
      if (faces[f].aperture == 0)
        velocity[f] = 0;
    potential = x;
    result.converged = true;
    return result;
  }
  for (size_t f = 0; f < faces.size(); ++f) {
    auto face = faces[f];
    face.area *= face.aperture;
    for (int k = g.Offsets()[f]; k < g.Offsets()[f + 1]; ++k) {
      if (g.Columns()[k] == face.low)
        diagonal[face.low] -= face.area * g.Weights()[k];
      if (g.Columns()[k] == face.high)
        diagonal[face.high] += face.area * g.Weights()[k];
    }
    if (face.axis == 0 && face.high < 0)
      diagonal[face.low] += face.area / face.distance;
  }
  for (size_t i = 0; i < n; ++i)
    if (!mesh.Cells()[i].active)
      diagonal[i] = 1;
  for (double d : diagonal)
    if (!(d > 0) || !std::isfinite(d))
      return result;
  // Right-preconditioned BiCGStab: transverse reconstruction is nonsymmetric.
  std::vector<double> r = rhs, shadow = r, p(n), v(n), s(n), t(n), phat(n), shat(n), ax(n);
  double rho_old = 1, alpha = 1, omega = 1;
  result.relative_residual = 1;
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    result.iterations = iteration + 1;
    double rho = Dot(shadow, r);
    if (!std::isfinite(rho) || std::abs(rho) < 1e-300 || std::abs(omega) < 1e-300)
      break;
    double beta = (rho / rho_old) * (alpha / omega);
    for (size_t i = 0; i < n; ++i) {
      p[i] = r[i] + beta * (p[i] - omega * v[i]);
      phat[i] = p[i] / diagonal[i];
    }
    apply(phat, v);
    double divisor = Dot(shadow, v);
    if (!std::isfinite(divisor) || std::abs(divisor) < 1e-300)
      break;
    alpha = rho / divisor;
    for (size_t i = 0; i < n; ++i) {
      s[i] = r[i] - alpha * v[i];
      shat[i] = s[i] / diagonal[i];
      x[i] += alpha * phat[i];
    }
    if (std::sqrt(Dot(s, s)) / norm > tolerance) {
      apply(shat, t);
      double tt = Dot(t, t);
      if (!std::isfinite(tt) || tt < 1e-300)
        break;
      omega = Dot(t, s) / tt;
      for (size_t i = 0; i < n; ++i)
        x[i] += omega * shat[i];
    }
    // Judge convergence from the actual operator residual, never the recurrence.
    apply(x, ax);
    for (size_t i = 0; i < n; ++i)
      r[i] = rhs[i] - ax[i];
    result.relative_residual = std::sqrt(Dot(r, r)) / norm;
    if (!std::isfinite(result.relative_residual))
      break;
    if (result.relative_residual <= tolerance) {
      result.converged = true;
      break;
    }
    rho_old = rho;
  }
  if (result.converged) {
    gradient(x);
    for (size_t f = 0; f < faces.size(); ++f)
      velocity[f] = faces[f].aperture > 0 ? velocity[f] - face_gradient[f] : 0;
    potential = std::move(x);
  }
  return result;
}
}  // namespace wind3d::amr
