#pragma once
#include <memory>
#include <string>
#include <vector>

#include "grassland/rhi/rhi.h"

namespace wing_fluid {
struct Parameters {
  float speed{1.0f};
  float angle{8.0f};
  float viscosity{0.0015f};
  int cycles{5};
  bool smooth_boundary{true};
  int profile{};
};

struct Diagnostics {
  float divergence_before{}, divergence_after{}, residual{}, max_speed{}, solid_speed{};
  bool finite{true};
};

// Per-unit-span fluid load, normalized by q*c and q*c^2 (rho cancels).
// World axes: +x downstream, +y up; positive moment is counterclockwise about mid-chord.
struct LoadCoefficients {
  float x{}, y{}, moment{}, angle{};
  int profile{};
  bool valid{};
};

// Backend-neutral state used only when transferring an existing flow between solvers.
// Coarse-grid corrections and advection scratch are rebuilt by the next step.
struct SimulationState {
  int width{}, height{};
  Parameters parameters;
  float time{}, last_dt{};
  bool loads_valid{};
  std::vector<float> u, v, pressure, rhs, residual, dye;

  bool Matches(int w, int h) const {
    return width == w && height == h && u.size() == size_t(w + 1) * h && v.size() == size_t(w) * (h + 1) &&
           pressure.size() == size_t(w) * h && rhs.size() == pressure.size() && residual.size() == pressure.size() &&
           dye.size() == pressure.size();
  }
};

// Both solvers own their GPU fields. The application owns the graphics submissions.
class Simulation {
 public:
  virtual ~Simulation() = default;
  virtual bool Reset(const Parameters &) = 0;
  virtual bool Step(const Parameters &, float dt) = 0;
  virtual bool SaveState(SimulationState &) = 0;
  virtual bool RestoreState(const SimulationState &) = 0;
  virtual bool BeginFrame(grassland::rhi::Event previous) = 0;
  virtual bool Export() = 0;
  virtual grassland::rhi::Event ReadyEvent() const = 0;
  virtual grassland::rhi::IBuffer *Field() const = 0;
  virtual bool Measure(Diagnostics &) = 0;
  virtual bool MeasureLoads(LoadCoefficients &) = 0;
  virtual bool ValidateLoads(float &relative_error) = 0;
  virtual bool ValidatePressure(float &relative_residual, float &relative_error) = 0;
  virtual int Levels() const = 0;
  virtual const std::string &Error() const = 0;
  virtual const char *Name() const = 0;
  static constexpr float domain_width = 6.0f;
  static constexpr float domain_height = 3.0f;
  static constexpr float chord = 1.25f;
};

std::unique_ptr<Simulation> CreateComputeSimulation(grassland::rhi::IDevice *device, int width, int height);
#ifdef LONGMARCH_CUDA_ENABLED
std::unique_ptr<Simulation> CreateCUDASimulation(grassland::rhi::IDevice *device, int width, int height);
#endif
}  // namespace wing_fluid
