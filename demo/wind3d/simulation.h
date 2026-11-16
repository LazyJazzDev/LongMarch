#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "aircraft.h"

namespace wind3d {

struct Cell {
  float u{}, v{}, w{}, pressure{};
};

struct Vertex {
  Vec3 position, normal, color;
};

std::vector<Vertex> AircraftMesh(const Controls &controls);
bool ValidateWallDiffusion(int cuda_device);
bool ValidateClosedSurface(int cuda_device, bool cut_cells);
bool ValidateFlowMaps(int cuda_device);

struct SolverOptions {
  // Five leapfrog midpoint projections followed by a bidirectional impulse
  // reconstruction. One published step is one complete map cycle.
  bool leapfrog_flow_maps{false};
  float pressure_tolerance{0.0001f};
  int pressure_cycles{64};
  bool corrected_transport{true};
  // Kinematic viscosity of the resolved flow, in m^2/s.
  float viscosity{0.005f};
  // Wall shear from the log law at the first fluid sample instead of laminar
  // no-slip diffusion; wall_viscosity is the molecular viscosity it uses.
  bool wall_model{false};
  float wall_viscosity{1.5e-5f};
  // With the wall model, whether the log-law shear also removes momentum from
  // the first fluid cells. Without a turbulence model nothing mixes momentum
  // back toward the wall, so an applied sink separates the flow prematurely.
  // When false the flow has slip walls (Euler) and the shear is a load only:
  // a viscous-inviscid split, as in panel methods with strip friction.
  bool wall_momentum{true};
  // Cut-cell boundary (requires wall_model without wall_momentum): faces carry
  // their open area fraction in the pressure operator, divergence and loads,
  // instead of the all-or-nothing voxel staircase.
  bool cut_cells{false};
  // Monotone tricubic transported values; aperture-weighted linear fallback near walls.
  bool cubic{false};
  // Length in metres of a sponge zone ending at the outlet, in which velocity
  // relaxes toward the free stream. It removes wake vortices before they cross
  // the zero-pressure outlet, where backflow is unstable. 0 disables it.
  float sponge{0};
  // Advective Courant number. Semi-Lagrangian transport is unconditionally
  // stable; its interpolation dissipation per unit time falls as dt grows.
  float cfl{0.5f};
  // Optional upper bound on the adaptive time step; 0 keeps the automatic rule.
  float max_dt{};
  Controls initial_controls;
  // Publish the downsampled display field; batch runs only need loads.
  bool display{true};
  bool record_loads{false};
  // Batch consumers receive every step through TakeNext(); at most two snapshots queue.
  bool retain_snapshots{false};
  int devices{1};
  uint64_t stop_after{};
  float stop_time{};
  // Optional lossless per-step diagnostic CSV (including startup transients).
  std::string diagnostics_path;
  // Restore before the first step; caller must supply the original grid and model.
  std::string initial_checkpoint;
  bool export_field{true};
  bool export_vorticity{false};
  bool export_final{false};
};

// Cruise model from Claude source 46184928, including monotone cubic transport.
inline SolverOptions OptimizedSolverOptions() {
  SolverOptions options;
  options.viscosity = 1.5e-5f;
  options.wall_model = true;
  options.wall_momentum = false;
  options.cut_cells = true;
  options.wall_viscosity = 9.2e-7f;
  options.sponge = 2;
  options.cfl = 0.125f;
  options.cubic = true;
  return options;
}

// Largest supported x resolution: face counts stay within 32-bit indices.
// Create also rejects grids beyond the free device memory.
constexpr int max_grid = 1536;
// Approximate device bytes per pressure cell for all solver allocations.
size_t SolverBytes(int nx);

struct Snapshot {
  bool leapfrog_flow_maps{};
  int nx{}, ny{}, nz{}, pressure_cycles{}, solver_nx{};
  float pressure_residual{};
  bool pressure_converged{};
  uint64_t step{}, generation{};
  float time{}, dt{}, divergence_before{}, divergence_after{}, step_seconds{};
  float transport_seconds{}, pressure_seconds{}, diagnostics_seconds{};
  Vec3 force{}, moment{};
  // Largest cell-centre speed and the centre of that cell.
  float max_speed{};
  Vec3 fastest{};
  // Time integrals of force and moment since the last reset (N s, N m s).
  double impulse[3]{}, angular_impulse[3]{};
  // Time integral of the wall-shear part of the force (N s).
  double friction_impulse[3]{};
  // Time integrals of the drag (x) and lift (y) force of each component.
  double drag_impulse[kComponents]{}, lift_impulse[kComponents]{};
  Controls controls;
  std::vector<Cell> cells;
  // Full solver-grid curl magnitude, x-fastest, in s^-1.
  std::vector<float> vorticity;
  std::shared_ptr<const std::vector<Vertex>> aircraft;
};

// One instantaneous sample per completed solver step; no display field is retained.
struct LoadSample {
  uint64_t step{};
  double time{};
  float dt{};
  Vec3 force{}, moment{};
  Controls controls;
};

struct LoadHistory {
  uint64_t generation{};
  bool capacity_limited{};
  std::vector<LoadSample> samples;
};

// Single worker owns all CUDA state. Published snapshots are immutable: the
// renderer never waits for a solver step or observes half-written velocity data.
class Simulation {
 public:
  static std::unique_ptr<Simulation> Create(int cuda_device, int nx, const SolverOptions &options = {});
  ~Simulation();
  Simulation(const Simulation &) = delete;
  Simulation &operator=(const Simulation &) = delete;

  void Configure(const Controls &controls, bool paused);
  void Reset();
  // Writes the latest completed step (fields, time, step and load integrals)
  // to `path` and returns its step number, or 0 on failure. Blocks until the
  // worker has written it between two steps.
  uint64_t Checkpoint(const std::string &path);
  // Makes the next Reset resume from a checkpoint instead of an impulsive
  // start. The configured controls must match the checkpoint's grid.
  void Restore(const std::string &path);
  bool Paused() const;
  std::shared_ptr<const Snapshot> Latest() const;
  // Nonblocking ordered dequeue; empty unless retain_snapshots is enabled.
  std::shared_ptr<const Snapshot> TakeNext();
  std::string Error() const;
  LoadHistory RecentLoads() const;

 private:
  Simulation();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace wind3d
