#pragma once
#include <cuda_runtime_api.h>

#include <memory>
#include <string>

namespace wing_fluid {
struct Parameters {
  float speed{1.0f};
  float angle{8.0f};
  float viscosity{0.0015f};
  int cycles{5};
};

struct Diagnostics {
  float divergence_before{}, divergence_after{}, residual{}, max_speed{}, solid_speed{};
  bool finite{true};
};

// CUDA owns the solver fields; the caller owns the stream and shared RHI display buffer.
class Simulation final {
 public:
  Simulation();
  ~Simulation();
  Simulation(const Simulation &) = delete;
  Simulation &operator=(const Simulation &) = delete;
  bool Initialize(int width, int height, cudaStream_t stream);
  bool Reset(const Parameters &parameters);
  bool Step(const Parameters &parameters, float dt);
  bool Export(void *float4_buffer);
  bool Measure(Diagnostics &diagnostics);
  bool ValidatePressure(float &relative_residual, float &relative_error);
  int Levels() const;
  const std::string &Error() const;
  static constexpr float domain_width = 6.0f;
  static constexpr float domain_height = 3.0f;
  static constexpr float chord = 1.25f;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace wing_fluid
