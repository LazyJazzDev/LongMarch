#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "simulation.h"
#include "wing_profile.h"

namespace wing_fluid {
namespace {
using namespace grassland::rhi;
#include "simulation.slang.h"
#include "wing_profile.h.h"

struct Level {
  uint32_t w{}, h{}, p{}, b{}, r{}, east{}, north{}, boundary{};
};

struct DispatchSettings {
  Level level, coarse;
  std::array<uint32_t, 4> fields0{}, fields1{};
  std::array<float, 4> values0{}, values1{};
  std::array<int32_t, 4> controls{};
  std::array<uint32_t, 4> scratch{};
};

static_assert(sizeof(DispatchSettings) == 160);

enum Kernel {
  SmoothGeometry,
  ActiveCells,
  Geometry,
  FineOperator,
  CoarsenOperator,
  Smooth,
  Residual,
  CoarseSolve,
  Restrict,
  Prolong,
  InitializeVelocity,
  InitializeDye,
  Boundaries,
  Advect,
  Diffuse,
  RHS,
  Project,
  AdvectDye,
  CorrectDye,
  Display,
  MeasureFields,
  Manufactured,
  SmoothLoads,
  VoxelLoads,
  Reduce
};

class ComputeSimulation final : public Simulation {
 public:
  ~ComputeSimulation() override {
    if (pending_)
      pending_->WaitHost();
  }

  bool Initialize(IDevice *device, int width, int height);
  bool Reset(const Parameters &parameters) override;
  bool Step(const Parameters &parameters, float dt) override;
  bool SaveState(SimulationState &) override;
  bool RestoreState(const SimulationState &) override;

  bool BeginFrame(Event) override {
    return true;
  }  // Same RHI queue as rendering.

  bool Export() override;

  Event ReadyEvent() const override {
    return pending_ ? pending_->GetEvent() : Event{};
  }

  IBuffer *Field() const override {
    return field_.Get();
  }

  bool Measure(Diagnostics &diagnostics) override;
  bool MeasureLoads(LoadCoefficients &loads) override;
  bool ValidateLoads(float &relative_error) override;
  bool ValidatePressure(float &relative_residual, float &relative_error) override;

  int Levels() const override {
    return int(levels_.size());
  }

  const std::string &Error() const override {
    return error_;
  }

  const char *Name() const override {
    return "Compute Shader";
  }

 private:
  bool Fail(const char *message) {
    if (error_.empty())
      error_ = message;
    return false;
  }

  bool Begin();
  bool Flush();
  bool Run(Kernel kernel, int count, Level level, Level coarse = {}, int argument = 0, bool diagnostic = false);
  bool Cycle(size_t index);
  bool Solve(int cycles);
  bool GeometryFor(const Parameters &parameters);
  bool Read(uint32_t offset, std::vector<float> &values);
  bool Reduced(int count, bool diagnostic, std::array<double, 7> &result);
  bool ReadLoads(std::array<double, 7> &values, int analytic = 0);

  void Update(const Parameters &p, float dt) {
    settings_.values0 = {p.speed, p.angle, p.viscosity, dt};
    settings_.values1 = {time_, float(p.profile), 0, 0};
    settings_.fields0 = {u_, v_, next_u_, next_v_};
    settings_.fields1 = {mask_, dye_, next_dye_, corrected_dye_};
  }

  uint32_t Allocate(uint32_t count) {
    auto offset = size_;
    size_ += count;
    return offset;
  }

  ObjPtr<IQueue> queue_;
  ObjPtr<IBuffer> heap_, field_, uniform_;
  ObjPtr<IComputeProgram> program_;
  ObjPtr<IBindingGroup> group_;
  ObjPtr<ICommandContext> commands_;
  ObjPtr<ISubmissionToken> pending_;
  std::vector<Level> levels_;
  DispatchSettings settings_{};
  Parameters parameters_{};
  uint32_t size_{}, u_{}, v_{}, next_u_{}, next_v_{}, mask_{}, dye_{}, next_dye_{}, corrected_dye_{}, stats_{},
      reduced_{};
  int recorded_{};
  float time_{}, last_dt_{1};
  bool loads_valid_{};
  std::string error_;
};

bool ComputeSimulation::Begin() {
  if (!error_.empty())
    return false;
  if (commands_)
    return true;
  commands_ = queue_->CreateCommandContext();
  return (commands_ && commands_->CmdBindComputeProgram(program_.Get()) && commands_->CmdBindGroup(0, group_.Get())) ||
         Fail("Compute command recording failed");
}

bool ComputeSimulation::Flush() {
  if (!commands_)
    return error_.empty();
  pending_ = commands_->Submit();
  commands_.Release();
  recorded_ = 0;
  return bool(pending_) || Fail("Compute submission failed");
}

bool ComputeSimulation::Run(Kernel kernel, int count, Level level, Level coarse, int argument, bool diagnostic) {
  if (!Begin())
    return false;
  settings_.level = level;
  settings_.coarse = coarse;
  settings_.controls = {int(kernel), argument, count, int(diagnostic)};
  settings_.scratch = {stats_, reduced_, 0, 0};
  if (!commands_->CmdUploadBuffer(uniform_.Get(), 0, &settings_, sizeof(settings_)) ||
      !commands_->CmdDispatch((count + 255) / 256))
    return Fail("Compute dispatch failed");
  return ++recorded_ < 1024 || Flush();
}

bool ComputeSimulation::Initialize(IDevice *device, int width, int height) {
  if (width < 64 || height * 2 != width || (width & (width - 1)))
    return Fail("Invalid MAC grid");
  queue_ = device->GetQueue(QUEUE_TYPE_MAIN);
  for (int w = width, h = height;; w /= 2, h /= 2) {
    uint32_t n = w * h;
    levels_.push_back(
        {uint32_t(w), uint32_t(h), Allocate(n), Allocate(n), Allocate(n), Allocate(n), Allocate(n), Allocate(n)});
    if (w == 4)
      break;
  }
  u_ = Allocate((width + 1) * height);
  v_ = Allocate(width * (height + 1));
  next_u_ = Allocate((width + 1) * height);
  next_v_ = Allocate(width * (height + 1));
  mask_ = Allocate(width * height);
  dye_ = Allocate(width * height);
  next_dye_ = Allocate(width * height);
  corrected_dye_ = Allocate(width * height);
  stats_ = Allocate(width * height * 7);
  reduced_ = Allocate(((width * height + 255) / 256) * 7);
  heap_ = device->CreateBuffer({uint64_t(size_) * 4});
  field_ = device->CreateBuffer({uint64_t(width) * height * 16});
  uniform_ = device->CreateBuffer({256});
  std::string source = "#define WING_PROFILE_SHADER\n";
  source.append(reinterpret_cast<const char *>(wing_profile_h), wing_profile_h_len);
  source.append("\n");
  source.append(reinterpret_cast<const char *>(simulation_slang), simulation_slang_len);
  ObjPtr<IShader> shader = device->CreateShaderFromSource(source, "CSMain", SHADER_STAGE_COMPUTE);
  BindingLayoutDesc layout{{{0, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                            {1, BINDING_RW_BYTE_ADDRESS_BUFFER, 1, SHADER_VISIBILITY_COMPUTE},
                            {2, BINDING_UNIFORM_BUFFER, 1, SHADER_VISIBILITY_COMPUTE}}};
  if (!queue_ || !heap_ || !field_ || !uniform_ || !shader)
    return Fail("Compute resource/shader creation failed");
  ComputeProgramDesc desc;
  desc.compute_shader = shader.Get();
  desc.pipeline_layout.groups = {layout};
  program_ = device->CreateComputeProgram(desc);
  group_ = device->CreateBindingGroup(layout);
  if (!program_ || !group_ || !group_->BindResource(0, heap_.Get()) || !group_->BindResource(1, field_.Get()) ||
      !group_->BindResource(2, uniform_.Get()) || !Begin())
    return Fail("Compute pipeline creation failed");
  std::vector<float> zero(size_);
  return commands_->CmdUploadBuffer(heap_.Get(), 0, zero.data(), uint64_t(size_) * 4) ||
         Fail("Compute initialization failed");
}

bool ComputeSimulation::GeometryFor(const Parameters &p) {
  auto l = levels_[0];
  int n = l.w * l.h;
  if (p.smooth_boundary) {
    if (!Run(SmoothGeometry, n, l) || !Run(ActiveCells, n, l))
      return false;
  } else if (!Run(Geometry, n, l) || !Run(FineOperator, n, l))
    return false;
  for (size_t i = 1; i < levels_.size(); ++i)
    if (!Run(CoarsenOperator, levels_[i].w * levels_[i].h, levels_[i - 1], levels_[i]))
      return false;
  return true;
}

bool ComputeSimulation::Cycle(size_t index) {
  auto l = levels_[index];
  int n = l.w * l.h;
  if (index + 1 == levels_.size())
    return Run(CoarseSolve, 1, l);
  for (int sweep = 0; sweep < 3; ++sweep)
    for (int color = 0; color < 2; ++color)
      if (!Run(Smooth, n, l, {}, color))
        return false;
  auto c = levels_[index + 1];
  if (!Run(Residual, n, l) || !Run(Restrict, c.w * c.h, l, c) || !Cycle(index + 1) || !Run(Prolong, n, l, c))
    return false;
  for (int sweep = 0; sweep < 3; ++sweep)
    for (int color = 0; color < 2; ++color)
      if (!Run(Smooth, n, l, {}, color))
        return false;
  return true;
}

bool ComputeSimulation::Solve(int cycles) {
  for (int i = 0; i < cycles; ++i)
    if (!Cycle(0))
      return false;
  return true;
}

bool ComputeSimulation::Reset(const Parameters &p) {
  time_ = 0;
  last_dt_ = 1;
  parameters_ = p;
  loads_valid_ = false;
  Update(p, last_dt_);
  auto l = levels_[0];
  int n = l.w * l.h, faces = l.w * (l.h + 1);
  return GeometryFor(p) && Run(InitializeVelocity, faces, l) && Run(Boundaries, faces, l) && Run(InitializeDye, n, l) &&
         Run(RHS, n, l) && Solve(32) && Run(Project, faces, l);
}

bool ComputeSimulation::Step(const Parameters &p, float dt) {
  Update(p, dt);
  last_dt_ = dt;
  if (p.angle != parameters_.angle || p.profile != parameters_.profile ||
      p.smooth_boundary != parameters_.smooth_boundary)
    if (!GeometryFor(p))
      return false;
  parameters_ = p;
  loads_valid_ = true;
  auto l = levels_[0];
  int n = l.w * l.h, faces = l.w * (l.h + 1);
  if (!Run(Boundaries, faces, l) || !Run(Advect, faces, l))
    return false;
  settings_.fields0 = {next_u_, next_v_, u_, v_};
  if (!Run(Boundaries, faces, l) || !Run(Diffuse, faces, l))
    return false;
  settings_.fields0 = {u_, v_, next_u_, next_v_};
  if (!Run(Boundaries, faces, l) || !Run(RHS, n, l) || !Solve(p.cycles) || !Run(Project, faces, l))
    return false;
  time_ += dt;
  settings_.values1[0] = time_;
  if (!Run(AdvectDye, n, l) || !Run(CorrectDye, n, l))
    return false;
  std::swap(dye_, corrected_dye_);
  settings_.fields1 = {mask_, dye_, next_dye_, corrected_dye_};
  return true;
}

bool ComputeSimulation::SaveState(SimulationState &state) {
  auto l = levels_[0];
  state = {};
  state.width = l.w;
  state.height = l.h;
  state.parameters = parameters_;
  state.time = time_;
  state.last_dt = last_dt_;
  state.loads_valid = loads_valid_;
  state.u.resize((l.w + 1) * l.h);
  state.v.resize(l.w * (l.h + 1));
  state.pressure.resize(l.w * l.h);
  state.rhs.resize(l.w * l.h);
  state.residual.resize(l.w * l.h);
  state.dye.resize(l.w * l.h);
  return Read(u_, state.u) && Read(v_, state.v) && Read(l.p, state.pressure) && Read(l.b, state.rhs) &&
         Read(l.r, state.residual) && Read(dye_, state.dye);
}

bool ComputeSimulation::RestoreState(const SimulationState &state) {
  auto l = levels_[0];
  if (!state.Matches(l.w, l.h))
    return Fail("Incompatible simulation state");
  parameters_ = state.parameters;
  time_ = state.time;
  last_dt_ = state.last_dt;
  loads_valid_ = state.loads_valid;
  Update(parameters_, last_dt_);
  if (!GeometryFor(parameters_) || !Begin())
    return false;
  for (auto pair : {std::make_pair(&state.u, u_),
                    {&state.v, v_},
                    {&state.pressure, l.p},
                    {&state.rhs, l.b},
                    {&state.residual, l.r},
                    {&state.dye, dye_}})
    if (!commands_->CmdUploadBuffer(heap_.Get(), uint64_t(pair.second) * 4, pair.first->data(), pair.first->size() * 4))
      return Fail("State upload failed");
  return (Flush() && pending_->WaitHost()) || Fail("State transfer failed");
}

bool ComputeSimulation::Export() {
  auto l = levels_[0];
  return Run(Display, l.w * l.h, l) && Flush();
}

bool ComputeSimulation::Read(uint32_t offset, std::vector<float> &values) {
  if (!Begin())
    return false;
  ObjPtr<IDownloadToken> download = commands_->CmdDownloadBuffer(heap_.Get(), uint64_t(offset) * 4, values.size() * 4);
  return (download && Flush() && download->WaitHost() && download->CopyTo(values.data(), values.size() * 4)) ||
         Fail("Compute readback failed");
}

bool ComputeSimulation::Reduced(int count, bool diagnostic, std::array<double, 7> &result) {
  if (!Run(Reduce, count, levels_[0], {}, 0, diagnostic))
    return false;
  std::vector<float> values(((count + 255) / 256) * 7);
  if (!Read(reduced_, values))
    return false;
  result = {};
  for (size_t i = 0; i < values.size(); ++i) {
    if (!std::isfinite(values[i]))
      return Fail("Nonfinite GPU reduction");
    int k = i % 7;
    if (diagnostic && k >= 4)
      result[k] = std::max(result[k], double(values[i]));
    else
      result[k] += values[i];
  }
  return true;
}

bool ComputeSimulation::Measure(Diagnostics &d) {
  auto l = levels_[0];
  int n = l.w * l.h;
  settings_.values0[3] = last_dt_;
  std::array<double, 7> values;
  if (!Run(Residual, n, l) || !Run(MeasureFields, n, l) || !Reduced(n, true, values))
    return false;
  double count = std::max(1.0, values[3]);
  d = {float(std::sqrt(values[0] / count)),
       float(std::sqrt(values[1] / count)),
       float(std::sqrt(values[2] / count)),
       float(values[4]),
       float(values[5]),
       values[6] == 0};
  return true;
}

bool ComputeSimulation::ReadLoads(std::array<double, 7> &values, int analytic) {
  int count = parameters_.smooth_boundary ? 512 : levels_[0].w * levels_[0].h;
  return Run(parameters_.smooth_boundary ? SmoothLoads : VoxelLoads, count, levels_[0], {}, analytic) &&
         Reduced(count, false, values);
}

bool ComputeSimulation::MeasureLoads(LoadCoefficients &load) {
  load = {};
  load.angle = parameters_.angle;
  load.profile = parameters_.profile;
  if (!loads_valid_)
    return true;
  std::array<double, 7> values;
  if (!ReadLoads(values))
    return false;
  double ref = 0.5 * parameters_.speed * parameters_.speed * chord;
  load.x = float((values[0] + values[3]) / ref);
  load.y = float((values[1] + values[4]) / ref);
  load.moment = float((values[2] + values[5]) / (ref * chord));
  load.valid = std::isfinite(load.x) && std::isfinite(load.y) && std::isfinite(load.moment);
  return load.valid || Fail("Nonfinite fluid load");
}

bool ComputeSimulation::ValidateLoads(float &relative_error) {
  auto l = levels_[0];
  double area = 0, mx = 0, my = 0;
  if (parameters_.smooth_boundary) {
    auto point = [&](int index) {
      int k = index <= 256 ? index : 512 - index;
      float q = 0.5f * (1 - std::cos(k * (6.28318530718f / 512)));
      float x = (q - 0.5f) * chord,
            y = (WingCamber(q, parameters_.profile) + WingThickness(q, parameters_.profile) * (index <= 256 ? -1 : 1)) *
                chord;
      float a = parameters_.angle * 0.01745329252f;
      return std::array<double, 2>{x * std::cos(a) + y * std::sin(a), -x * std::sin(a) + y * std::cos(a)};
    };
    for (int i = 0; i < 512; ++i) {
      auto a = point(i), b = point(i + 1);
      double cross = a[0] * b[1] - b[0] * a[1];
      area += cross / 2;
      mx += (a[0] + b[0]) * cross / 6;
      my += (a[1] + b[1]) * cross / 6;
    }
  } else {
    std::vector<float> mask(l.w * l.h);
    if (!Read(mask_, mask))
      return false;
    double h = domain_width / l.w;
    for (size_t i = 0; i < mask.size(); ++i)
      if (!mask[i]) {
        area += h * h;
        mx += h * h * ((i % l.w + 0.5) * h - 2);
        my += h * h * ((i / l.w + 0.5) * h - 1.5);
      }
  }
  std::array<double, 7> constant, linear;
  if (!ReadLoads(constant, 1) || !ReadLoads(linear, 2))
    return false;
  double expected[3] = {-0.7 * area, 0.4 * area, 0.4 * mx + 0.7 * my}, error = 0;
  for (int k = 0; k < 3; ++k)
    error = std::max({error, std::abs(constant[k]), std::abs(linear[k] - expected[k])});
  relative_error = float(error / std::max(area, 1e-20));
  return true;
}

bool ComputeSimulation::ValidatePressure(float &residual, float &error) {
  auto l = levels_[0];
  int n = l.w * l.h;
  if (!Run(Manufactured, n, l, {}, 0) || !Run(Manufactured, n, l, {}, 1) || !Solve(40) || !Run(Residual, n, l))
    return false;
  std::vector<float> b(n), r(n), p(n), exact(n), mask(n);
  if (!Read(l.b, b) || !Read(l.r, r) || !Read(l.p, p) || !Read(next_dye_, exact) || !Read(mask_, mask))
    return false;
  double b2 = 0, r2 = 0, e2 = 0, p2 = 0;
  for (int i = 0; i < n; ++i) {
    b2 += double(b[i]) * b[i];
    r2 += double(r[i]) * r[i];
    if (mask[i]) {
      e2 += double(p[i] - exact[i]) * (p[i] - exact[i]);
      p2 += double(exact[i]) * exact[i];
    }
  }
  residual = float(std::sqrt(r2 / std::max(b2, 1e-30)));
  error = float(std::sqrt(e2 / std::max(p2, 1e-30)));
  return true;
}
}  // namespace

std::unique_ptr<Simulation> CreateComputeSimulation(grassland::rhi::IDevice *device, int width, int height) {
  auto result = std::make_unique<ComputeSimulation>();
  if (!result->Initialize(device, width, height)) {
    std::cerr << result->Error() << '\n';
    return {};
  }
  return result;
}
}  // namespace wing_fluid
