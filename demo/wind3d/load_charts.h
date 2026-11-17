#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "simulation.h"

namespace wind3d::load_charts {
struct Statistic {
  double mean{}, deviation{}, minimum{std::numeric_limits<double>::infinity()},
      maximum{-std::numeric_limits<double>::infinity()};
};

struct Window {
  size_t first{}, valid_count{};
  double start{}, end{}, dt_min{}, dt_max{};
  bool contiguous{true};
  std::vector<unsigned char> valid;
  std::vector<size_t> changes;
  std::array<Statistic, 6> statistics;
};

inline bool SameControls(const Controls &a, const Controls &b) {
  return a.speed == b.speed && a.angle == b.angle && a.flap == b.flap && a.aileron == b.aileron &&
         a.elevator == b.elevator && a.rudder == b.rudder && SameAirframe(a.airframe, b.airframe);
}

inline double Value(const LoadSample &sample, int component) {
  Vec3 v = component < 3 ? sample.force : sample.moment;
  return component % 3 == 0 ? v.x : component % 3 == 1 ? v.y : v.z;
}

inline Window Select(const LoadHistory &history, double seconds, uint64_t skip_steps, bool hide_changes) {
  Window result;
  const auto &samples = history.samples;
  result.valid.resize(samples.size());
  if (samples.empty())
    return result;
  double cutoff = samples.back().time - seconds;
  while (result.first < samples.size() &&
         (samples[result.first].time < cutoff || samples[result.first].step <= skip_steps))
    ++result.first;
  if (result.first == samples.size())
    return result;
  result.start = samples[result.first].time;
  result.end = samples.back().time;
  result.dt_min = std::numeric_limits<double>::infinity();
  uint64_t invalid_until = 0;
  double weight = 0;
  for (size_t i = 0; i < samples.size(); ++i) {
    const auto &s = samples[i];
    if (i && !SameControls(samples[i - 1].controls, s.controls)) {
      result.changes.push_back(i);
      invalid_until = s.step + 20;
    }
    if (i < result.first)
      continue;
    if (i > result.first && s.step != samples[i - 1].step + 1)
      result.contiguous = false;
    result.dt_min = std::min(result.dt_min, double(s.dt));
    result.dt_max = std::max(result.dt_max, double(s.dt));
    bool valid = !(hide_changes && s.step < invalid_until);
    for (int k = 0; k < 6; ++k)
      valid &= std::isfinite(Value(s, k));
    if (!valid)
      continue;
    result.valid[i] = 1;
    ++result.valid_count;
    double dt = std::max(0.0, std::min(double(s.dt), s.time - cutoff));
    double next_weight = weight + dt;
    for (int k = 0; k < 6; ++k) {
      auto &stat = result.statistics[k];
      double value = Value(s, k);
      stat.minimum = std::min(stat.minimum, value);
      stat.maximum = std::max(stat.maximum, value);
      if (next_weight > 0) {
        double delta = value - stat.mean;
        stat.mean += delta * dt / next_weight;
        stat.deviation += dt * delta * (value - stat.mean);
      }
    }
    weight = next_weight;
  }
  for (auto &stat : result.statistics)
    stat.deviation = weight > 0 ? std::sqrt(std::max(0.0, stat.deviation / weight)) : 0;
  return result;
}

struct Settings {
  int window_index{2}, skip_index{2};
  bool demean{}, hide_changes{true};
};

// Call from the existing ImGui frame. Returns reserved logical viewport height.
float Draw(const LoadHistory &history, Settings &settings, bool &visible, float panel_width);
}  // namespace wind3d::load_charts
