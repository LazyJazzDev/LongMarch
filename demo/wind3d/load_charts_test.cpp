#include "load_charts.h"

#include <iostream>

int main() {
  using namespace wind3d;
  auto near = [](double a, double b) { return std::abs(a - b) < 1e-5; };
  LoadHistory history;
  double times[] = {.1, .3, .6};
  float dt[] = {.1f, .2f, .3f}, values[] = {10, 20, 40};
  for (int i = 0; i < 3; ++i) {
    LoadSample s{};
    s.step = i + 1;
    s.time = times[i];
    s.dt = dt[i];
    s.force.x = values[i];
    history.samples.push_back(s);
  }
  auto w = load_charts::Select(history, .4, 0, false);
  if (w.valid_count != 2 || !near(w.statistics[0].mean, 35) || !near(w.statistics[0].deviation, std::sqrt(75.0)))
    return 1;
  w = load_charts::Select(history, .4, 2, false);
  if (w.valid_count != 1 || !near(w.statistics[0].mean, 40))
    return 2;
  history.samples.clear();
  for (int i = 1; i <= 24; ++i) {
    LoadSample s{};
    s.step = i;
    s.time = i * .125;
    s.dt = .125f;
    s.force.x = float(i);
    s.controls.angle = i >= 4 ? 10.f : 0.f;
    history.samples.push_back(s);
  }
  w = load_charts::Select(history, 2, 0, true);
  if (w.valid_count != 1 || !near(w.statistics[0].mean, 24))
    return 3;
  w = load_charts::Select(history, 3, 0, true);
  if (w.valid_count != 4 || !near(w.statistics[0].mean, 7.5))
    return 4;
  w = load_charts::Select(history, 3, 0, false);
  if (w.valid_count != 24 || !near(w.statistics[0].mean, 12.5))
    return 5;
  history.samples[12].step += 2;
  if (load_charts::Select(history, 3, 0, false).contiguous)
    return 6;
  if (load_charts::Select({}, 3, 0, false).valid_count)
    return 7;
  std::cout << "PASS: weighted windows, startup filter, control-change gaps, missing steps and reset\n";
}
