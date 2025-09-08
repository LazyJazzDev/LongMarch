#include "grassland/util/fps_counter.h"

namespace CD {

void FPSCounter::TickFrame() {
  auto tp = std::chrono::high_resolution_clock::now();
  frames_.emplace(tp);
  tp -= std::chrono::seconds(1);
  while (frames_.size() > 2 && frames_.front() < tp) {
    frames_.pop();
  }
}

float FPSCounter::GetFPS() const {
  if (frames_.size() < 2) {
    // return nan
    return 0.0f;
  } else {
    float duration_second = std::chrono::duration<float>(frames_.back() - frames_.front()).count();
    return (frames_.size() - 1) / duration_second;
  }
}

float FPSCounter::TickFPS() {
  TickFrame();
  return GetFPS();
}

void FPSCounter::Reset() {
  while (!frames_.empty()) {
    frames_.pop();
  }
}

void FPSCounter::PyBind(pybind11::module_ &m) {
  pybind11::class_<FPSCounter> fps_counter(m, "FPSCounter");
  fps_counter.def(pybind11::init<>())
      .def("tick_frame", &FPSCounter::TickFrame)
      .def("get_fps", &FPSCounter::GetFPS)
      .def("tick_fps", &FPSCounter::TickFPS)
      .def("reset", &FPSCounter::Reset);
}

}  // namespace CD
