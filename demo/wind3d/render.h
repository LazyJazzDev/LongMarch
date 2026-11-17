#pragma once
#include "grassland/rhi/rhi.h"
#include "simulation.h"

namespace wind3d {
struct ViewSettings {
  float yaw{-0.85f}, pitch{0.40f}, distance{10.5f};
  bool flow{false}, loads{true}, volume{true};
  int streamline_density{2};
  float streamline_width{1.5f};
};

bool Render(grassland::rhi::IDevice *device,
            Simulation &simulation,
            Controls controls,
            int frames,
            bool headless,
            const std::string &capture,
            const ViewSettings &view = {});
}  // namespace wind3d
