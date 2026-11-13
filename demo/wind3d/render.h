#pragma once
#include "grassland/rhi/rhi.h"
#include "simulation.h"

namespace wind3d {
bool Render(grassland::rhi::IDevice *device,
            Simulation &simulation,
            Controls controls,
            int frames,
            bool headless,
            const std::string &capture);
}
