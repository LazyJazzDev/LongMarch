#pragma once

namespace grassland::rhi {
class IDevice;
}

// Launcher declarations only; examples do not depend on this header.
int RunWindow(grassland::rhi::IDevice *device);
int RunTriangle(grassland::rhi::IDevice *device);
int RunCube(grassland::rhi::IDevice *device);
int RunTexture(grassland::rhi::IDevice *device);
int RunBlend(grassland::rhi::IDevice *device);
int RunNBody(grassland::rhi::IDevice *device);
int RunRayTracing(grassland::rhi::IDevice *device);
int RunRayQuery(grassland::rhi::IDevice *device);
int RunHDR(grassland::rhi::IDevice *device);
int RunImGui(grassland::rhi::IDevice *device);
int RunImGuiMultiWindow(grassland::rhi::IDevice *device);
