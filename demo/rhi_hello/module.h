#pragma once
#include "grassland/rhi/rhi_util.h"
int RunWindow(grassland::rhi::BackendAPI api, int frames, bool headless);
int RunTriangle(grassland::rhi::BackendAPI api, int frames, bool headless);

int RunCube(grassland::rhi::BackendAPI api, int frames, bool headless);
int RunTexture(grassland::rhi::BackendAPI api, int frames, bool headless);
int RunBlend(grassland::rhi::BackendAPI api, int frames, bool headless);

int RunNBody(grassland::rhi::BackendAPI api, int frames, bool headless);

int RunRayTracing(grassland::rhi::BackendAPI api, int frames, bool headless);
int RunRayQuery(grassland::rhi::BackendAPI api, int frames, bool headless);

int RunHDR(grassland::rhi::BackendAPI api, int frames, bool headless);
