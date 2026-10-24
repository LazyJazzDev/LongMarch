#pragma once
#include <string>

#include "grassland/rhi/rhi_util.h"
int RunWindow(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});
int RunTriangle(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});

int RunCube(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});
int RunTexture(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});
int RunBlend(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});

int RunNBody(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});

int RunRayTracing(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});
int RunRayQuery(grassland::rhi::BackendAPI api, int frames, bool headless, const std::string &capture_directory = {});

int RunHDR(grassland::rhi::BackendAPI api, int frames, const std::string &capture_directory = {});

int RunImGui(grassland::rhi::BackendAPI api,
             int frames,
             bool headless,
             const char *font_file,
             float font_size,
             const std::string &capture_directory = {});
int RunImGuiMultiWindow(grassland::rhi::BackendAPI api,
                        int frames,
                        bool headless,
                        const char *font_file,
                        float font_size,
                        const std::string &capture_directory = {});
