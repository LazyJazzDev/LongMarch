#pragma once
#include <string>
#include <vector>

#include "grassland/rhi/rhi.h"

bool SaveImage(const std::string &directory_name,
               const std::string &name,
               const std::vector<uint8_t> &pixels,
               const grassland::rhi::ImageDesc &desc,
               grassland::rhi::ColorSpace space,
               float white_scale);
