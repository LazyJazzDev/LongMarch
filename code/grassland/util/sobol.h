#pragma once
#include <cstdint>
#include "grassland/util/util_util.h"

namespace grassland {
std::vector<uint32_t> SobolTableGen(unsigned int N, unsigned int D, const std::string &dir_file);
}
