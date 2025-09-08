#pragma once
#include "grassland/util/util_util.h"

namespace CD {
std::vector<uint32_t> SobolTableGen(unsigned int N, unsigned int D, const std::string &dir_file);
}
