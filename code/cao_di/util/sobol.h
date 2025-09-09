#pragma once
#include "cao_di/util/util_util.h"

namespace CD {
std::vector<uint32_t> SobolTableGen(unsigned int N, unsigned int D, const std::string &dir_file);
}
