#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace life_demo {
// Little-endian, versioned, indexed nonempty tiles. Sparse tiles store only
// nonzero words; dense tiles store their 8 KiB bit payload without word indices.
bool WriteTiledState(std::ostream &output, const std::vector<uint32_t> &words, std::string &error);
bool ReadTiledState(std::istream &input, std::vector<uint32_t> &words, std::string &error);
inline constexpr char kTiledStateMagic[] = "GOLSPRS1";
}  // namespace life_demo
