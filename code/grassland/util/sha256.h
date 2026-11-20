#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace grassland {
// Streaming SHA-256 for resource integrity, independent of platform crypto SDKs.
class SHA256 {
 public:
  void Update(std::string_view bytes);

  std::string Finish();

 private:
  void Transform();

  std::array<uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<uint8_t, 64> block_{};
  size_t used_ = 0;
  uint64_t length_ = 0;
};
}  // namespace grassland
