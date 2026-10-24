#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace grassland {
// Return 0 on success. Paths use forward slashes; failed reads leave no usable data.
class FileSystem {
 public:
  virtual ~FileSystem() = default;
  virtual int ReadFile(const std::string &file_path, std::vector<uint8_t> &data) const = 0;
};
}  // namespace grassland
