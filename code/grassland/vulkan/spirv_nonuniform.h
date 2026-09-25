#pragma once

#include <algorithm>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>

namespace grassland::vulkan {

// DXC can retain NonUniform on a ByteAddressBuffer array index while losing it
// on the access-chain result when a resource is passed through a local/helper.
// Vulkan requires the pointer used by the load to carry the decoration.
inline std::vector<uint32_t> RestoreStorageBufferNonUniform(const std::vector<uint32_t> &code) {
  if (code.size() < 5 || code[0] != 0x07230203)
    throw std::invalid_argument("Invalid SPIR-V module");
  constexpr uint32_t kNonUniform = 5300, kStorageBufferCapability = 5308;
  std::set<uint32_t> nonuniform, storage_pointers, added;
  bool capability = false;
  size_t annotations = code.size();
  for (size_t i = 5; i < code.size();) {
    const uint32_t length = code[i] >> 16, op = code[i] & 0xffff;
    if (!length || length > code.size() - i)
      throw std::invalid_argument("Invalid SPIR-V instruction");
    if (op == 17 && length == 2 && code[i + 1] == kStorageBufferCapability)
      capability = true;
    if (op == 71 && length >= 3 && code[i + 2] == kNonUniform)
      nonuniform.insert(code[i + 1]);
    if (op == 32 && length == 4 && code[i + 2] == 12)
      storage_pointers.insert(code[i + 1]);
    // Insert annotations before the types/constants/global declarations.
    if (op >= 19 && op <= 39)
      annotations = std::min(annotations, i);
    i += length;
  }
  bool changed, storage_nonuniform = false;
  do {
    changed = false;
    for (size_t i = 5; i < code.size(); i += code[i] >> 16) {
      const uint32_t length = code[i] >> 16, op = code[i] & 0xffff;
      if ((op != 65 && op != 66) || length < 5 || !storage_pointers.count(code[i + 1]))
        continue;
      bool divergent = false;
      for (size_t j = 3; j < length; ++j)
        divergent |= nonuniform.count(code[i + j]) != 0;
      storage_nonuniform |= divergent || nonuniform.count(code[i + 2]) != 0;
      if (divergent && nonuniform.insert(code[i + 2]).second) {
        added.insert(code[i + 2]);
        changed = true;
      }
    }
  } while (changed);
  if (added.empty() && (!storage_nonuniform || capability))
    return code;
  std::vector<uint32_t> result(code.begin(), code.begin() + 5);
  if (!capability)
    result.insert(result.end(), {(2u << 16) | 17u, kStorageBufferCapability});
  result.insert(result.end(), code.begin() + 5, code.begin() + annotations);
  for (auto id : added)
    result.insert(result.end(), {(3u << 16) | 71u, id, kNonUniform});
  result.insert(result.end(), code.begin() + annotations, code.end());
  return result;
}

}  // namespace grassland::vulkan
