#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace grassland::rhi {
// Per-thread offline bundle configuration. Prepare and replay must use the same
// shader sources, compiler version, platform macros and pipeline layouts.
struct ShaderCacheSettings {
  std::filesystem::path directory;
  bool read_only{true};
  // iOS-compatible MSL, including the simulator's Tier 1 argument buffers.
  bool ios{false};
};

void ConfigureShaderCache(const ShaderCacheSettings &settings);
const ShaderCacheSettings &GetShaderCacheSettings();
std::string ShaderCacheKey(const std::vector<std::string> &parts);
bool ReadShaderCache(const std::string &key, std::vector<uint8_t> &data);
bool WriteShaderCache(const std::string &key, const std::vector<uint8_t> &data);
bool ReadShaderThreadGroup(const std::string &key, std::array<uint32_t, 3> &size);
bool WriteShaderThreadGroup(const std::string &key, const std::array<uint32_t, 3> &size);
}  // namespace grassland::rhi
