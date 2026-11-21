#pragma once
#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace grassland::rhi {
enum class ShaderTargetPlatform { Native, Windows, MacOS, Linux, IOS, OHOS };

// Per-thread offline bundle configuration. Prepare and replay must use the same
// shader sources, compiler version, platform macros and pipeline layouts.
struct ShaderCacheSettings {
  std::filesystem::path directory;
  bool read_only{true};
  // iOS-compatible MSL, including the simulator's Tier 1 argument buffers.
  bool ios{false};
  // Explicit cross-compilation target. Native preserves host detection and ios.
  ShaderTargetPlatform target_platform{ShaderTargetPlatform::Native};
};

// Configure before creating shaders/programs on this thread. Keep the target
// policy unchanged while using those programs/bindings. Preparation is a single
// writer operation; regenerate the entire bundle when sources/includes or the
// host compiler change. An empty directory disables caching in online builds.
// Failure leaves the previous configuration unchanged.
bool ConfigureShaderCache(const ShaderCacheSettings &settings);
const ShaderCacheSettings &GetShaderCacheSettings();
std::string ShaderCacheKey(const std::vector<std::string> &parts);
bool ReadShaderCache(const std::string &key, std::vector<uint8_t> &data);
bool WriteShaderCache(const std::string &key, const std::vector<uint8_t> &data);
bool ReadShaderThreadGroup(const std::string &key, std::array<uint32_t, 3> &size);
bool WriteShaderThreadGroup(const std::string &key, const std::array<uint32_t, 3> &size);
}  // namespace grassland::rhi
