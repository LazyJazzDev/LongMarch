#pragma once

#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {
// Per rendering thread. Mobile bundles are read-only; preparation runs on a Mac.
struct ShaderCacheSettings {
  std::filesystem::path directory;
  bool read_only{true};
  bool ios{true};
};
void ConfigureShaderCache(const ShaderCacheSettings &settings);
const ShaderCacheSettings &GetShaderCacheSettings();
std::string ShaderCacheKey(const std::vector<std::string> &parts);
bool ReadShaderCache(const std::string &key, std::vector<uint8_t> &data);
void WriteShaderCache(const std::string &key, const std::vector<uint8_t> &data);
std::string ShaderRequestKey(const VirtualFileSystem &vfs,
                             const std::string &file,
                             const std::string &entry,
                             const std::string &target,
                             const std::vector<std::string> &args);
}  // namespace grassland::graphics
