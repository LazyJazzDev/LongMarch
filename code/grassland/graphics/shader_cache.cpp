#include "grassland/graphics/shader_cache.h"

#include <fstream>
#include <stdexcept>

#include "grassland/graphics/sha256.h"

namespace grassland::graphics {
namespace {
thread_local ShaderCacheSettings settings;
}

void ConfigureShaderCache(const ShaderCacheSettings &value) {
  settings = value;
  if (!settings.directory.empty() && !settings.read_only)
    std::filesystem::create_directories(settings.directory);
}

const ShaderCacheSettings &GetShaderCacheSettings() {
  return settings;
}

std::string ShaderCacheKey(const std::vector<std::string> &parts) {
  detail::SHA256 hash;
  for (const auto &part : parts) {
    // Preserve the iOS cache format, including unambiguous field lengths.
    hash.Update(std::to_string(part.size()) + ":");
    hash.Update(part);
  }
  return hash.Finish();
}

bool ReadShaderCache(const std::string &key, std::vector<uint8_t> &data) {
  if (settings.directory.empty())
    return false;
  std::ifstream input(settings.directory / key, std::ios::binary | std::ios::ate);
  if (!input) {
    if (settings.read_only)
      throw std::runtime_error("Missing bundled shader " + key + "; regenerate the resource bundle for this backend");
    return false;
  }
  auto length = input.tellg();
  if (length <= 64 || length > 128 * 1024 * 1024)
    throw std::runtime_error("Invalid shader cache size: " + key);
  input.seekg(0);
  std::string digest(64, '\0');
  input.read(digest.data(), 64);
  data.resize(static_cast<size_t>(length) - 64);
  input.read(reinterpret_cast<char *>(data.data()), data.size());
  if (!input || ShaderCacheKey({std::string(data.begin(), data.end())}) != digest)
    throw std::runtime_error("Corrupt bundled shader: " + key);
  return true;
}

void WriteShaderCache(const std::string &key, const std::vector<uint8_t> &data) {
  if (settings.directory.empty() || settings.read_only)
    return;
  auto path = settings.directory / key;
  auto temporary = path.string() + ".tmp";
  std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
  output << ShaderCacheKey({std::string(data.begin(), data.end())});
  output.write(reinterpret_cast<const char *>(data.data()), data.size());
  output.close();
  if (!output)
    throw std::runtime_error("Cannot write shader cache " + temporary);
  std::filesystem::rename(temporary, path);
}

std::string ShaderRequestKey(const VirtualFileSystem &vfs,
                             const std::string &file,
                             const std::string &entry,
                             const std::string &target,
                             const std::vector<std::string> &args) {
  std::vector<std::string> parts{"sparkium-slang-release-v2", file, entry, target};
  parts.push_back(std::to_string(args.size()));
  parts.insert(parts.end(), args.begin(), args.end());
  for (const auto &path : vfs.ListFiles()) {
    std::vector<uint8_t> data;
    vfs.ReadFile(path, data);
    parts.push_back(path);
    parts.emplace_back(data.begin(), data.end());
  }
  return "slang-" + ShaderCacheKey(parts);
}
}  // namespace grassland::graphics
