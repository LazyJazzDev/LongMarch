#include "grassland/graphics/shader_cache.h"

#ifdef __APPLE__
#include <CommonCrypto/CommonDigest.h>

#include <fstream>
#include <stdexcept>

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
  CC_SHA256_CTX hash;
  CC_SHA256_Init(&hash);
  for (const auto &part : parts) {
    // Decimal length plus separator makes field boundaries unambiguous on every architecture.
    auto length = std::to_string(part.size()) + ":";
    CC_SHA256_Update(&hash, length.data(), static_cast<CC_LONG>(length.size()));
    CC_SHA256_Update(&hash, part.data(), static_cast<CC_LONG>(part.size()));
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &hash);
  const char *hex = "0123456789abcdef";
  std::string result;
  for (auto value : digest) {
    result += hex[value >> 4];
    result += hex[value & 15];
  }
  return result;
}

bool ReadShaderCache(const std::string &key, std::vector<uint8_t> &data) {
  if (settings.directory.empty())
    return false;
  std::ifstream input(settings.directory / key, std::ios::binary | std::ios::ate);
  if (!input) {
    if (settings.read_only)
      throw std::runtime_error("Missing bundled shader " + key + "; regenerate the iOS resource bundle on macOS");
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
  std::vector<std::string> parts{"sparkium-dxc-release-v1", file, entry, target};
  parts.push_back(std::to_string(args.size()));
  parts.insert(parts.end(), args.begin(), args.end());
  for (const auto &path : vfs.ListFiles()) {
    std::vector<uint8_t> data;
    vfs.ReadFile(path, data);
    parts.push_back(path);
    parts.emplace_back(data.begin(), data.end());
  }
  return "hlsl-" + ShaderCacheKey(parts);
}
}  // namespace grassland::graphics
#endif
