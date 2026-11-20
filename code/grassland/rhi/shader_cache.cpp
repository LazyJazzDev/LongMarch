#include "grassland/rhi/shader_cache.h"

#include <algorithm>
#include <fstream>

#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/sha256.h"

namespace grassland::rhi {
namespace {
thread_local ShaderCacheSettings settings;

bool ValidKey(const std::string &key) {
  return !key.empty() && std::all_of(key.begin(), key.end(), [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-';
  });
}
}  // namespace

bool ConfigureShaderCache(const ShaderCacheSettings &value) {
  if (!value.directory.empty()) {
    std::error_code error;
    if (!value.read_only)
      std::filesystem::create_directories(value.directory, error);
    if (error || !std::filesystem::is_directory(value.directory, error)) {
      backend::ReportError(-1, "Shader cache directory is unavailable: " + value.directory.string());
      return false;
    }
  }
  settings = value;
  return true;
}

const ShaderCacheSettings &GetShaderCacheSettings() {
  return settings;
}

std::string ShaderCacheKey(const std::vector<std::string> &parts) {
  detail::SHA256 hash;
  for (auto &p : parts) {
    hash.Update(std::to_string(p.size()) + ":");
    hash.Update(p);
  }
  return hash.Finish();
}

bool ReadShaderCache(const std::string &key, std::vector<uint8_t> &data) {
  data.clear();
  if (settings.directory.empty() || !ValidKey(key))
    return false;
  std::ifstream file(settings.directory / key, std::ios::binary | std::ios::ate);
  if (!file) {
    if (settings.read_only)
      backend::ReportError(-1, "Missing bundled shader " + key);
    return false;
  }
  auto n = file.tellg();
  if (n < 64 || n > 128 * 1024 * 1024) {
    backend::ReportError(-1, "Invalid shader cache size: " + key);
    return false;
  }
  file.seekg(0);
  std::string digest(64, '\0');
  file.read(digest.data(), 64);
  std::vector<uint8_t> bytes(static_cast<size_t>(n) - 64);
  file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!file || ShaderCacheKey({std::string(bytes.begin(), bytes.end())}) != digest) {
    backend::ReportError(-1, "Corrupt shader cache: " + key);
    return false;
  }
  data = std::move(bytes);
  return true;
}

bool WriteShaderCache(const std::string &key, const std::vector<uint8_t> &data) {
  if (!ValidKey(key))
    return false;
  if (settings.directory.empty() || settings.read_only)
    return true;
  auto path = settings.directory / key;
  auto temp = path.string() + ".tmp";
  std::ofstream file(temp, std::ios::binary | std::ios::trunc);
  file << ShaderCacheKey({std::string(data.begin(), data.end())});
  file.write(reinterpret_cast<const char *>(data.data()), data.size());
  file.close();
  if (!file) {
    backend::ReportError(-1, "Cannot write shader cache " + key);
    return false;
  }
  std::error_code ec;
  std::filesystem::rename(temp, path, ec);
  if (ec)
    backend::ReportError(-1, "Cannot finalize shader cache " + key + ": " + ec.message());
  return !ec;
}

bool ReadShaderThreadGroup(const std::string &key, std::array<uint32_t, 3> &size) {
  std::vector<uint8_t> bytes;
  if (!ReadShaderCache("threads-" + key, bytes) || bytes.size() != 12)
    return false;
  for (size_t i = 0; i < 3; ++i) {
    size[i] = 0;
    for (size_t j = 0; j < 4; ++j)
      size[i] |= uint32_t(bytes[i * 4 + j]) << (j * 8);
    if (!size[i])
      return false;
  }
  return true;
}

bool WriteShaderThreadGroup(const std::string &key, const std::array<uint32_t, 3> &size) {
  std::vector<uint8_t> bytes(12);
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 4; ++j)
      bytes[i * 4 + j] = uint8_t(size[i] >> (j * 8));
  return WriteShaderCache("threads-" + key, bytes);
}
}  // namespace grassland::rhi
