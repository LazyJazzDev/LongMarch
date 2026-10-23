#include "grassland/util/virtual_file_system.h"

#include <algorithm>
#include <fstream>
#include <optional>

namespace grassland {
namespace {
std::optional<std::string> NormalizeFilePath(std::string name) {
  // std::filesystem only recognizes backslashes as separators on Windows.
  // Convert first so virtual paths have the same meaning on every platform.
  if (name.find('\0') != std::string::npos || name.find(':') != std::string::npos)
    return std::nullopt;
  std::replace(name.begin(), name.end(), '\\', '/');
  const auto first = name.find_first_not_of('/');
  if (first == std::string::npos)
    return std::nullopt;
  // Strip the virtual root before normalization so /../file cannot silently
  // clamp to /file. No path is interpreted as a host drive or network share.
  const auto path = std::filesystem::u8path(name.substr(first));
  const auto leaf = path.filename();
  if (leaf.empty() || leaf == "." || leaf == "..")
    return std::nullopt;
  const auto normalized = path.lexically_normal();
  if (normalized.empty() || *normalized.begin() == "..")
    return std::nullopt;
  return normalized.generic_u8string();
}

bool StartsWith(const std::string &value, const std::string &prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}
}  // namespace

VirtualFileSystem::VirtualFileSystem() = default;
VirtualFileSystem::VirtualFileSystem(const VirtualFileSystem &other) = default;
VirtualFileSystem::VirtualFileSystem(VirtualFileSystem &&other) noexcept = default;
VirtualFileSystem &VirtualFileSystem::operator=(const VirtualFileSystem &other) = default;

VirtualFileSystem &VirtualFileSystem::operator=(VirtualFileSystem &&other) noexcept {
  if (this != &other)
    files_ = std::move(other.files_);
  return *this;
}

int VirtualFileSystem::WriteFile(const std::string &file_name, const std::vector<uint8_t> &data) {
  const auto key = NormalizeFilePath(file_name);
  if (!key)
    return -1;
  // A file cannot replace an implicit directory, or live beneath another file.
  const auto prefix = *key + '/';
  const auto child = files_.lower_bound(prefix);
  if (child != files_.end() && StartsWith(child->first, prefix))
    return -1;
  auto parent = std::filesystem::u8path(*key).parent_path();
  while (!parent.empty()) {
    if (files_.find(parent.generic_u8string()) != files_.end())
      return -1;
    parent = parent.parent_path();
  }
  files_.insert_or_assign(*key, data);
  return 0;
}

int VirtualFileSystem::WriteFile(const std::string &file_name, const std::string &data) {
  return WriteFile(file_name, data.data(), data.size());
}

int VirtualFileSystem::WriteFile(const std::string &file_name, const char *data, size_t size) {
  if (!data && size)
    return -1;
  std::vector<uint8_t> bytes;
  if (size)
    bytes.assign(data, data + size);
  return WriteFile(file_name, bytes);
}

int VirtualFileSystem::ReadFile(const std::string &file_name, std::vector<uint8_t> &data) const {
  const auto key = NormalizeFilePath(file_name);
  if (!key)
    return -1;
  const auto file = files_.find(*key);
  if (file == files_.end())
    return -1;
  data = file->second;
  return 0;
}

void VirtualFileSystem::Print() const {
  for (const auto &[name, data] : files_)
    printf("%s (%zu bytes)\n", name.c_str(), data.size());
}

VirtualFileSystem VirtualFileSystem::LoadDirectory(const std::filesystem::path &path) {
  VirtualFileSystem vfs;
  for (const auto &entry : std::filesystem::recursive_directory_iterator(path)) {
    if (!entry.is_regular_file())
      continue;
    // A lexical relative path preserves the enumerated name even for symlinks.
    const auto relative_path = entry.path().lexically_relative(path);
    std::ifstream file(entry.path(), std::ios::binary);
    if (file) {
      file.seekg(0, std::ios::end);
      const auto size = file.tellg();
      if (size < 0)
        continue;
      file.seekg(0, std::ios::beg);
      std::vector<uint8_t> data(static_cast<size_t>(size));
      if (size && !file.read(reinterpret_cast<char *>(data.data()), size))
        continue;
      vfs.WriteFile(relative_path.generic_u8string(), data);
    }
  }
  return vfs;
}

void VirtualFileSystem::SaveToDirectory(const std::filesystem::path &path) const {
  std::filesystem::create_directories(path);
  for (const auto &[name, data] : files_) {
    const auto destination = path / std::filesystem::u8path(name);
    std::filesystem::create_directories(destination.parent_path());
    std::ofstream file(destination, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
  }
}
}  // namespace grassland
