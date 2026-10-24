#pragma once
#include <filesystem>
#include <map>

#include "grassland/util/file_system.h"
#include "grassland/util/util_util.h"

namespace grassland {

class VirtualFileSystemEntry {
 public:
  VirtualFileSystemEntry(VirtualFileSystemEntry *parent = nullptr) : parent_(parent) {
  }

  virtual ~VirtualFileSystemEntry() = default;

  virtual std::unique_ptr<VirtualFileSystemEntry> deep_copy(VirtualFileSystemEntry *parent = nullptr) const = 0;

  virtual void SaveToPath(const std::filesystem::path &path) const = 0;

 protected:
  VirtualFileSystemEntry *parent_;
};

class VirtualFileSystem : public FileSystem {
 public:
  VirtualFileSystem();

  VirtualFileSystem(const VirtualFileSystem &other);
  VirtualFileSystem(VirtualFileSystem &&other) noexcept;

  VirtualFileSystem &operator=(const VirtualFileSystem &other);
  VirtualFileSystem &operator=(VirtualFileSystem &&other) noexcept;

  // UTF-8 virtual paths: both separators are accepted on every host. Leading
  // separators refer to the VFS root; drive-qualified paths and root escapes fail.
  // Paths are lexical and case-sensitive. Directories are implicit.
  int WriteFile(const std::string &file_path, const std::vector<uint8_t> &data);
  int WriteFile(const std::string &file_path, const std::string &data);
  int WriteFile(const std::string &file_path, const char *data, size_t size);

  int ReadFile(const std::string &file_path, std::vector<uint8_t> &data) const override;

  void Print() const;

  static VirtualFileSystem LoadDirectory(const std::filesystem::path &path);

  void SaveToDirectory(const std::filesystem::path &path) const;

 private:
  std::map<std::string, std::vector<uint8_t>> files_;
};

}  // namespace grassland
