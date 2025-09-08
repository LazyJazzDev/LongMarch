#pragma once
#include <filesystem>
#include <map>

#include "grassland/util/util_util.h"

namespace CD {

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

class VirtualFileSystem {
 public:
  VirtualFileSystem();

  VirtualFileSystem(const VirtualFileSystem &other);
  VirtualFileSystem(VirtualFileSystem &&other) noexcept;

  VirtualFileSystem &operator=(const VirtualFileSystem &other);
  VirtualFileSystem &operator=(VirtualFileSystem &&other) noexcept;

  int WriteFile(const std::string &file_name, const std::vector<uint8_t> &data);
  int WriteFile(const std::string &file_name, const std::string &data);
  int WriteFile(const std::string &file_name, const char *data, size_t size);

  int ReadFile(const std::string &file_name, std::vector<uint8_t> &data) const;

  void Print() const;

  static VirtualFileSystem LoadDirectory(const std::filesystem::path &path);

  void SaveToDirectory(const std::filesystem::path &path) const;

 private:
  VirtualFileSystemEntry *AccessFile(const std::string &path, bool create_if_not_exists = false) const;

  std::unique_ptr<VirtualFileSystemEntry> root_;
};

}  // namespace CD
