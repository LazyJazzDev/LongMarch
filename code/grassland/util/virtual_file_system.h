#pragma once
#include <filesystem>
#include <map>

#include "grassland/util/util_util.h"

namespace grassland {

class VirtualFileSystemEntry {
 public:
  VirtualFileSystemEntry(VirtualFileSystemEntry *parent = nullptr) : parent_(parent) {
  }
  virtual ~VirtualFileSystemEntry() = default;

 protected:
  VirtualFileSystemEntry *parent_;
};

class VirtualFileSystem {
 public:
  VirtualFileSystem();
  int WriteFile(const std::string &file_name, const std::vector<uint8_t> &data);
  int WriteFile(const std::string &file_name, const std::string &data);
  int WriteFile(const std::string &file_name, const char *data, size_t size);

  int ReadFile(const std::string &file_name, std::vector<uint8_t> &data) const;

  void Print() const;

 private:
  VirtualFileSystemEntry *AccessFile(const std::string &path, bool create_if_not_exists = false) const;

  std::unique_ptr<VirtualFileSystemEntry> root_;
};
}  // namespace grassland
