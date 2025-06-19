#include "grassland/util/virtual_file_system.h"

namespace grassland {
class VirtualFileSystemFile;
class VirtualFileSystemDirectory : public VirtualFileSystemEntry {
 public:
  VirtualFileSystemDirectory(VirtualFileSystemEntry *parent) : VirtualFileSystemEntry(parent) {
  }
  VirtualFileSystemDirectory *enter(const std::string &name, bool create_if_not_exists = false) {
    if (name == ".") {
      return this;
    }
    if (name == "..") {
      return dynamic_cast<VirtualFileSystemDirectory *>(parent_);
    }
    if (!subentries_.count(name)) {
      if (!create_if_not_exists) {
        return nullptr;
      }
      subentries_[name] = std::make_unique<VirtualFileSystemDirectory>(this);
    }
    return dynamic_cast<VirtualFileSystemDirectory *>(subentries_.at(name).get());
  }

  VirtualFileSystemFile *open(const std::string &name, bool create_if_not_exists = false);

  std::map<std::string, std::unique_ptr<VirtualFileSystemEntry>> subentries_;
};

class VirtualFileSystemFile : public VirtualFileSystemEntry {
 public:
  VirtualFileSystemFile(VirtualFileSystemEntry *parent) : VirtualFileSystemEntry(parent) {
  }
  std::vector<uint8_t> data;
};

VirtualFileSystemFile *VirtualFileSystemDirectory::open(const std::string &name, bool create_if_not_exists) {
  if (name == ".") {
    return nullptr;
  }
  if (name == "..") {
    return nullptr;
  }
  if (!subentries_.count(name)) {
    if (!create_if_not_exists) {
      return nullptr;
    }
    subentries_[name] = std::make_unique<VirtualFileSystemFile>(this);
  }
  return dynamic_cast<VirtualFileSystemFile *>(subentries_.at(name).get());
}

VirtualFileSystem::VirtualFileSystem() {
  root_ = std::make_unique<VirtualFileSystemDirectory>(nullptr);
}

int VirtualFileSystem::WriteFile(const std::string &file_name, const std::vector<uint8_t> &data) {
  auto file = dynamic_cast<VirtualFileSystemFile *>(AccessFile(file_name, true));
  if (!file) {
    return -1;
  }
  file->data = data;
  return 0;
}

int VirtualFileSystem::WriteFile(const std::string &file_name, const std::string &data) {
  return WriteFile(file_name, data.data(), data.size());
}

int VirtualFileSystem::WriteFile(const std::string &file_name, const char *data, size_t size) {
  return WriteFile(file_name, std::vector<uint8_t>(data, data + size));
}

int VirtualFileSystem::ReadFile(const std::string &file_name, std::vector<uint8_t> &data) const {
  auto file = dynamic_cast<VirtualFileSystemFile *>(AccessFile(file_name, false));
  if (!file) {
    return -1;
  }
  data = file->data;
  return 0;
}

void VirtualFileSystem::Print() const {
  std::function<void(const std::string &, const VirtualFileSystemDirectory *)> print_directory;
  print_directory = [&](const std::string &cwd, const VirtualFileSystemDirectory *dir) -> void {
    for (auto &[name, entry] : dir->subentries_) {
      auto subdir = dynamic_cast<const VirtualFileSystemDirectory *>(entry.get());
      if (subdir) {
        print_directory(cwd + name + "/", subdir);
      }
      auto file = dynamic_cast<const VirtualFileSystemFile *>(entry.get());
      if (file) {
        printf("%s%s (%zu bytes)\n", cwd.c_str(), name.c_str(), file->data.size());
      }
    }
  };
  print_directory("", dynamic_cast<const VirtualFileSystemDirectory *>(root_.get()));
}

VirtualFileSystemEntry *VirtualFileSystem::AccessFile(const std::string &path, bool create_if_not_exists) const {
  VirtualFileSystemDirectory *cwd = dynamic_cast<VirtualFileSystemDirectory *>(root_.get());
  if (!cwd) {
    return nullptr;
  }
  std::filesystem::path p(path);
  if (p.begin() == p.end()) {
    return nullptr;
  }
  auto current = p.begin(), last = p.end();
  last--;
  while (current != last) {
    auto part = current->string();
    if (part == "/" || part == "\\") {
      cwd = dynamic_cast<VirtualFileSystemDirectory *>(root_.get());
    } else {
      cwd = cwd->enter(part, create_if_not_exists);
    }
    if (!cwd) {
      return nullptr;
    }
    current++;
  }
  return cwd->open(last->string(), create_if_not_exists);
}

}  // namespace grassland
