#include "grassland/util/virtual_file_system.h"

#include <fstream>
#include <stack>

namespace CD {
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

  std::unique_ptr<VirtualFileSystemEntry> deep_copy(VirtualFileSystemEntry *parent) const override {
    std::unique_ptr<VirtualFileSystemDirectory> copy = std::make_unique<VirtualFileSystemDirectory>(parent);
    for (auto &[name, entry] : subentries_) {
      copy->subentries_[name] = entry->deep_copy(copy.get());
    }
    return copy;
  }

  void SaveToPath(const std::filesystem::path &path) const override {
    // make sure path directory exists
    std::filesystem::create_directories(path);
    for (const auto &[name, entry] : subentries_) {
      entry->SaveToPath(path / name);
    }
  }
  std::map<std::string, std::unique_ptr<VirtualFileSystemEntry>> subentries_;
};

class VirtualFileSystemFile : public VirtualFileSystemEntry {
 public:
  VirtualFileSystemFile(VirtualFileSystemEntry *parent) : VirtualFileSystemEntry(parent) {
  }

  std::unique_ptr<VirtualFileSystemEntry> deep_copy(VirtualFileSystemEntry *parent) const override {
    std::unique_ptr<VirtualFileSystemFile> copy = std::make_unique<VirtualFileSystemFile>(parent);
    copy->data = data;  // Copy the file data
    return copy;
  }

  void SaveToPath(const std::filesystem::path &path) const override {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size());
    file.close();
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

VirtualFileSystem::VirtualFileSystem(const VirtualFileSystem &other) {
  root_ = other.root_->deep_copy(nullptr);
}

VirtualFileSystem::VirtualFileSystem(VirtualFileSystem &&other) noexcept {
  root_ = std::move(other.root_);
}

VirtualFileSystem &VirtualFileSystem::operator=(const VirtualFileSystem &other) {
  root_ = other.root_->deep_copy(nullptr);
  return *this;
}

VirtualFileSystem &VirtualFileSystem::operator=(VirtualFileSystem &&other) noexcept {
  root_ = std::move(other.root_);
  return *this;
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

VirtualFileSystem VirtualFileSystem::LoadDirectory(const std::filesystem::path &path) {
  VirtualFileSystem vfs;

  for (auto &entry : std::filesystem::recursive_directory_iterator(path)) {
    auto relative_path = std::filesystem::relative(entry.path(), path);
    if (entry.is_regular_file()) {
      std::vector<uint8_t> data;
      std::ifstream file(entry.path(), std::ios::binary);
      if (file) {
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        data.resize(size);
        file.read(reinterpret_cast<char *>(data.data()), size);
        vfs.WriteFile(relative_path.string(), data);
      }
    }
  }

  return vfs;
}

void VirtualFileSystem::SaveToDirectory(const std::filesystem::path &path) const {
  root_->SaveToPath(path);
}

}  // namespace CD
