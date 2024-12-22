#include "grassland/util/file_probe.h"

#include "filesystem"

namespace grassland {

FileProbe::FileProbe() {
}

FileProbe::~FileProbe() {
}

void FileProbe::AddSearchPath(const std::string &path) {
  search_paths_.push_back(path);
}

std::string FileProbe::FindFile(const std::string &filename) const {
  for (const auto &path : search_paths_) {
    std::string full_path = path + filename;
    if (std::filesystem::exists(full_path) &&
        std::filesystem::is_regular_file(full_path)) {
      return full_path;
    }
  }
  return "";
}

FileProbe &FileProbe::GetInstance() {
  static FileProbe instance;
  static bool initialized = false;
  if (!initialized) {
    instance.AddSearchPath("");
    instance.AddSearchPath(LONGMARCH_ASSETS_DIR);
    instance.AddSearchPath("./");
    instance.AddSearchPath("../");
    instance.AddSearchPath("../../");
    initialized = true;
  }
  return instance;
}

std::ostream &operator<<(std::ostream &os, const FileProbe &probe) {
  os << "Search paths:\n-------------" << std::endl;
  for (const auto &path : probe.search_paths_) {
    os << path << std::endl;
  }
  return os;
}

std::string FindAssetsFile(const std::string &filename) {
  return FileProbe::GetInstance().FindFile(filename);
}

}  // namespace grassland
