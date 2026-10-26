#include "grassland/rhi/rhi_device.h"

#include <fstream>

#include "grassland/rhi/rhi_shader.h"
#include "grassland/util/virtual_file_system.h"

namespace grassland::rhi {
namespace {
// The entry lives in a temporary VFS; included files use the system filesystem.
class ShaderSourceFileSystem final : public VirtualFileSystem {
 public:
  int ReadFile(const std::string &file_path, std::vector<uint8_t> &data) const override {
    if (VirtualFileSystem::ReadFile(file_path, data) == 0)
      return 0;
    std::ifstream file(std::filesystem::u8path(file_path), std::ios::binary | std::ios::ate);
    if (!file)
      return -1;
    const auto size = file.tellg();
    if (size < 0)
      return -1;
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size && !file.read(reinterpret_cast<char *>(bytes.data()), size))
      return -1;
    data = std::move(bytes);
    return 0;
  }
};
}  // namespace

IDevice::~IDevice() = default;

IShader *IDevice::CreateShader(const std::string &filename,
                               const std::string &entry_point,
                               ShaderStage stage,
                               const FileSystem *fs,
                               const std::vector<std::string> &include_paths,
                               const std::vector<std::pair<std::string, std::string>> &defines) {
  ShaderDesc desc;
  desc.filename = filename;
  desc.entry_point = entry_point;
  desc.stage = stage;
  desc.file_system = fs;
  desc.include_paths = include_paths;
  desc.defines = defines;
  return CreateShader(desc);
}

IShader *IDevice::CreateShaderFromSource(const std::string &source,
                                         const std::string &entry_point,
                                         ShaderStage stage,
                                         const std::vector<std::string> &include_paths,
                                         const std::vector<std::pair<std::string, std::string>> &defines) {
  ShaderSourceFileSystem files;
  if (files.WriteFile("main.slang", source))
    return nullptr;
  ShaderDesc desc;
  desc.entry_point = entry_point;
  desc.stage = stage;
  desc.file_system = &files;
  desc.include_paths = include_paths;
  desc.defines = defines;
  return CreateShader(desc);
}
}  // namespace grassland::rhi
