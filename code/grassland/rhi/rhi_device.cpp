#include "grassland/rhi/rhi_device.h"

#include "grassland/rhi/rhi_shader.h"
#include "grassland/util/virtual_file_system.h"

namespace grassland::rhi {
IDevice::~IDevice() = default;

IShader *IDevice::CreateShader(const std::string &filename,
                               const std::string &entry_point,
                               ShaderStage stage,
                               FileSystem *fs) {
  ShaderDesc desc;
  desc.filename = filename;
  desc.entry_point = entry_point;
  desc.stage = stage;
  desc.file_system = fs;
  return CreateShader(desc);
}

IShader *IDevice::CreateShaderFromSource(const std::string &source, const std::string &entry_point, ShaderStage stage) {
  VirtualFileSystem files;
  if (files.WriteFile("main.slang", source))
    return nullptr;
  ShaderDesc desc;
  desc.entry_point = entry_point;
  desc.stage = stage;
  desc.file_system = &files;
  return CreateShader(desc);
}
}  // namespace grassland::rhi
