#include <fstream>
#include <iostream>
#include <thread>

#include "grassland/rhi/backend/backend_shader_cache.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/util/virtual_file_system.h"

using namespace grassland;
using namespace grassland::rhi;
using namespace grassland::rhi::backend;

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;
  bool native = false;
  std::thread thread([&] { native = GetShaderCacheSettings().target_platform != ShaderTargetPlatform::Native; });
  thread.join();
  if (!native)
    return 2;
  const std::filesystem::path directory(argv[1]);
  VirtualFileSystem fs;
  fs.WriteFile("main.slang", "#if RHI_PLATFORM_OHOS\nfloat platform = 5;\n#else\nfloat platform = 4;\n#endif\n");
  ShaderDesc desc;
  desc.file_system = &fs;
  desc.entry_point = "main";
  std::string ohos, ios, replay;
  if (!ConfigureShaderCache({directory, false, false, ShaderTargetPlatform::OHOS}) ||
      !backend::PreprocessShader(desc, BACKEND_API_VULKAN, ohos) || ohos.find("= 5") == std::string::npos)
    return 3;
  if (!ConfigureShaderCache({directory, true, false, ShaderTargetPlatform::IOS}) ||
      backend::PreprocessShader(desc, BACKEND_API_VULKAN, replay))
    return 4;
  if (!ConfigureShaderCache({directory, false, false, ShaderTargetPlatform::IOS}) || !GetShaderCacheSettings().ios ||
      !backend::PreprocessShader(desc, BACKEND_API_VULKAN, ios) || ios == ohos)
    return 5;
  if (ConfigureShaderCache({directory, true, true, ShaderTargetPlatform::OHOS}) ||
      GetShaderCacheSettings().target_platform != ShaderTargetPlatform::IOS)
    return 6;
  if (!ConfigureShaderCache({directory, true, false, ShaderTargetPlatform::OHOS}) ||
      !backend::PreprocessShader(desc, BACKEND_API_VULKAN, replay) || replay != ohos)
    return 7;
  unsigned count = 0;
  for (auto &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.path().filename().string().find("pre-") != 0)
      continue;
    ++count;
    std::ofstream corrupt(entry.path(), std::ios::binary | std::ios::trunc);
    corrupt << "corrupt";
  }
  if (count != 2 || backend::PreprocessShader(desc, BACKEND_API_VULKAN, replay))
    return 8;
  std::cout << "PASS default platform, target isolation, macros, replay, conflict and corruption rejection\n";
}
