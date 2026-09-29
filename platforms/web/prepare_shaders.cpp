// Compiles the browser apps' Slang requests to WGSL into a shader cache.
//
// The web build compiles no shaders: WebGPUCore::CreateShader reads the cache
// entry keyed by the same virtual file system, entry point, target and "-target
// wgsl" arguments. Each request below mirrors one call in the browser apps.
#include <iostream>

#include "grassland/graphics/shader.h"
#include "grassland/graphics/shader_cache.h"

namespace {
#include "demo_shaders.inl"

const std::vector<std::string> kWGSL{"-target", "wgsl"};

void Compile(const grassland::VirtualFileSystem &vfs,
             const std::string &file,
             const std::string &entry,
             const std::string &target) {
  if (grassland::graphics::CompileShader(vfs, file, entry, target, kWGSL).data.empty())
    throw std::runtime_error("Cannot compile " + file + " " + entry + " to WGSL");
  std::cout << "WGSL " << file << ' ' << entry << '\n';
}
}  // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: web_shader_prepare <shader-cache-directory>\n";
    return 2;
  }
  try {
    grassland::graphics::ConfigureShaderCache({argv[1], false, false});
    const auto demos = GetShaderVirtualFileSystem();
    // Games pass one shader's text, which Core::CreateShader stores as shader.slang.
    for (const std::string game : {"gol", "2048"})
      for (const std::string shader : {"super", "resolve"}) {
        std::vector<uint8_t> text;
        demos.ReadFile(game + "/shaders/" + shader + ".slang", text);
        grassland::VirtualFileSystem vfs;
        vfs.WriteFile("shader.slang", std::string(text.begin(), text.end()));
        Compile(vfs, "shader.slang", "VSMain", "vs_6_0");
        Compile(vfs, "shader.slang", "PSMain", "ps_6_0");
      }
    // DemoSession::InitializeNBody.
    Compile(demos, "nbody_cs/shaders/nbody.slang", "CSMain", "cs_6_0");
    Compile(demos, "nbody_cs/shaders/particle.slang", "VSMain", "vs_6_0");
    Compile(demos, "nbody_cs/shaders/particle.slang", "PSMain", "ps_6_0");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
