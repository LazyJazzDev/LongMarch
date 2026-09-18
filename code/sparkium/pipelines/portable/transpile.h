#pragma once

#include <map>
#include <string>
#include <vector>

#include "grassland/util/virtual_file_system.h"

namespace sparkium::portable {

// Rewrites the HLSL subset used by code/sparkium/shaders into C++ that
// compiles against hlsl_compat.h. Both the CPU and the CUDA backend consume
// the generated sources, so shading behavior cannot diverge between them.
class Transpiler {
 public:
  explicit Transpiler(const grassland::VirtualFileSystem &vfs);

  // Expand includes recursively (rooted at the shader VFS) and apply the
  // HLSL->C++ rewrites. `defines` are set before reading the file.
  std::string Process(const std::string &path, const std::map<std::string, std::string> &defines);

  // Applies the line-level HLSL->C++ rewrites to a code snippet whose
  // includes were already stripped (material sampler implementations).
  std::string TransformSnippet(const std::string &source);

 private:
  std::string ProcessFile(const std::string &path, std::map<std::string, std::string> &defines);
  std::string Transform(const std::string &source);

  const grassland::VirtualFileSystem &vfs_;
  std::map<std::string, std::string> defines_;
  // Object-like defines that alias vector types (`#define Spectrum float3`),
  // collected while expanding files and expanded by Transform().
  std::map<std::string, std::string> type_aliases_;
  // Function-like macros (#define saturatef(x) ...) with joined bodies.
  std::map<std::string, std::string> function_macros;
};

}  // namespace sparkium::portable
