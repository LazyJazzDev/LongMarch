#pragma once
#include "grassland/graphics/shader.h"

namespace grassland::graphics {

// A source snapshot. Includes and compiler arguments are owned and compiled per program.
class ShaderCode {
 public:
  ShaderCode(const std::string &source,
             const std::string &entry_point,
             const std::string &target,
             const std::vector<std::string> &args = {});
  ShaderCode(const VirtualFileSystem &vfs,
             const std::string &source_file,
             const std::string &entry_point,
             const std::string &target,
             const std::vector<std::string> &args = {});

  const std::string &EntryPoint() const {
    return entry_point_;
  }

  const std::string &Target() const {
    return target_;
  }

  static std::string ResourceBindingDefinitions(BackendAPI api,
                                                const std::vector<std::pair<ResourceType, int>> &bindings);

 private:
  friend class ProgramShaderBindings;
  std::unique_ptr<Shader> Compile(Core *core, const std::vector<std::pair<ResourceType, int>> &bindings) const;
  VirtualFileSystem vfs_;
  std::string source_file_, entry_point_, target_;
  std::vector<std::string> args_;
};

}  // namespace grassland::graphics
