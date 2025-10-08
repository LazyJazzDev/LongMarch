#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Shader {
 public:
  virtual ~Shader() = default;
  virtual std::string EntryPoint() const = 0;

#if defined(LONGMARCH_PYTHON_ENABLED)
  static void PybindClassRegistration(py::classh<Shader> &c);
#endif
};

CompiledShaderBlob CompileShader(const std::string &source_code,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args = {});

CompiledShaderBlob CompileShader(const VirtualFileSystem &vfs,
                                 const std::string &source_file,
                                 const std::string &entry_point,
                                 const std::string &target,
                                 const std::vector<std::string> &args = {});

}  // namespace grassland::graphics
