#pragma once
#include "grassland/rhi/rhi_object.h"
#include "grassland/util/file_system.h"

namespace grassland::rhi {
struct ShaderDesc {
  // Nonempty source selects in-memory input; filename then supplies its virtual path.
  std::string source;
  std::string entry_point;
  ShaderStage stage{SHADER_STAGE_VERTEX};
  std::string filename;
  // Borrowed only during CreateShader. GetDesc retains this pointer as metadata;
  // callers must establish its lifetime before dereferencing it. nullptr uses disk.
  FileSystem *file_system{};
  std::vector<std::string> include_paths;
  std::vector<std::pair<std::string, std::string>> defines;
};

// Includes and preprocessing are resolved at creation; layout macros and backend
// compilation are resolved when creating a Program. FileSystem is not retained.
class IShader : public IObject {
 public:
  virtual const ShaderDesc &GetDesc() const = 0;

 protected:
  ~IShader() override;
};
}  // namespace grassland::rhi
