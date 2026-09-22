#include "grassland/graphics/shader_code.h"

#include <sstream>
#include <stdexcept>

#include "grassland/graphics/core.h"

namespace grassland::graphics {

ShaderCode::ShaderCode(const std::string &source,
                       const std::string &entry_point,
                       const std::string &target,
                       const std::vector<std::string> &args)
    : source_file_("shader.hlsl"),
      entry_point_(entry_point),
      target_(target),
      args_(args) {
  vfs_.WriteFile(source_file_, source);
}

ShaderCode::ShaderCode(const VirtualFileSystem &vfs,
                       const std::string &source_file,
                       const std::string &entry_point,
                       const std::string &target,
                       const std::vector<std::string> &args)
    : vfs_(vfs),
      source_file_(source_file),
      entry_point_(entry_point),
      target_(target),
      args_(args) {
}

std::string ShaderCode::ResourceBindingDefinitions(BackendAPI api,
                                                   const std::vector<std::pair<ResourceType, int>> &bindings) {
  if (api != BACKEND_API_METAL && api != BACKEND_API_VULKAN && api != BACKEND_API_D3D12)
    throw std::invalid_argument("unsupported shader binding backend");
  std::ostringstream source;
  source << "#define LM_RESOURCE_BINDING_SELECT(slot, type, name) LM_RESOURCE_BINDING_ ## slot(type, name)\n"
            "#define RESOURCE_BINDING(slot, type, name) LM_RESOURCE_BINDING_SELECT(slot, type, name)\n";
  uint64_t offset = 0;
  for (size_t slot = 0; slot < bindings.size(); ++slot) {
    auto [type, count] = bindings[slot];
    if (type < RESOURCE_TYPE_UNIFORM_BUFFER || type > RESOURCE_TYPE_WRITABLE_STORAGE_BUFFER || count <= 0)
      throw std::invalid_argument("invalid resource binding type or count");
    const char registers[] = {'b', 't', 't', 'u', 's', 't', 'u'};
    source << "#define LM_RESOURCE_BINDING_" << slot << "(type, name) ";
    if (api == BACKEND_API_METAL)
      source << "[[vk::binding(" << slot << ", 0)]] ";
    source << "type name";
    if (count > 1)
      source << '[' << count << ']';
    source << " : register(" << registers[type] << (api == BACKEND_API_METAL ? offset : 0) << ", space"
           << (api == BACKEND_API_METAL ? 0 : slot) << ")\n";
    offset += count;
    if (offset > UINT32_MAX)
      throw std::overflow_error("resource binding index overflow");
  }
  return source.str();
}

std::unique_ptr<Shader> ShaderCode::Compile(Core *core,
                                            const std::vector<std::pair<ResourceType, int>> &bindings) const {
  if (!core)
    throw std::invalid_argument("missing shader compilation core");
  if (entry_point_.empty() || target_.size() < 4)
    throw std::invalid_argument("missing shader entry point or target");
  std::vector<uint8_t> bytes;
  if (vfs_.ReadFile(source_file_, bytes))
    throw std::runtime_error("shader source not found: " + source_file_);
  auto vfs = vfs_;
  auto source = ResourceBindingDefinitions(core->API(), bindings);
  source += "#line 1\n";
  source.append(bytes.begin(), bytes.end());
  vfs.WriteFile(source_file_, source);
  std::unique_ptr<Shader> shader;
  if (core->CreateShader(vfs, source_file_, entry_point_, target_, args_, &shader) || !shader)
    throw std::runtime_error("failed to finalize shader: " + source_file_ + " / " + entry_point_);
  return shader;
}

}  // namespace grassland::graphics
