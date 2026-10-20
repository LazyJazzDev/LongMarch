#pragma once
#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IBuffer;
class IImage;

enum BindingType {
  BINDING_UNIFORM_BUFFER,
  BINDING_STORAGE_BUFFER,
  BINDING_RW_STORAGE_BUFFER,
  BINDING_TEXTURE2D,
  BINDING_RW_TEXTURE2D,
  BINDING_SAMPLER
};

enum ShaderVisibility { SHADER_VISIBILITY_VERTEX = 1, SHADER_VISIBILITY_FRAGMENT = 2, SHADER_VISIBILITY_ALL = 3 };

struct BindingDesc {
  uint32_t binding{};
  BindingType type{BINDING_UNIFORM_BUFFER};
  uint32_t count{1};
  uint32_t visibility{SHADER_VISIBILITY_ALL};
};

// Value descriptions are copied into Programs and Groups; later edits do not affect them.
struct BindingLayoutDesc {
  std::vector<BindingDesc> bindings;
};

struct PipelineLayoutDesc {
  std::vector<BindingLayoutDesc> groups;
};

enum SamplerFilter { SAMPLER_FILTER_NEAREST, SAMPLER_FILTER_LINEAR };

enum SamplerAddressMode { SAMPLER_ADDRESS_REPEAT, SAMPLER_ADDRESS_CLAMP_TO_EDGE, SAMPLER_ADDRESS_MIRRORED_REPEAT };

struct SamplerDesc {
  SamplerFilter filter{SAMPLER_FILTER_LINEAR};
  SamplerAddressMode address_u{SAMPLER_ADDRESS_REPEAT}, address_v{SAMPLER_ADDRESS_REPEAT};
};

// A borrowed buffer view. size == 0 selects the remaining bytes; stride defaults
// to 32-bit elements for structured buffers and is ignored for uniform buffers.
struct BufferView {
  IBuffer *buffer{};
  uint64_t offset{}, size{};
  uint32_t stride{4};
};

// Creation allocates the layout without populating resources. BindResource writes
// slots and retains resources. CmdBindGroup captures the current version; later
// writes never change recorded commands. CPU access to a group is externally synchronized.
class IBindingGroup : public IObject {
 public:
  virtual const BindingLayoutDesc &GetLayout() const = 0;
  bool BindResource(uint32_t binding, IBuffer *buffer);
  bool BindResource(uint32_t binding, const BufferView &buffer);
  bool BindResource(uint32_t binding, IImage *image);
  bool BindResource(uint32_t binding, const SamplerDesc &sampler);
  bool BindResource(uint32_t binding, const std::vector<IBuffer *> &buffers, uint32_t first_element = 0);
  virtual bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) = 0;
  virtual bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) = 0;
  virtual bool BindResource(uint32_t binding, const std::vector<SamplerDesc> &samplers, uint32_t first_element = 0) = 0;

 protected:
  ~IBindingGroup() override = default;
};
}  // namespace grassland::rhi
