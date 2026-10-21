#pragma once
#include "grassland/rhi/rhi_object.h"
#include "grassland/rhi/rhi_sampler.h"

namespace grassland::rhi {
class IBuffer;
class IImage;
class ITopLevelAccelerationStructure;

typedef enum {
  BINDING_UNIFORM_BUFFER,
  BINDING_STRUCTURED_BUFFER,
  BINDING_RW_STRUCTURED_BUFFER,
  BINDING_BYTE_ADDRESS_BUFFER,
  BINDING_RW_BYTE_ADDRESS_BUFFER,
  BINDING_TEXTURE2D,
  BINDING_RW_TEXTURE2D,
  BINDING_SAMPLER,
  BINDING_ACCELERATION_STRUCTURE
} BindingType;

typedef enum {
  SHADER_VISIBILITY_VERTEX = 1,
  SHADER_VISIBILITY_FRAGMENT = 2,
  SHADER_VISIBILITY_COMPUTE = 4,
  SHADER_VISIBILITY_RAY_GENERATION = 8,
  SHADER_VISIBILITY_MISS = 16,
  SHADER_VISIBILITY_CLOSEST_HIT = 32,
  SHADER_VISIBILITY_ANY_HIT = 64,
  SHADER_VISIBILITY_INTERSECTION = 128,
  SHADER_VISIBILITY_CALLABLE = 256,
  SHADER_VISIBILITY_ALL = 511
} ShaderVisibility;

struct BindingDesc {
  uint32_t binding{};
  BindingType type{BINDING_UNIFORM_BUFFER};
  uint32_t count{1};
  uint32_t visibility{SHADER_VISIBILITY_ALL};
  // Element stride in bytes; only meaningful for [RW_]STRUCTURED_BUFFER.
  uint32_t stride{};
};

// Value descriptions are copied into Programs and Groups; later edits do not affect them.
struct BindingLayoutDesc {
  std::vector<BindingDesc> bindings;
};

struct PipelineLayoutDesc {
  std::vector<BindingLayoutDesc> groups;
};

// A borrowed byte range. size == 0 selects the remaining bytes.
// Structured ranges must be multiples of BindingDesc::stride.
struct BufferView {
  IBuffer *buffer{};
  uint64_t offset{}, size{};
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
  bool BindResource(uint32_t binding, ISampler *sampler);
  bool BindResource(uint32_t binding, ITopLevelAccelerationStructure *structure);
  virtual bool BindResource(uint32_t binding,
                            const std::vector<ITopLevelAccelerationStructure *> &structures,
                            uint32_t first_element = 0) = 0;
  bool BindResource(uint32_t binding, const std::vector<IBuffer *> &buffers, uint32_t first_element = 0);
  virtual bool BindResource(uint32_t binding, const std::vector<BufferView> &buffers, uint32_t first_element = 0) = 0;
  virtual bool BindResource(uint32_t binding, const std::vector<IImage *> &images, uint32_t first_element = 0) = 0;
  virtual bool BindResource(uint32_t binding, const std::vector<ISampler *> &samplers, uint32_t first_element = 0) = 0;

 protected:
  ~IBindingGroup() override = default;
};
}  // namespace grassland::rhi
