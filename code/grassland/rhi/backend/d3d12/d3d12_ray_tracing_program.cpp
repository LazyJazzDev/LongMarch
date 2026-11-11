#include "grassland/rhi/backend/d3d12/d3d12_ray_tracing_program.h"

#include <cstring>

#include "grassland/rhi/backend/backend_binding_group.h"
#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"
#include "grassland/rhi/backend/d3d12/d3d12_ray_tracing_program_impl.h"

namespace grassland::rhi::backend::d3d12 {
D3D12RayTracingProgramImpl::D3D12RayTracingProgramImpl(ID3D12Device *device, const RayTracingProgramDesc &desc) {
  InitializeRayTracingProgram(device, desc);
}

ID3D12StateObject *D3D12RayTracingProgramImpl::NativePipeline() const {
  return pipeline_.Get();
}

const IBackendShaderBindingTable *D3D12RayTracingProgramImpl::DefaultShaderBindingTable() const {
  return table_.Get();
}

IShaderBindingTable *D3D12RayTracingProgramImpl::CreateShaderBindingTable(const ShaderBindingTableDesc &desc) {
  if (!ValidateShaderBindingTable(desc))
    return nullptr;
  return BuildShaderBindingTable(desc);
}

ID3D12RayTracingProgram *D3D12RayTracingProgramImpl::Create(ID3D12Device *owner_device,
                                                            const RayTracingProgramDesc &desc) {
  auto object = std::unique_ptr<D3D12RayTracingProgramImpl>(new D3D12RayTracingProgramImpl(owner_device, desc));
  if (!object->ValidateLayout() || object->desc_.max_recursion_depth > 31 || object->desc_.max_attribute_size > 32)
    return nullptr;
  auto native = owner_device->NativeDevice();
  ComPtr<ID3D12Device5> device;
  if (FAILED(native->QueryInterface(IID_PPV_ARGS(&device))))
    return nullptr;
  if (!object->CreateLayout(native, object->desc_.pipeline_layout, D3D12_ROOT_SIGNATURE_FLAG_NONE))
    return nullptr;
  std::vector<IShader *> shaders{object->desc_.ray_generation};
  shaders.insert(shaders.end(), object->desc_.miss_shaders.begin(), object->desc_.miss_shaders.end());
  std::vector<std::array<uint32_t, 3>> indices;
  for (auto group : object->desc_.hit_groups) {
    std::array<uint32_t, 3> item{UINT32_MAX, UINT32_MAX, UINT32_MAX};
    IShader *members[] = {group.closest_hit, group.any_hit, group.intersection};
    for (int i = 0; i < 3; ++i)
      if (members[i]) {
        item[i] = uint32_t(shaders.size());
        shaders.push_back(members[i]);
      }
    indices.push_back(item);
  }
  auto callable_start = shaders.size();
  shaders.insert(shaders.end(), object->desc_.callable_shaders.begin(), object->desc_.callable_shaders.end());
  std::vector<std::vector<uint8_t>> code(shaders.size());
  std::vector<std::wstring> names(shaders.size()), entries(shaders.size()), hit_names(indices.size());
  std::vector<D3D12_EXPORT_DESC> exports(shaders.size());
  std::vector<D3D12_DXIL_LIBRARY_DESC> libraries(shaders.size());
  std::vector<D3D12_HIT_GROUP_DESC> hits(indices.size());
  std::vector<D3D12_STATE_SUBOBJECT> subobjects;
  auto header = ShaderMacroHeader(object->desc_.pipeline_layout, BACKEND_API_D3D12);
  for (size_t i = 0; i < shaders.size(); ++i) {
    if (!CompileSlang(shaders[i], header, BACKEND_API_D3D12, code[i], &object->desc_.pipeline_layout))
      return nullptr;
    names[i] = L"rhi_shader_" + std::to_wstring(i);
    const auto &entry = shaders[i]->GetEntry();
    entries[i].assign(entry.begin(), entry.end());
    exports[i].Name = names[i].c_str();
    exports[i].ExportToRename = entries[i].c_str();
    libraries[i].DXILLibrary = {code[i].data(), code[i].size()};
    libraries[i].NumExports = 1;
    libraries[i].pExports = &exports[i];
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libraries[i]});
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    hit_names[i] = L"rhi_hit_" + std::to_wstring(i);
    auto &hit = hits[i];
    hit.HitGroupExport = hit_names[i].c_str();
    hit.Type = indices[i][2] == UINT32_MAX ? D3D12_HIT_GROUP_TYPE_TRIANGLES : D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;
    if (indices[i][0] != UINT32_MAX)
      hit.ClosestHitShaderImport = names[indices[i][0]].c_str();
    if (indices[i][1] != UINT32_MAX)
      hit.AnyHitShaderImport = names[indices[i][1]].c_str();
    if (indices[i][2] != UINT32_MAX)
      hit.IntersectionShaderImport = names[indices[i][2]].c_str();
    subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hit});
  }
  D3D12_RAYTRACING_SHADER_CONFIG shader_config{object->desc_.max_payload_size, object->desc_.max_attribute_size};
  D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config{object->desc_.max_recursion_depth};
  auto root = object->NativeRootSignature();
  subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader_config});
  subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipeline_config});
  subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &root});
  D3D12_STATE_OBJECT_DESC create{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, UINT(subobjects.size()),
                                 subobjects.data()};
  ComPtr<ID3D12StateObject> pipeline;
  if (!Check(device->CreateStateObject(&create, IID_PPV_ARGS(&pipeline)), "Create ray tracing pipeline"))
    return nullptr;
  ComPtr<ID3D12StateObjectProperties> properties;
  if (FAILED(pipeline.As(&properties)))
    return nullptr;
  std::vector<std::vector<std::wstring>> tables{{names[0]}, {}, hit_names, {}};
  for (size_t i = 0; i < object->desc_.miss_shaders.size(); ++i)
    tables[1].push_back(names[1 + i]);
  for (size_t i = callable_start; i < names.size(); ++i)
    tables[3].push_back(names[i]);
  for (size_t t = 0; t < 4; ++t) {
    object->identifiers_[t].resize(tables[t].size() * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    for (size_t i = 0; i < tables[t].size(); ++i) {
      auto identifier = properties->GetShaderIdentifier(tables[t][i].c_str());
      if (!identifier)
        return nullptr;
      std::memcpy(object->identifiers_[t].data() + i * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES, identifier,
                  D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    }
  }
  object->table_ = ObjectPtr<IBackendShaderBindingTable>::Adopt(
      object->BuildShaderBindingTable(object->DefaultShaderBindingTableDesc()));
  if (!object->table_)
    return nullptr;
  object->pipeline_ = std::move(pipeline);
  return object.release();
}

IBackendShaderBindingTable *D3D12RayTracingProgramImpl::BuildShaderBindingTable(const ShaderBindingTableDesc &desc) {
  constexpr uint64_t stride = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
  constexpr uint64_t alignment = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
  std::vector<uint8_t> data;
  std::array<IBackendShaderBindingTable::Region, 4> regions{};
  const std::vector<uint32_t> raygen{0};
  const std::vector<uint32_t> *indices[] = {&raygen, &desc.miss_shader_indices, &desc.hit_group_indices,
                                            &desc.callable_shader_indices};
  for (size_t t = 0; t < 4; ++t) {
    if (indices[t]->empty())
      continue;
    data.resize((data.size() + alignment - 1) & ~(alignment - 1));
    auto offset = data.size();
    regions[t] = {offset, indices[t]->size() * stride, stride};
    data.resize(offset + regions[t].size);
    for (size_t i = 0; i < indices[t]->size(); ++i)
      std::memcpy(data.data() + offset + i * stride, identifiers_[t].data() + (*indices[t])[i] * stride, stride);
  }
  auto owner = rhi::QueryInterfacePtr<ID3D12Device>(device_).Get();
  auto buffer =
      ObjectPtr<ID3D12AllocatedBuffer>::Adopt(ID3D12AllocatedBuffer::Create(owner, {uint64_t(data.size())}, true));
  if (!buffer || !buffer->Write(data.data(), data.size()))
    return nullptr;
  auto address = buffer->NativeBuffer()->GetGPUVirtualAddress();
  for (auto &region : regions)
    if (region.size)
      region.address += address;
  return IBackendShaderBindingTable::Create(this, desc, std::move(buffer), regions);
}

ID3D12RayTracingProgram *ID3D12RayTracingProgram::Create(ID3D12Device *device, const RayTracingProgramDesc &desc) {
  return D3D12RayTracingProgramImpl::Create(device, desc);
}

void D3D12RayTracingProgramImpl::InitializeRayTracingProgram(IBackendDevice *device,
                                                             const RayTracingProgramDesc &desc) {
  device_ = device;
  desc_ = desc;

  auto retain = [&](IShader *shader) {
    if (shader)
      shaders_.push_back(ObjectPtr<IShader>::Retain(shader));
  };
  retain(desc.ray_generation);
  for (auto shader : desc.miss_shaders)
    retain(shader);
  for (const auto &group : desc.hit_groups) {
    retain(group.closest_hit);
    retain(group.any_hit);
    retain(group.intersection);
  }
  for (auto shader : desc.callable_shaders)
    retain(shader);
}

const RayTracingProgramDesc &D3D12RayTracingProgramImpl::GetDesc() const {
  return desc_;
}

IBackendDevice *D3D12RayTracingProgramImpl::Owner() const {
  return device_;
}

bool D3D12RayTracingProgramImpl::ValidateLayout() const {
  auto stage = [](IShader *shader, ShaderStage expected) { return shader && shader->GetStage() == expected; };
  if (!stage(desc_.ray_generation, SHADER_STAGE_RAY_GENERATION) || !desc_.max_recursion_depth ||
      desc_.pipeline_layout.groups.size() > 8)
    return false;
  for (auto shader : desc_.miss_shaders)
    if (!stage(shader, SHADER_STAGE_MISS))
      return false;
  for (const auto &group : desc_.hit_groups) {
    if (!group.closest_hit && !group.any_hit && !group.intersection)
      return false;
    if (group.closest_hit && !stage(group.closest_hit, SHADER_STAGE_CLOSEST_HIT))
      return false;
    if (group.any_hit && !stage(group.any_hit, SHADER_STAGE_ANY_HIT))
      return false;
    if (group.intersection && !stage(group.intersection, SHADER_STAGE_INTERSECTION))
      return false;
  }
  for (auto shader : desc_.callable_shaders)
    if (!stage(shader, SHADER_STAGE_CALLABLE))
      return false;
  for (const auto &group : desc_.pipeline_layout.groups)
    if (!ValidateBindingLayout(group))
      return false;
  return true;
}

bool D3D12RayTracingProgramImpl::ValidateShaderBindingTable(const ShaderBindingTableDesc &desc) const {
  for (auto index : desc.miss_shader_indices)
    if (index >= desc_.miss_shaders.size())
      return false;
  for (auto index : desc.hit_group_indices)
    if (index >= desc_.hit_groups.size())
      return false;
  for (auto index : desc.callable_shader_indices)
    if (index >= desc_.callable_shaders.size())
      return false;
  return true;
}

ShaderBindingTableDesc D3D12RayTracingProgramImpl::DefaultShaderBindingTableDesc() const {
  ShaderBindingTableDesc result;
  auto fill = [](auto &indices, size_t count) {
    for (size_t i = 0; i < count; ++i)
      indices.push_back(uint32_t(i));
  };
  fill(result.miss_shader_indices, desc_.miss_shaders.size());
  fill(result.hit_group_indices, desc_.hit_groups.size());
  fill(result.callable_shader_indices, desc_.callable_shaders.size());
  return result;
}

ID3D12RootSignature *D3D12RayTracingProgramImpl::NativeRootSignature() const {
  return root_signature_.Get();
}

uint32_t D3D12RayTracingProgramImpl::RootParameter(uint32_t group, bool sampler) const {
  return root_parameters_[group][sampler];
}

bool D3D12RayTracingProgramImpl::CreateLayout(::ID3D12Device *device,
                                              const PipelineLayoutDesc &layout,
                                              D3D12_ROOT_SIGNATURE_FLAGS flags) {
  D3D12_ROOT_SIGNATURE_DESC root{};
  root.Flags = flags;
  const auto mappings = MapBindings(layout, BACKEND_API_D3D12);
  std::vector<std::vector<D3D12_DESCRIPTOR_RANGE>> ranges(layout.groups.size() * 2);
  for (auto m : mappings) {
    D3D12_DESCRIPTOR_RANGE range{};
    range.RangeType = m.desc.type == BINDING_SAMPLER          ? D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER
                      : m.desc.type == BINDING_UNIFORM_BUFFER ? D3D12_DESCRIPTOR_RANGE_TYPE_CBV
                      : (m.desc.type == BINDING_RW_STRUCTURED_BUFFER || m.desc.type == BINDING_RW_BYTE_ADDRESS_BUFFER ||
                         m.desc.type == BINDING_RW_TEXTURE2D)
                          ? D3D12_DESCRIPTOR_RANGE_TYPE_UAV
                          : D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    range.NumDescriptors = m.desc.count;
    range.BaseShaderRegister = m.shader_register;
    range.RegisterSpace = m.group;
    range.OffsetInDescriptorsFromTableStart = m.table_offset;
    ranges[m.group * 2 + (m.desc.type == BINDING_SAMPLER)].push_back(range);
  }
  std::vector<D3D12_ROOT_PARAMETER> parameters;
  root_parameters_.assign(layout.groups.size(), {UINT32_MAX, UINT32_MAX});
  for (size_t i = 0; i < ranges.size(); ++i)
    if (!ranges[i].empty()) {
      root_parameters_[i / 2][i % 2] = uint32_t(parameters.size());
      D3D12_ROOT_PARAMETER p{};
      p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
      p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
      p.DescriptorTable = {UINT(ranges[i].size()), ranges[i].data()};
      parameters.push_back(p);
    }
  root.NumParameters = UINT(parameters.size());
  root.pParameters = parameters.data();
  ComPtr<ID3DBlob> blob, errors;
  if (!Check(D3D12SerializeRootSignature(&root, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &errors),
             "Serialize root signature"))
    return false;
  if (!Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                         IID_PPV_ARGS(&root_signature_)),
             "Create root signature"))
    return false;
  return true;
}
}  // namespace grassland::rhi::backend::d3d12

LONGMARCH_RHI_UUID(grassland::rhi::backend::d3d12::ID3D12RayTracingProgram, "5d1da804-a5b3-4469-a380-9d7a44cee916")
