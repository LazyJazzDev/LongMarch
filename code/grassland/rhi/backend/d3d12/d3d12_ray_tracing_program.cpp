#include "grassland/rhi/backend/d3d12/d3d12_ray_tracing_program.h"

#include <cstring>

#include "grassland/rhi/backend/backend_shader_compiler.h"
#include "grassland/rhi/backend/d3d12/d3d12_device.h"

namespace grassland::rhi::backend::d3d12 {
D3D12RayTracingProgram::D3D12RayTracingProgram(D3D12Device *device, const RayTracingProgramDesc &desc)
    : RayTracingProgram(device, desc) {
}

ID3D12StateObject *D3D12RayTracingProgram::NativePipeline() const {
  return pipeline_.Get();
}

const ShaderBindingTable *D3D12RayTracingProgram::DefaultShaderBindingTable() const {
  return table_.Get();
}

IShaderBindingTable *D3D12RayTracingProgram::CreateShaderBindingTable(const ShaderBindingTableDesc &desc) {
  if (!ValidateShaderBindingTable(desc))
    return nullptr;
  return BuildShaderBindingTable(desc);
}

D3D12RayTracingProgram *D3D12RayTracingProgram::Create(D3D12Device *owner_device, const RayTracingProgramDesc &desc) {
  auto object = std::unique_ptr<D3D12RayTracingProgram>(new D3D12RayTracingProgram(owner_device, desc));
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
    if (!CompileSlang(ShaderCompileDesc(shaders[i]), header, BACKEND_API_D3D12, code[i],
                      &object->desc_.pipeline_layout))
      return nullptr;
    names[i] = L"rhi_shader_" + std::to_wstring(i);
    const auto &entry = shaders[i]->GetDesc().entry_point;
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
  object->table_ =
      ObjectPtr<ShaderBindingTable>::Adopt(object->BuildShaderBindingTable(object->DefaultShaderBindingTableDesc()));
  if (!object->table_)
    return nullptr;
  object->pipeline_ = std::move(pipeline);
  return object.release();
}

ShaderBindingTable *D3D12RayTracingProgram::BuildShaderBindingTable(const ShaderBindingTableDesc &desc) {
  constexpr uint64_t stride = D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES;
  constexpr uint64_t alignment = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
  std::vector<uint8_t> data;
  std::array<ShaderBindingTable::Region, 4> regions{};
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
  auto owner = static_cast<D3D12Device *>(device_);
  auto buffer =
      ObjectPtr<D3D12AllocatedBuffer>::Adopt(D3D12AllocatedBuffer::Create(owner, {uint64_t(data.size())}, true));
  if (!buffer || !buffer->Write(data.data(), data.size()))
    return nullptr;
  auto address = buffer->NativeBuffer()->GetGPUVirtualAddress();
  for (auto &region : regions)
    if (region.size)
      region.address += address;
  return new ShaderBindingTable(this, desc, std::move(buffer), regions);
}

}  // namespace grassland::rhi::backend::d3d12
