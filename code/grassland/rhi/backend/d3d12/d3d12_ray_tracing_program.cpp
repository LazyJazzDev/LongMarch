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

D3D12_DISPATCH_RAYS_DESC D3D12RayTracingProgram::DispatchDesc() const {
  return dispatch_;
}

bool D3D12RayTracingProgram::Finalize() {
  if (finalized_)
    return true;
  if (!ValidateLayout() || desc_.max_recursion_depth > 31 || desc_.max_attribute_size > 32)
    return false;
  auto owner = static_cast<D3D12Device *>(device_);
  auto native = owner->NativeDevice();
  ComPtr<ID3D12Device5> device;
  if (FAILED(native->QueryInterface(IID_PPV_ARGS(&device))))
    return false;
  if (!CreateLayout(native, desc_.pipeline_layout, D3D12_ROOT_SIGNATURE_FLAG_NONE))
    return false;
  std::vector<IShader *> shaders{desc_.ray_generation};
  shaders.insert(shaders.end(), desc_.miss_shaders.begin(), desc_.miss_shaders.end());
  std::vector<std::array<uint32_t, 3>> indices;
  for (auto group : desc_.hit_groups) {
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
  shaders.insert(shaders.end(), desc_.callable_shaders.begin(), desc_.callable_shaders.end());
  std::vector<std::vector<uint8_t>> code(shaders.size());
  std::vector<std::wstring> names(shaders.size()), entries(shaders.size()), hit_names(indices.size());
  std::vector<D3D12_EXPORT_DESC> exports(shaders.size());
  std::vector<D3D12_DXIL_LIBRARY_DESC> libraries(shaders.size());
  std::vector<D3D12_HIT_GROUP_DESC> hits(indices.size());
  std::vector<D3D12_STATE_SUBOBJECT> subobjects;
  auto header = ShaderMacroHeader(desc_.pipeline_layout, BACKEND_API_D3D12);
  for (size_t i = 0; i < shaders.size(); ++i) {
    if (!CompileSlang(shaders[i]->GetDesc(), header, BACKEND_API_D3D12, code[i], &desc_.pipeline_layout))
      return false;
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
  D3D12_RAYTRACING_SHADER_CONFIG shader_config{desc_.max_payload_size, desc_.max_attribute_size};
  D3D12_RAYTRACING_PIPELINE_CONFIG pipeline_config{desc_.max_recursion_depth};
  auto root = NativeRootSignature();
  subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader_config});
  subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipeline_config});
  subobjects.push_back({D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &root});
  D3D12_STATE_OBJECT_DESC create{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, UINT(subobjects.size()),
                                 subobjects.data()};
  ComPtr<ID3D12StateObject> pipeline;
  if (!Check(device->CreateStateObject(&create, IID_PPV_ARGS(&pipeline)), "Create ray tracing pipeline"))
    return false;
  ComPtr<ID3D12StateObjectProperties> properties;
  if (FAILED(pipeline.As(&properties)))
    return false;
  std::vector<uint8_t> data;
  std::array<uint64_t, 4> offsets{}, sizes{};
  std::vector<std::vector<std::wstring>> tables{{names[0]}, {}, hit_names, {}};
  for (size_t i = 0; i < desc_.miss_shaders.size(); ++i)
    tables[1].push_back(names[1 + i]);
  for (size_t i = callable_start; i < names.size(); ++i)
    tables[3].push_back(names[i]);
  for (size_t t = 0; t < 4; ++t) {
    data.resize((data.size() + 63) & ~size_t(63));
    offsets[t] = data.size();
    sizes[t] = tables[t].size() * 32;
    data.resize(data.size() + sizes[t]);
    for (size_t i = 0; i < tables[t].size(); ++i) {
      auto identifier = properties->GetShaderIdentifier(tables[t][i].c_str());
      if (!identifier)
        return false;
      std::memcpy(data.data() + offsets[t] + i * 32, identifier, 32);
    }
  }
  auto table =
      ObjectPtr<D3D12AllocatedBuffer>::Adopt(D3D12AllocatedBuffer::Create(owner, {uint64_t(data.size())}, true));
  if (!table)
    return false;
  if (!table->Write(data.data(), data.size()))
    return false;
  const auto address = table->NativeBuffer()->GetGPUVirtualAddress();
  dispatch_.RayGenerationShaderRecord = {address, 32};
  if (sizes[1])
    dispatch_.MissShaderTable = {address + offsets[1], sizes[1], 32};
  if (sizes[2])
    dispatch_.HitGroupTable = {address + offsets[2], sizes[2], 32};
  if (sizes[3])
    dispatch_.CallableShaderTable = {address + offsets[3], sizes[3], 32};
  table_ = std::move(table);
  pipeline_ = std::move(pipeline);
  finalized_ = true;
  return true;
}
}  // namespace grassland::rhi::backend::d3d12
