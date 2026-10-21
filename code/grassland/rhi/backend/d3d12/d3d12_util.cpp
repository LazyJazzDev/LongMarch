#include "grassland/rhi/backend/d3d12/d3d12_util.h"

#include "grassland/rhi/backend/backend_binding_group.h"

namespace grassland::rhi::backend::d3d12 {
bool CreateRootSignature(ID3D12Device *device,
                         const PipelineLayoutDesc &layout,
                         bool graphics,
                         ComPtr<ID3D12RootSignature> &signature,
                         std::vector<std::array<uint32_t, 2>> &parameters_map) {
  D3D12_ROOT_SIGNATURE_DESC root{};
  root.Flags = graphics ? D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT : D3D12_ROOT_SIGNATURE_FLAG_NONE;
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
  parameters_map.assign(layout.groups.size(), {UINT32_MAX, UINT32_MAX});
  for (size_t i = 0; i < ranges.size(); ++i)
    if (!ranges[i].empty()) {
      parameters_map[i / 2][i % 2] = uint32_t(parameters.size());
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
  if (!Check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&signature)),
             "Create root signature"))
    return false;
  return true;
}

D3D12_BLEND NativeBlendFactor(BlendFactor factor, bool alpha) {
  // Alpha equations use the alpha component of color factors as well.
  if (alpha && factor >= BLEND_FACTOR_SRC_COLOR && factor <= BLEND_FACTOR_ONE_MINUS_DST_COLOR)
    factor = static_cast<BlendFactor>(factor + BLEND_FACTOR_SRC_ALPHA - BLEND_FACTOR_SRC_COLOR);
  const D3D12_BLEND factors[] = {D3D12_BLEND_ZERO,          D3D12_BLEND_ONE,           D3D12_BLEND_SRC_COLOR,
                                 D3D12_BLEND_INV_SRC_COLOR, D3D12_BLEND_DEST_COLOR,    D3D12_BLEND_INV_DEST_COLOR,
                                 D3D12_BLEND_SRC_ALPHA,     D3D12_BLEND_INV_SRC_ALPHA, D3D12_BLEND_DEST_ALPHA,
                                 D3D12_BLEND_INV_DEST_ALPHA};
  return factors[factor];
}

D3D12_BLEND_OP NativeBlendOp(BlendOp op) {
  const D3D12_BLEND_OP operations[] = {D3D12_BLEND_OP_ADD, D3D12_BLEND_OP_SUBTRACT, D3D12_BLEND_OP_REV_SUBTRACT,
                                       D3D12_BLEND_OP_MIN, D3D12_BLEND_OP_MAX};
  return operations[op];
}

bool SupportsImageStorage(ID3D12Device *device, ImageFormat format) {
  const auto native = NativeImageFormat(format);
  if (native == DXGI_FORMAT_UNKNOWN || IsDepthFormat(format))
    return false;
  D3D12_FEATURE_DATA_FORMAT_SUPPORT support{native};
  if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))))
    return false;
  return (support.Support1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) != 0;
}

DXGI_FORMAT NativeImageFormat(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_RGBA8_UNORM:
      return DXGI_FORMAT_R8G8B8A8_UNORM;
    case IMAGE_FORMAT_BGRA8_UNORM:
      return DXGI_FORMAT_B8G8R8A8_UNORM;
    case IMAGE_FORMAT_D32_FLOAT:
      return DXGI_FORMAT_D32_FLOAT;
    case IMAGE_FORMAT_D24_UNORM_S8_UINT:
      return DXGI_FORMAT_D24_UNORM_S8_UINT;
    default:
      return DXGI_FORMAT_UNKNOWN;
  }
}

bool Check(HRESULT result, const char *operation) {
  if (SUCCEEDED(result))
    return true;
  ReportError(result, operation);
  return false;
}
}  // namespace grassland::rhi::backend::d3d12
