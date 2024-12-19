#pragma once
#include "grassland/d3d12/direct3d12.h"
#include "grassland/graphics/interface.h"

namespace grassland::graphics::backend {

DXGI_FORMAT ImageFormatToDXGIFormat(ImageFormat format);

DXGI_FORMAT InputTypeToDXGIFormat(InputType type);

D3D12_DESCRIPTOR_RANGE_TYPE ResourceTypeToD3D12DescriptorRangeType(
    ResourceType type);

D3D12_CULL_MODE CullModeToD3D12CullMode(CullMode mode);

class D3D12Core;
class D3D12Buffer;
class D3D12Image;
class D3D12Shader;
class D3D12Program;
class D3D12CommandContext;
class D3D12Window;

struct D3D12ResourceBinding {
  D3D12ResourceBinding();

  D3D12ResourceBinding(D3D12Buffer *buffer);

  D3D12ResourceBinding(D3D12Image *image);

  D3D12Buffer *buffer;
  D3D12Image *image;
};

}  // namespace grassland::graphics::backend
