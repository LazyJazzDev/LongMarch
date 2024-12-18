#pragma once
#include "grassland/d3d12/direct3d12.h"
#include "grassland/graphics/interface.h"

namespace grassland::graphics::backend {

DXGI_FORMAT ImageFormatToDXGIFormat(ImageFormat format);

DXGI_FORMAT InputTypeToDXGIFormat(InputType type);
class D3D12Core;
class D3D12Buffer;
class D3D12Image;
class D3D12Shader;
class D3D12Program;
class D3D12CommandContext;
class D3D12Window;

}  // namespace grassland::graphics::backend
