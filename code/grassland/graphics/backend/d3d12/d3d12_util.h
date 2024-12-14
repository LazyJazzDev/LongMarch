#pragma once
#include "grassland/d3d12/direct3d12.h"
#include "grassland/graphics/interface.h"

namespace grassland::graphics::backend {

DXGI_FORMAT ImageFormatToDXGIFormat(ImageFormat format);

DXGI_FORMAT InputTypeToDXGIFormat(InputType type);

}  // namespace grassland::graphics::backend
