#pragma once
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include "grassland/rhi/backend/backend_util.h"

namespace grassland::rhi::backend::d3d12 {
bool SupportsImageStorage(ID3D12Device *device, ImageFormat format);
DXGI_FORMAT NativeImageFormat(ImageFormat format);
using Microsoft::WRL::ComPtr;
bool Check(HRESULT result, const char *operation);
}  // namespace grassland::rhi::backend::d3d12
