#pragma once
#define NOMINMAX

#include <d3d12.h>
#include <d3dcompiler.h>
#include <d3dx12.h>
#include <dxgi1_6.h>
#include <wrl.h>

#include "GLFW/glfw3.h"
#define GLFW_EXPOSE_NATIVE_WIN32
#include "GLFW/glfw3native.h"
#include "grassland/util/util.h"

namespace grassland::d3d12 {

using Microsoft::WRL::ComPtr;

void ThrowError(const std::string &message);

template <class... Args>
void ThrowError(const std::string &message, Args &&...args) {
  ThrowError(fmt::format(message, std::forward<Args>(args)...));
}

void ThrowIfFailed(HRESULT hr, const std::string &message);

template <class... Args>
void ThrowIfFailed(HRESULT hr, const std::string &message, Args &&...args) {
  ThrowIfFailed(hr, fmt::format(message, std::forward<Args>(args)...));
}

void Warning(const std::string &message);

template <class... Args>
void Warning(const std::string &message, Args &&...args) {
  Warning(fmt::format(message, std::forward<Args>(args)...));
}

void SetErrorMessage(const std::string &message);

template <class... Args>
void SetErrorMessage(const std::string &message, Args &&...args) {
  SetErrorMessage(fmt::format(message, std::forward<Args>(args)...));
}

std::string GetErrorMessage();

std::string HRESULTToString(HRESULT hr);

#define RETURN_IF_FAILED_HR(cmd, ...)                               \
  do {                                                              \
    HRESULT res = cmd;                                              \
    if (FAILED(res)) {                                              \
      ::grassland::d3d12::SetErrorMessage(__VA_ARGS__);             \
      ::grassland::d3d12::SetErrorMessage(                          \
          "HRESULT: {}", ::grassland::d3d12::HRESULTToString(res)); \
      return res;                                                   \
    }                                                               \
                                                                    \
  } while (false)

struct DeviceFeatureRequirement;

class DXGIFactory;
class Adapter;
class Device;

}  // namespace grassland::d3d12
