#include "grassland/graphics/graphics.h"

namespace grassland::graphics {

void PybindModuleRegistration(pybind11::module_ &m) {
  pybind11::enum_<BackendAPI> backend_api(m, "BackendAPI");
  backend_api.value("BACKEND_API_VULKAN", BackendAPI::BACKEND_API_VULKAN);
  backend_api.value("BACKEND_API_D3D12", BackendAPI::BACKEND_API_D3D12);
  backend_api.export_values();

  pybind11::enum_<ImageFormat> image_format(m, "ImageFormat");
  image_format.value("IMAGE_FORMAT_UNDEFINED", ImageFormat::IMAGE_FORMAT_UNDEFINED);
  image_format.value("IMAGE_FORMAT_B8G8R8A8_UNORM", ImageFormat::IMAGE_FORMAT_B8G8R8A8_UNORM);
  image_format.value("IMAGE_FORMAT_R8G8B8A8_UNORM", ImageFormat::IMAGE_FORMAT_R8G8B8A8_UNORM);
  image_format.value("IMAGE_FORMAT_R32G32B32A32_SFLOAT", ImageFormat::IMAGE_FORMAT_R32G32B32A32_SFLOAT);
  image_format.value("IMAGE_FORMAT_R32G32B32_SFLOAT", ImageFormat::IMAGE_FORMAT_R32G32B32_SFLOAT);
  image_format.value("IMAGE_FORMAT_R32G32_SFLOAT", ImageFormat::IMAGE_FORMAT_R32G32_SFLOAT);
  image_format.value("IMAGE_FORMAT_R32_SFLOAT", ImageFormat::IMAGE_FORMAT_R32_SFLOAT);
  image_format.value("IMAGE_FORMAT_D32_SFLOAT", ImageFormat::IMAGE_FORMAT_D32_SFLOAT);
  image_format.value("IMAGE_FORMAT_R16G16B16A16_SFLOAT", ImageFormat::IMAGE_FORMAT_R16G16B16A16_SFLOAT);
  image_format.export_values();

  pybind11::enum_<InputType> input_type(m, "InputType");
  input_type.value("INPUT_TYPE_UINT", InputType::INPUT_TYPE_UINT);
  input_type.value("INPUT_TYPE_INT", InputType::INPUT_TYPE_INT);
  input_type.value("INPUT_TYPE_FLOAT", InputType::INPUT_TYPE_FLOAT);
  input_type.value("INPUT_TYPE_UINT2", InputType::INPUT_TYPE_UINT2);
  input_type.value("INPUT_TYPE_INT2", InputType::INPUT_TYPE_INT2);
  input_type.value("INPUT_TYPE_FLOAT2", InputType::INPUT_TYPE_FLOAT2);
  input_type.value("INPUT_TYPE_UINT3", InputType::INPUT_TYPE_UINT3);
  input_type.value("INPUT_TYPE_INT3", InputType::INPUT_TYPE_INT3);
  input_type.value("INPUT_TYPE_FLOAT3", InputType::INPUT_TYPE_FLOAT3);
  input_type.value("INPUT_TYPE_UINT4", InputType::INPUT_TYPE_UINT4);
  input_type.value("INPUT_TYPE_INT4", InputType::INPUT_TYPE_INT4);
  input_type.value("INPUT_TYPE_FLOAT4", InputType::INPUT_TYPE_FLOAT4);
  input_type.export_values();

  pybind11::enum_<BufferType> buffer_type(m, "BufferType");
  buffer_type.value("BUFFER_TYPE_VERTEX", BufferType::BUFFER_TYPE_STATIC);
  buffer_type.value("BUFFER_TYPE_INDEX", BufferType::BUFFER_TYPE_DYNAMIC);

  Core::PybindModuleRegistration(m);
  Window::PybindModuleRegistration(m);
}

}  // namespace grassland::graphics
