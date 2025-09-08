#include "grassland/graphics/graphics.h"

namespace CD::graphics {

void PyBind(pybind11::module_ &m) {
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

  pybind11::class_<ColorClearValue> color_clear_value(m, "ColorClearValue");
  color_clear_value.def(pybind11::init<float, float, float, float>(), pybind11::arg("r") = 0.0f,
                        pybind11::arg("g") = 0.0f, pybind11::arg("b") = 0.0f, pybind11::arg("a") = 1.0f);
  color_clear_value.def_readwrite("r", &ColorClearValue::r);
  color_clear_value.def_readwrite("g", &ColorClearValue::g);
  color_clear_value.def_readwrite("b", &ColorClearValue::b);
  color_clear_value.def_readwrite("a", &ColorClearValue::a);
  color_clear_value.def("__repr__", [](const ColorClearValue &c) {
    return pybind11::str("ColorClearValue({},{},{},{})").format(c.r, c.g, c.b, c.a);
  });

  pybind11::class_<DepthClearValue> depth_clear_value(m, "DepthClearValue");
  depth_clear_value.def(pybind11::init<float>(), pybind11::arg("depth") = 1.0f);
  depth_clear_value.def_readwrite("depth", &DepthClearValue::depth);
  depth_clear_value.def("__repr__",
                        [](const DepthClearValue &c) { return pybind11::str("DepthClearValue({})").format(c.depth); });

  pybind11::class_<Extent2D> extent2d(m, "Extent2D");
  extent2d.def(pybind11::init<uint32_t, uint32_t>(), pybind11::arg("width") = 0, pybind11::arg("height") = 0);
  extent2d.def_readwrite("width", &Extent2D::width);
  extent2d.def_readwrite("height", &Extent2D::height);
  extent2d.def("__repr__",
               [](const Extent2D &e) { return pybind11::str("Extent2D({}, {})").format(e.width, e.height); });

  Core::PyBind(m);
  Window::PyBind(m);
  Image::PyBind(m);
  CommandContext::PyBind(m);
}

}  // namespace CD::graphics
