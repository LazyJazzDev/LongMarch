#include "snow_mount/visualizer/visualizer_util.h"

namespace XS::visualizer {

void Material::PyBind(pybind11::module_ &m) {
  pybind11::class_<Material> material(m, "Material");
  material.def(pybind11::init([](const Vector4<float> &albedo) {
                 Material material;
                 material.albedo = EigenToGLM(albedo);
                 return material;
               }),
               pybind11::arg("albedo") = Vector4<float>{0.8f, 0.8f, 0.8f, 1.0f});
  material.def_property(
      "albedo", [](const Material &material) { return GLMToEigen(material.albedo); },
      [](Material &material, const Vector4<float> &albedo) { material.albedo = EigenToGLM(albedo); });
  material.def("__repr__", [](const Material &material) {
    return fmt::format("Material(albedo=<{}, {}, {}, {}>)", material.albedo.r, material.albedo.g, material.albedo.b,
                       material.albedo.a);
  });
}

graphics::ImageFormat TextureTypeToImageFormat(TextureType type) {
  switch (type) {
    case TEXTURE_TYPE_HDR:
      return graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT;
    case TEXTURE_TYPE_SDR:
      return graphics::IMAGE_FORMAT_R8G8B8A8_UNORM;
  }
  return graphics::IMAGE_FORMAT_R8G8B8A8_UNORM;
}

graphics::ImageFormat FilmChannelImageFormat(FilmChannel channel) {
  switch (channel) {
    case FILM_CHANNEL_EXPOSURE:
    case FILM_CHANNEL_ALBEDO:
    case FILM_CHANNEL_POSITION:
    case FILM_CHANNEL_NORMAL:
      return graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT;
    case FILM_CHANNEL_DEPTH:
      return graphics::IMAGE_FORMAT_D32_SFLOAT;
    default:
      throw std::runtime_error("Invalid FilmChannel");
  }
}

}  // namespace XS::visualizer
