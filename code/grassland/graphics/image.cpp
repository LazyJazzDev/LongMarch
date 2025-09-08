#include "grassland/graphics/image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "core.h"
#include "stb_image.h"

namespace CD::graphics {

void Image::PyBind(pybind11::module &m) {
  pybind11::class_<Image, std::shared_ptr<Image>> image(m, "Image");
  image.def("extent", &Image::Extent);
  image.def("format", &Image::Format);
  image.def("__repr__", [](const Image &image) {
    return pybind11::str("Image({}, {}, {})").format(image.Extent().width, image.Extent().height, image.Format());
  });
}

int LoadImageFromFile(Core *core, const std::string &file_path, double_ptr<Image> pp_image) {
  int w, h, c;
  {
    auto data = stbi_load(file_path.c_str(), &w, &h, &c, 4);
    if (data) {
      core->CreateImage(w, h, IMAGE_FORMAT_R8G8B8A8_UNORM, pp_image);
      pp_image->UploadData(data);
      stbi_image_free(data);
      return 0;
    }
  }
  {
    auto data = stbi_loadf(file_path.c_str(), &w, &h, &c, 4);
    if (data) {
      core->CreateImage(w, h, IMAGE_FORMAT_R32G32B32A32_SFLOAT, pp_image);
      pp_image->UploadData(data);
      stbi_image_free(data);
      return 0;
    }
  }
  return -1;
}

}  // namespace CD::graphics
