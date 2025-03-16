#include "grassland/graphics/image.h"

namespace grassland::graphics {

void Image::PyBind(pybind11::module &m) {
  pybind11::class_<Image, std::shared_ptr<Image>> image(m, "Image");
  image.def("extent", &Image::Extent);
  image.def("format", &Image::Format);
  image.def("__repr__", [](const Image &image) {
    return pybind11::str("Image({}, {}, {})").format(image.Extent().width, image.Extent().height, image.Format());
  });
}

}  // namespace grassland::graphics
