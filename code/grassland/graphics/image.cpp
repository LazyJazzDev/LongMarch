#include "grassland/graphics/image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "core.h"
#include "stb_image.h"

namespace grassland::graphics {

#if defined(LONGMARCH_PYTHON_ENABLED)
void Image::PybindClassRegistration(py::classh<Image> &c) {
  c.def("extent", &Image::Extent, "Get the image extent (width, height)");
  c.def("format", &Image::Format, "Get the image format");

  // Overloaded upload_data function
  c.def(
      "upload_data", [](Image *image, py::bytes data) { image->UploadData(PyBytes_AsString(data.ptr())); },
      py::arg("data"), "Upload full data to the image");
  c.def(
      "upload_data",
      [](Image *image, py::bytes data, const Offset2D &offset, const Extent2D &extent) {
        image->UploadData(PyBytes_AsString(data.ptr()), offset, extent);
      },
      py::arg("data"), py::arg("offset"), py::arg("extent"),
      "Upload partial data to the image at specified offset and extent");

  // Overloaded download_data function
  c.def(
      "download_data",
      [](Image *image) {
        // Calculate expected data size based on image format and dimensions
        auto extent = image->Extent();
        auto format = image->Format();

        size_t bytes_per_pixel = PixelSize(format);
        size_t total_size = extent.width * extent.height * bytes_per_pixel;

        // Download data to vector first
        std::vector<uint8_t> data(total_size);
        image->DownloadData(data.data());

        // Create py::bytes from vector data
        return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
      },
      "Download full data from the image");
  c.def(
      "download_data",
      [](Image *image, const Offset2D &offset, const Extent2D &extent) {
        // Calculate expected data size based on image format and partial dimensions
        auto format = image->Format();
        size_t bytes_per_pixel = PixelSize(format);
        size_t partial_size = extent.width * extent.height * bytes_per_pixel;

        // Download partial data to vector first
        std::vector<uint8_t> data(partial_size);
        image->DownloadData(data.data(), offset, extent);

        // Create py::bytes from vector data
        return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
      },
      py::arg("offset"), py::arg("extent"), "Download partial data from the image at specified offset and extent");

  c.def("__repr__", [](Image *image) {
    auto extent = image->Extent();
    return py::str("Image(width={}, height={})").format(extent.width, extent.height);
  });
}
#endif

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

}  // namespace grassland::graphics
