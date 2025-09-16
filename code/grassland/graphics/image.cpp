#include "grassland/graphics/image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "core.h"
#include "stb_image.h"

namespace grassland::graphics {

namespace {
// Utility function to get bytes per pixel for image formats
size_t GetImageFormatBytesPerPixel(ImageFormat format) {
  switch (format) {
    case IMAGE_FORMAT_R8G8B8A8_UNORM:
    case IMAGE_FORMAT_B8G8R8A8_UNORM:
      return 4;
    case IMAGE_FORMAT_R32G32B32A32_SFLOAT:
      return 16;
    case IMAGE_FORMAT_R32G32B32_SFLOAT:
      return 12;
    case IMAGE_FORMAT_R32G32_SFLOAT:
      return 8;
    case IMAGE_FORMAT_R32_SFLOAT:
    case IMAGE_FORMAT_D32_SFLOAT:
      return 4;
    case IMAGE_FORMAT_R16G16B16A16_SFLOAT:
      return 8;
    case IMAGE_FORMAT_R32_UINT:
    case IMAGE_FORMAT_R32_SINT:
      return 4;
    case IMAGE_FORMAT_UNDEFINED:
    default:
      return 4;  // Default fallback
  }
}
}  // namespace

void Image::PybindClassRegistration(py::classh<Image> &c) {
  c.def("extent", &Image::Extent, "Get the image extent (width, height)");
  c.def("format", &Image::Format, "Get the image format");
  c.def(
      "upload_data", [](Image *image, py::bytes data) { image->UploadData(PyBytes_AsString(data.ptr())); },
      py::arg("data"), "Upload data to the image");
  c.def(
      "download_data",
      [](Image *image) {
        // Calculate expected data size based on image format and dimensions
        auto extent = image->Extent();
        auto format = image->Format();

        size_t bytes_per_pixel = GetImageFormatBytesPerPixel(format);
        size_t total_size = extent.width * extent.height * bytes_per_pixel;

        // Download data to vector first
        std::vector<uint8_t> data(total_size);
        image->DownloadData(data.data());

        // Create py::bytes from vector data
        return py::bytes(reinterpret_cast<const char *>(data.data()), data.size());
      },
      "Download data from the image");
  c.def("__repr__", [](Image *image) {
    auto extent = image->Extent();
    return py::str("Image(width={}, height={})").format(extent.width, extent.height);
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

}  // namespace grassland::graphics
