#pragma once
#include "grassland/graphics/graphics_util.h"

namespace grassland::graphics {

class Image {
 public:
  virtual ~Image() = default;
  virtual Extent2D Extent() const = 0;
  virtual ImageFormat Format() const = 0;
  virtual void UploadData(const void *data) const = 0;
  virtual void DownloadData(void *data) const = 0;
};

int LoadImageFromFile(Core *core, const std::string &file_path, double_ptr<Image> pp_image);

}  // namespace grassland::graphics
