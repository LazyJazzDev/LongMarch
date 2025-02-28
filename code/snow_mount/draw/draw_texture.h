#pragma once
#include "snow_mount/draw/draw_util.h"

namespace snow_mount::draw {

class Texture {
 public:
  Texture(Core *core, int width, int height);

  Core *DrawCore() const {
    return core_;
  }

  graphics::Image *Image() const {
    return image_.get();
  }

  void UploadData(const void *data) {
    image_->UploadData(data);
  }

 private:
  Core *core_;
  std::unique_ptr<graphics::Image> image_;
};

}  // namespace snow_mount::draw
