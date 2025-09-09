#pragma once
#include "snowberg/draw/draw_util.h"

namespace snowberg::draw {

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

}  // namespace snowberg::draw
