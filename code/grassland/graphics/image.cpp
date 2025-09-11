#include "grassland/graphics/image.h"

#define STB_IMAGE_IMPLEMENTATION
#include "core.h"
#include "stb_image.h"

namespace grassland::graphics {

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
