#include <long_march.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace long_march;

int main() {
  std::unique_ptr<graphics::Core> core_;

  graphics::CreateCore(graphics::BACKEND_API_VULKAN, graphics::Core::Settings{}, &core_);
  core_->InitializeLogicalDeviceAutoSelect(true);
  sparks::Core sparks_core(core_.get());
  sparks_core.GetShadersVFS().Print();

  sparks::Scene scene(&sparks_core);
  sparks::Film film(&sparks_core, 1280, 720);
  Mesh<float> cube_mesh;
  cube_mesh.LoadObjFile(FindAssetFile("meshes/cube.obj"));
  sparks::Geometry geometry(&sparks_core, cube_mesh);
  sparks::Camera camera(&sparks_core);

  scene.Render(&camera, &film);

  std::unique_ptr<graphics::Image> image;
  core_->CreateImage(1280, 720, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &image);
  sparks_core.ConvertFilmToImage(film, image.get());
  std::vector<uint8_t> image_data(1280 * 720 * 4);
  image->DownloadData(image_data.data());
  stbi_write_bmp("output.bmp", 1280, 720, 4, image_data.data());
}
