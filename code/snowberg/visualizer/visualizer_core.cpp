#include "snowberg/visualizer/visualizer_core.h"

#include <queue>

#include "snowberg/visualizer/visualizer_camera.h"
#include "snowberg/visualizer/visualizer_entity.h"
#include "snowberg/visualizer/visualizer_film.h"
#include "snowberg/visualizer/visualizer_mesh.h"
#include "snowberg/visualizer/visualizer_ownership_holder.h"
#include "snowberg/visualizer/visualizer_render_context.h"
#include "snowberg/visualizer/visualizer_scene.h"

namespace snowberg::visualizer {
Core::Core(graphics::Core *core) : core_(core) {
  ownership_holders_.resize(core->FramesInFlight());
}

graphics::Core *Core::GraphicsCore() const {
  return core_;
}

std::shared_ptr<Core> Core::CreateCore(graphics::Core *graphics_core) {
  return std::shared_ptr<Core>(new Core(graphics_core));
}

std::shared_ptr<Camera> Core::CreateCamera(const Matrix4<float> &proj, const Matrix4<float> &view) {
  std::shared_ptr<Camera> camera = std::shared_ptr<Camera>(new Camera{shared_from_this()});
  camera->proj = EigenToGLM(proj);
  camera->view = EigenToGLM(view);
  return camera;
}

std::shared_ptr<Mesh> Core::CreateMesh() {
  return std::shared_ptr<Mesh>{new Mesh(shared_from_this())};
}

std::shared_ptr<Film> Core::CreateFilm(int width, int height) {
  return std::shared_ptr<Film>{new Film(shared_from_this(), width, height)};
}

std::shared_ptr<Scene> Core::CreateScene() {
  return std::shared_ptr<Scene>{new Scene(shared_from_this())};
}

std::shared_ptr<Core> CreateCore(graphics::Core *graphics_core) {
  return Core::CreateCore(graphics_core);
}

int Core::Render(graphics::CommandContext *context,
                 const std::shared_ptr<Scene> &scene,
                 const std::shared_ptr<Camera> &camera,
                 const std::shared_ptr<Film> &film) {
  std::vector<graphics::Image *> images = {film->GetImage(FILM_CHANNEL_EXPOSURE), film->GetImage(FILM_CHANNEL_ALBEDO),
                                           film->GetImage(FILM_CHANNEL_POSITION), film->GetImage(FILM_CHANNEL_NORMAL)};
  CameraInfo cam_info = camera->GetInfo();
  camera->camera_buffer_->UploadData(&cam_info, sizeof(CameraInfo));

  RenderContext render_context{};
  render_context.cmd_ctx = context;
  render_context.film = film.get();
  render_context.camera_buffer = camera->camera_buffer_.get();
  render_context.ownership_holder = &ownership_holders_[context->GetCore()->CurrentFrame()];

  render_context.ownership_holder->AddCamera(camera);
  render_context.ownership_holder->AddFilm(film);
  context->CmdClearImage(film->GetImage(FILM_CHANNEL_EXPOSURE), {0.0f, 0.0f, 0.0f, 0.0f});
  context->CmdClearImage(film->GetImage(FILM_CHANNEL_ALBEDO), {0.0f, 0.0f, 0.0f, 0.0f});
  context->CmdClearImage(film->GetImage(FILM_CHANNEL_POSITION), {0.0f, 0.0f, 0.0f, 0.0f});
  context->CmdClearImage(film->GetImage(FILM_CHANNEL_NORMAL), {0.0f, 0.0f, 0.0f, 0.0f});
  context->CmdClearImage(film->GetImage(FILM_CHANNEL_DEPTH), {1.0f, 0.0f, 0.0f, 0.0f});
  context->CmdBeginRendering(images, film->GetImage(FILM_CHANNEL_DEPTH));

  graphics::Extent2D extent = film->Extent();
  graphics::Viewport viewport = {0.0f, 0.0f, static_cast<float>(extent.width), static_cast<float>(extent.height),
                                 0.0f, 1.0f};
  graphics::Scissor scissor = {{0, 0}, extent};
  context->CmdSetViewport(viewport);
  context->CmdSetScissor(scissor);

  std::queue<uint64_t> to_del_entities;
  std::vector<std::shared_ptr<Entity>> entities_holder;
  for (auto &[id, entity] : scene->entities_) {
    auto locked_entity = entity.lock();
    render_context.ownership_holder->AddEntity(locked_entity);
    if (locked_entity) {
      if (locked_entity->ExecuteStage(RENDER_STAGE_RASTER_GEOMETRY_PASS, render_context)) {
        to_del_entities.push(id);
      }
      entities_holder.push_back(std::move(locked_entity));
    } else {
      to_del_entities.push(id);
    }
  }

  context->CmdEndRendering();

  context->CmdBeginRendering({film->GetImage(FILM_CHANNEL_EXPOSURE)}, nullptr);

  for (auto &entity : entities_holder) {
    if (entity) {
      entity->ExecuteStage(RENDER_STAGE_RASTER_LIGHTING_PASS, render_context);
    }
  }

  context->CmdEndRendering();

  context->PushPostExecutionCallback([holder = render_context.ownership_holder]() { holder->Clear(); });
  while (!to_del_entities.empty()) {
    scene->entities_.erase(to_del_entities.front());
    to_del_entities.pop();
  }
  return 0;
}

}  // namespace snowberg::visualizer
