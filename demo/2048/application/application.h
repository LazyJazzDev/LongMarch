#pragma once

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "long_march.h"

using namespace grassland;

class DeviceModel;

class Listener;

struct InstanceInfo {
  glm::mat4 model;
  glm::vec4 color;
  glm::uvec4 extra;
};

// A small 2D instanced renderer on top of grassland::graphics. Models are drawn
// in framebuffer pixel coordinates (origin at the top-left corner) with depth
// testing, then resolved from a supersampled target for anti-aliasing.
class Application {
 public:
  Application(const std::string &name, int width, int height, graphics::BackendAPI api);

  virtual ~Application();

  // Runs until the window closes, or after max_frames frames when positive.
  void Run(int max_frames = 0);

  // Saves the last presented frame as a PNG file when the application exits.
  void SetScreenshotPath(const std::string &path) {
    screenshot_path_ = path;
  }

  [[nodiscard]] graphics::Core *Core() const {
    return core_.get();
  }

  [[nodiscard]] graphics::Window *GetWindow() const {
    return window_.get();
  }

  [[nodiscard]] std::string Name() const {
    return name_;
  }

  // Framebuffer size in pixels, the coordinate space of all draw calls.
  [[nodiscard]] glm::ivec2 FramebufferSize() const {
    return framebuffer_size_;
  }

  void DrawModel(DeviceModel *device_model, const InstanceInfo &instance_info);

  // Moves every model drawn so far this frame into a second frame, which is
  // blended over the main frame with the given opacity.
  void CaptureSecondFrame(float alpha);

  void RegisterListener(Listener *listener);

  void UnregisterListener(Listener *listener);

 protected:
  virtual void CustomOnUpdate();

  virtual void CustomOnClose();

  virtual void CustomOnInit();

  // Called after the framebuffer size changes, before the next update.
  virtual void OnFramebufferResize();

  glm::vec4 clear_color_{0.0f, 0.0f, 0.0f, 1.0f};

 private:
  struct FrameTarget {
    std::unique_ptr<graphics::Image> color_image;
    std::unique_ptr<graphics::Buffer> instance_buffer;
    std::vector<std::pair<DeviceModel *, InstanceInfo>> instances;
  };

  void OnInit();

  void OnUpdate();

  void OnRender();

  void OnClose();

  void BuildScreenFrameObjects();

  void UpdateTitle();

  void SaveScreenshot();

  template <class Func, class... Args>
  void NotifyListeners(Func func, Args... args) {
    // Listeners may register or unregister themselves while handling events.
    auto listeners = listeners_;
    for (auto listener : listeners) {
      if (listeners_.count(listener)) {
        (listener->*func)(args...);
      }
    }
  }

  void RenderFrameTarget(graphics::CommandContext *context, FrameTarget &target);

  std::string name_;
  std::string screenshot_path_;
  std::unique_ptr<graphics::Core> core_;
  std::unique_ptr<graphics::Window> window_;

  std::unique_ptr<graphics::Shader> vertex_shader_;
  std::unique_ptr<graphics::Shader> pixel_shader_;
  std::unique_ptr<graphics::Program> program_;
  std::unique_ptr<graphics::Shader> resolve_vertex_shader_;
  std::unique_ptr<graphics::Shader> resolve_pixel_shader_;
  std::unique_ptr<graphics::Program> resolve_program_;

  std::unique_ptr<graphics::Buffer> global_uniform_buffer_;
  std::unique_ptr<graphics::Buffer> resolve_uniform_buffer_;
  std::unique_ptr<graphics::Buffer> instance_index_buffer_;

  std::unique_ptr<graphics::Image> depth_image_;
  std::unique_ptr<graphics::Image> present_image_;
  FrameTarget main_frame_;
  FrameTarget second_frame_;
  float second_frame_alpha_{0.0f};

  glm::ivec2 framebuffer_size_{0};
  int supersample_scale_{1};

  std::set<Listener *> listeners_{};

  uint32_t mouse_move_callback_{};
  uint32_t mouse_button_callback_{};
  uint32_t cursor_enter_callback_{};
  uint32_t focus_callback_{};

  int fps_frames_{0};
  double fps_start_time_{0.0};
};
