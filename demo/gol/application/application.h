#pragma once

#include <array>
#include <glm/gtc/matrix_transform.hpp>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "../../demo_window.h"
#include "grassland/util/util.h"

using namespace grassland;

class DeviceModel;

class Listener;

struct InstanceInfo {
  glm::mat4 model;
  glm::vec4 color;
  glm::uvec4 extra;
  glm::vec4 clip_rect{-1.0e9f, -1.0e9f, 1.0e9f, 1.0e9f};
};

// A small 2D instanced renderer on top of grassland::rhi. Models are drawn
// in framebuffer pixel coordinates (origin at the top-left corner) with depth
// testing, then resolved from a supersampled target for anti-aliasing.
class Application {
 public:
  Application(const std::string &name, int width, int height, rhi::BackendAPI api);

  virtual ~Application();

  // Runs until the window closes, or after max_frames frames when positive.
  bool Run(int max_frames = 0);

  // Saves the last presented frame as a PNG file when the application exits.
  void SetScreenshotPath(const std::string &path) {
    screenshot_path_ = path;
  }

  rhi::ObjPtr<rhi::IBuffer> CreateBuffer(uint64_t size);
  bool UploadBuffer(rhi::ObjPtr<rhi::IBuffer> &buffer, const void *data, uint64_t size);

  [[nodiscard]] demo::Window *GetWindow() const {
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

  // Draws one instance together with per-instance data words that its shader
  // reads from the instance buffer. extra.y receives the data's byte offset.
  void DrawModelWithData(DeviceModel *device_model, InstanceInfo instance_info, const std::vector<uint32_t> &data);

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
    rhi::ObjPtr<rhi::IImage> color_image;
    rhi::ObjPtr<rhi::IBuffer> instance_buffer;
    std::vector<std::pair<DeviceModel *, InstanceInfo>> instances;
    // Data words stored after the instance records, and the instances whose
    // extra.y holds an offset into them, relative to the end of the records.
    std::vector<uint32_t> data;
    std::vector<size_t> data_instances;
    std::vector<uint8_t> upload;
  };

  bool Check(bool success, const char *operation);
  bool WaitFrames();
  bool OnInit();

  void OnUpdate();

  bool OnRender();

  void OnClose();

  bool BuildScreenFrameObjects();

  void UpdateTitle();

  bool SaveScreenshot();

  template <class Func, class... Args>
  void NotifyListeners(Func func, Args... args) {
    // Listeners may register or unregister themselves while handling events.
    const auto revision = listener_revision_;
    const std::vector<Listener *> listeners(listeners_.begin(), listeners_.end());
    for (auto listener : listeners) {
      if (revision == listener_revision_ || listeners_.count(listener)) {
        (listener->*func)(args...);
      }
    }
  }

  bool RenderFrameTarget(rhi::ICommandContext *context, FrameTarget &target);

  std::string name_;
  std::string screenshot_path_;
  rhi::BackendAPI api_;
  int initial_width_, initial_height_;
  bool failed_{false};
  bool custom_initialized_{false};
  rhi::ObjPtr<rhi::ICore> core_;
  rhi::ObjPtr<rhi::IDevice> device_;
  rhi::ObjPtr<rhi::IQueue> queue_;
  std::unique_ptr<demo::Window> window_;
  rhi::ObjPtr<rhi::ICommandContext> upload_context_;
  std::array<rhi::ObjPtr<rhi::ISubmissionToken>, 2> submissions_;
  size_t submission_index_{};
  rhi::ObjPtr<rhi::IBindingGroup> instance_group_, global_group_;
  rhi::ObjPtr<rhi::IBindingGroup> resolve_group_, main_image_group_, second_image_group_;

  rhi::ObjPtr<rhi::IShader> vertex_shader_;
  rhi::ObjPtr<rhi::IShader> pixel_shader_;
  rhi::ObjPtr<rhi::IProgram> program_;
  rhi::ObjPtr<rhi::IShader> resolve_vertex_shader_;
  rhi::ObjPtr<rhi::IShader> resolve_pixel_shader_;
  rhi::ObjPtr<rhi::IProgram> resolve_program_;

  rhi::ObjPtr<rhi::IBuffer> global_uniform_buffer_;
  rhi::ObjPtr<rhi::IBuffer> resolve_uniform_buffer_;
  rhi::ObjPtr<rhi::IBuffer> instance_index_buffer_;
  size_t instance_index_count_{};

  rhi::ObjPtr<rhi::IImage> depth_image_;
  rhi::ObjPtr<rhi::IImage> present_image_;
  FrameTarget main_frame_;
  FrameTarget second_frame_;
  float second_frame_alpha_{0.0f};

  glm::ivec2 framebuffer_size_{0};
  int supersample_scale_{1};

  std::set<Listener *> listeners_{};
  uint64_t listener_revision_{};

  uint32_t mouse_move_callback_{};
  uint32_t mouse_button_callback_{};
  uint32_t cursor_enter_callback_{};
  uint32_t focus_callback_{};

  grassland::FPSCounter fps_counter_;
  double title_time_{0.0};
};
