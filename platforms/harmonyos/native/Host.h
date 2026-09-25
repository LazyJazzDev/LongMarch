#pragma once
#include <rawfile/raw_file_manager.h>

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

#include "RenderSession.h"
#include "Surface.h"
#include "demos/DemoSession.h"

namespace longmarch::harmony {
class Host {
 public:
  static Host &Get();
  void Initialize(NativeResourceManager *manager, std::string files_dir);
  void Command(std::string json);
  void Attach(OHNativeWindow *window, uint32_t width, uint32_t height);
  void Detach();
  std::string Status();
  ~Host();

 private:
  Host();
  void Post(std::function<void()> work);
  void Run();
  void Execute(const std::string &json);
  void Frame();
  void Publish();
  void Stop();
  void Resize();
  void Extract(NativeResourceManager *manager, const std::string &files_dir);

  DesktopGameSession *Game() {
    return demo_ ? demo_->Game() : nullptr;
  }

  std::mutex mutex_, status_mutex_;
  std::condition_variable changed_;
  std::deque<std::function<void()>> commands_;
  std::thread thread_;
  bool quit_ = false;
  std::string status_ = "{}";
  std::unique_ptr<DemoSession> demo_;
  std::unique_ptr<RenderSession> scene_;
  std::unique_ptr<Surface> surface_;
  OHNativeWindow *window_ = nullptr;
  uint32_t width_ = 0, height_ = 0;
  std::filesystem::path resources_;
  std::string catalog_ = "[]", selection_, scene_id_, error_;
  bool ready_ = false, active_ = true, dirty_ = false, paused_ = false, hdr_ = true;
  int limit_ = 32, particles_ = 4096, galaxies_ = 10, resets_ = 0;
  int size_axis_ = 0, size_value_ = 64, file_revision_ = 0;
  uint64_t frames_ = 0;
  float step_ = .03f, yaw_ = 0, pitch_ = 0, exposure_ = 0, render_scale_ = 1;
  double zoom_ = 1, pan_x_ = 0, pan_y_ = 0, fps_ = 0, elapsed_ = 0;
  double delay_ = 0;
  std::chrono::steady_clock::time_point deadline_{}, last_present_{};
};
}  // namespace longmarch::harmony
