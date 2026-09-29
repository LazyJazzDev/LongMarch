#include "Host.h"

#include <hilog/log.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rawfile/raw_file.h>

#include <cmath>
#include <fstream>
#include <future>

#include "grassland/graphics/sha256.h"

namespace longmarch::harmony {
namespace {
std::string ReadFile(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot read " + path.string());
  return {std::istreambuf_iterator<char>(file), {}};
}

std::string Text(const rapidjson::Value &v, const char *key, const char *fallback = "") {
  return v.IsObject() && v.HasMember(key) && v[key].IsString() ? v[key].GetString() : fallback;
}

double Number(const rapidjson::Value &v, const char *key, double fallback = 0) {
  double result = v.IsObject() && v.HasMember(key) && v[key].IsNumber() ? v[key].GetDouble() : fallback;
  return std::isfinite(result) ? result : fallback;
}

bool Boolean(const rapidjson::Value &v, const char *key, bool fallback = false) {
  return v.IsObject() && v.HasMember(key) && v[key].IsBool() ? v[key].GetBool() : fallback;
}

std::string Raw(NativeResourceManager *manager, const std::string &name) {
  auto *file = OH_ResourceManager_OpenRawFile(manager, name.c_str());
  if (!file)
    throw std::runtime_error("Missing bundled resource " + name);
  std::unique_ptr<RawFile, decltype(&OH_ResourceManager_CloseRawFile)> owner(file, OH_ResourceManager_CloseRawFile);
  const long length = OH_ResourceManager_GetRawFileSize(file);
  if (length < 0 || length > 256 * 1024 * 1024)
    throw std::runtime_error("Invalid resource size " + name);
  std::string data(static_cast<size_t>(length), '\0');
  size_t offset = 0;
  while (offset < data.size()) {
    int count =
        OH_ResourceManager_ReadRawFile(file, data.data() + offset, std::min(size_t(1 << 20), data.size() - offset));
    if (count <= 0)
      throw std::runtime_error("Cannot read bundled resource " + name);
    offset += count;
  }
  return data;
}

std::string Digest(const std::string &data) {
  grassland::graphics::detail::SHA256 hash;
  hash.Update(data);
  return hash.Finish();
}
}  // namespace

Host &Host::Get() {
  static Host host;
  return host;
}

Host::Host() {
  thread_ = std::thread([this] { Run(); });
}

Host::~Host() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    quit_ = true;
  }
  changed_.notify_one();
  thread_.join();
}

void Host::Post(std::function<void()> work) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    commands_.push_back(std::move(work));
  }
  changed_.notify_one();
}

void Host::Initialize(NativeResourceManager *manager, std::string directory) {
  Post([this, manager, directory] {
    std::unique_ptr<NativeResourceManager, decltype(&OH_ResourceManager_ReleaseNativeResourceManager)> owner(
        manager, OH_ResourceManager_ReleaseNativeResourceManager);
    Extract(manager, directory);
  });
}

void Host::Extract(NativeResourceManager *manager, const std::string &directory) {
  if (!manager)
    throw std::runtime_error("Resource manager is unavailable");
  auto manifest = Raw(manager, "Resources/manifest.json");
  rapidjson::Document document;
  document.Parse(manifest.c_str());
  if (document.HasParseError() || !document.IsObject() || !document.HasMember("files") || !document["files"].IsArray())
    throw std::runtime_error("Invalid resource manifest");
  auto root = std::filesystem::path(directory) / ("resources-" + Digest(manifest).substr(0, 20));
  if (!std::filesystem::exists(root / ".complete")) {
    for (auto &entry : document["files"].GetArray()) {
      auto path = std::filesystem::path(Text(entry, "path"));
      if (path.empty() || path.is_absolute())
        throw std::runtime_error("Invalid bundled path");
      for (const auto &part : path)
        if (part == "..")
          throw std::runtime_error("Invalid bundled path");
      auto data = Raw(manager, "Resources/" + path.generic_string());
      if (Digest(data) != Text(entry, "sha256"))
        throw std::runtime_error("Resource digest mismatch: " + path.string());
      std::filesystem::create_directories((root / path).parent_path());
      std::ofstream file(root / path, std::ios::binary | std::ios::trunc);
      file.write(data.data(), data.size());
      file.close();
      if (!file)
        throw std::runtime_error("Cannot extract " + path.string());
    }
    std::ofstream complete(root / ".complete");
    complete << Digest(manifest);
    complete.close();
    if (!complete)
      throw std::runtime_error("Cannot finish resource extraction");
  }
  catalog_ = ReadFile(root / "catalog.json");
  rapidjson::Document catalog;
  catalog.Parse(catalog_.c_str());
  if (catalog.HasParseError() || !catalog.IsArray() || catalog.Empty())
    throw std::runtime_error("Invalid scene catalog");
  resources_ = root;
  ready_ = true;
  error_.clear();
}

void Host::Command(std::string json) {
  Post([this, json = std::move(json)] { Execute(json); });
}

void Host::Attach(OHNativeWindow *window, uint32_t width, uint32_t height) {
  OH_NativeWindow_NativeObjectReference(window);
  Post([this, window, width, height] {
    surface_.reset();
    if (window_)
      OH_NativeWindow_NativeObjectUnreference(window_);
    window_ = window;
    width_ = width;
    height_ = height;
    Resize();
    dirty_ = true;
  });
}

void Host::Detach() {
  auto done = std::make_shared<std::promise<void>>();
  auto wait = done->get_future();
  Post([this, done] {
    surface_.reset();
    if (window_)
      OH_NativeWindow_NativeObjectUnreference(window_);
    window_ = nullptr;
    done->set_value();
  });
  wait.get();
}

void Host::Stop() {
  surface_.reset();
  demo_.reset();
  scene_.reset();
  selection_.clear();
  scene_id_.clear();
  dirty_ = false;
  frames_ = 0;
  size_axis_ = 0;
  fps_ = elapsed_ = 0;
  last_present_ = {};
}

void Host::Resize() {
  if (demo_ && width_ && height_) {
    if (Game()) {
      demo_->Resize(width_, height_);
      Game()->SetBottomControlInset(bottom_inset_);
    } else if (selection_ == "nbody_cs" || selection_ == "graphics_hello_resize") {
      demo_->Resize(std::max(1, int(width_ * render_scale_)), std::max(1, int(height_ * render_scale_)));
    }
  }
}

void Host::Execute(const std::string &json) {
  rapidjson::Document c;
  c.Parse(json.c_str());
  if (c.HasParseError() || !c.IsObject())
    throw std::runtime_error("Invalid native command");
  const auto type = Text(c, "type");
  if (type == "active") {
    active_ = Boolean(c, "value");
    last_present_ = {};
    if (Game()) {
      Game()->Window()->SendFocus(active_);
      Game()->ResetClock();
    }
    dirty_ = true;
  } else if (type == "stop") {
    Stop();
    error_.clear();
  } else if (type == "open") {
    Stop();
    error_.clear();
    if (!ready_)
      throw std::runtime_error("Resources are still being prepared");
    selection_ = Text(c, "demo");
    scene_id_ = Text(c, "scene", "cornell_box");
    zoom_ = 1;
    pan_x_ = pan_y_ = 0;
    paused_ = false;
    if (selection_ == "sparkium") {
      rapidjson::Document catalog;
      catalog.Parse(catalog_.c_str());
      bool found = false;
      if (catalog.IsArray())
        for (auto &item : catalog.GetArray())
          if (Text(item, "id") == scene_id_)
            found = true;
      if (!found)
        throw std::runtime_error("Unknown scene");
      scene_ = std::make_unique<RenderSession>(resources_, scene_id_, 0, false, 0,
                                               grassland::graphics::BACKEND_API_VULKAN, true);
    } else {
      demo_ = std::make_unique<DemoSession>(resources_, selection_, false, grassland::graphics::BACKEND_API_VULKAN);
      if (Game())
        Game()->EnableNativeSizeControls();
      Resize();
    }
    dirty_ = true;
  } else if (type == "display") {
    hdr_ = Boolean(c, "hdr", true);
    exposure_ = std::clamp(Number(c, "exposure"), -5.0, 5.0);
    dirty_ = true;
  } else if (type == "limit") {
    limit_ = int(std::clamp(Number(c, "value", 32), 1.0, 4096.0));
    dirty_ = true;
  } else if (type == "pause") {
    paused_ = Boolean(c, "value");
    dirty_ = true;
  } else if (type == "resetFilm" && scene_) {
    scene_->ResetFilm();
    elapsed_ = 0;
    dirty_ = true;
  } else if (type == "nbody") {
    particles_ = int(std::clamp(Number(c, "particles", 4096), 128.0, 65536.0));
    galaxies_ = int(std::clamp(Number(c, "galaxies", 10), 1.0, 20.0));
    step_ = std::clamp(Number(c, "step", .03), .0001, .1);
    render_scale_ = std::clamp(Number(c, "scale", 1), .25, 1.0);
    if (Boolean(c, "reset"))
      ++resets_;
    Resize();
    dirty_ = true;
  } else if (type == "view") {
    zoom_ = std::clamp(Number(c, "zoom", 1), 1.0, 16.0);
    pan_x_ = Number(c, "x");
    pan_y_ = Number(c, "y");
    dirty_ = true;
  } else if (type == "rotate") {
    yaw_ = Number(c, "yaw");
    pitch_ = std::clamp(Number(c, "pitch"), -1.5, 1.5);
    dirty_ = true;
  } else if (type == "size" && Game()) {
    Game()->SetGridDimension(int(Number(c, "axis")), int(std::clamp(Number(c, "value", 64), 2.0, 256.0)));
    size_value_ = int(std::clamp(Number(c, "value", 64), 2.0, 256.0));
    dirty_ = true;
  } else if (type == "bottomInset") {
    bottom_inset_ = float(std::clamp(Number(c, "value"), 0.0, 0.25));
    Resize();
    dirty_ = true;
  } else if (type == "dismissSize") {
    size_axis_ = 0;
  } else if (type == "file" && Game()) {
    file_revision_ = int(Number(c, "request"));
    auto error = Game()->CompleteFile(Text(c, "path"));
    if (!error.empty())
      throw std::runtime_error(error);
    dirty_ = true;
  } else if (type == "icons" && Game()) {
    Game()->SetIconOrientation(Number(c, "angle"));
    dirty_ = true;
  } else if (type == "input" && Game()) {
    auto *game = Game();
    game->PrepareInput(delay_ < 0);
    auto *window = game->Window();
    const int kind = int(Number(c, "kind")), value = int(Number(c, "value"));
    double x = Number(c, "x") * width_, y = Number(c, "y") * height_;
    window->SendPointer(x, y);
    if (kind == 1) {
      window->CursorEnterEvent().InvokeCallbacks(true);
      window->SendMouseButton(value, 1);
    }
    if (kind == 2) {
      window->SendMouseButton(value, 0);
      window->CursorEnterEvent().InvokeCallbacks(false);
    }
    if (kind == 5) {
      window->SendFocus(value != 0);
      if (value)
        game->ResetClock();
    }
    if (kind == 4) {
      window->SendKey(value, 1);
      window->SendKey(value, 0);
    }
    if (kind == 6)
      window->MagnifyEvent().InvokeCallbacks(grassland::graphics::MagnifyGesture{
          std::clamp(Number(c, "value", 1), .1, 10.0), x, y, grassland::graphics::MagnifyPhase::kUpdate});
    dirty_ = true;
  }
}

void Host::Frame() {
  auto start = std::chrono::steady_clock::now();
  auto *core = scene_ ? scene_->Graphics() : demo_->Core();
  if (!surface_)
    surface_ = std::make_unique<Surface>(core, window_);
  // Match iOS: display-encoded game/UI colors belong on an SDR surface.
  const bool hdr_content = scene_ || selection_ == "graphics_hello_hdr" || selection_ == "nbody_cs";
  surface_->Resize(width_, height_, hdr_ && hdr_content);
  grassland::graphics::Image *image;
  bool sampled = false;
  if (scene_) {
    if (!paused_ && scene_->Samples() < limit_) {
      scene_->Render();
      sampled = true;
    }
    image = scene_->Develop(surface_->HDR(), exposure_);
  } else {
    demo_->Configure(particles_, galaxies_, step_, !paused_, yaw_, pitch_, resets_);
    demo_->Render();
    image = demo_->Image();
    if (Game()) {
      auto size = Game()->TakeSizeControlRequest();
      if (size.x) {
        size_axis_ = size.x;
        size_value_ = size.y;
      }
    }
  }
  const double duration = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (!scene_ || sampled) {
    elapsed_ += duration;
  }
  const bool presented = surface_->Present(image, scene_ ? zoom_ : 1, scene_ ? pan_x_ : 0, scene_ ? pan_y_ : 0,
                                           selection_ == "graphics_hello_hdr", selection_ == "nbody_cs");
  const auto presented_at = std::chrono::steady_clock::now();
  if (presented) {
    const double interval =
        std::chrono::duration<double>(presented_at - (last_present_.time_since_epoch().count() ? last_present_ : start))
            .count();
    fps_ = interval > 0 ? 1 / interval : 0;
    last_present_ = presented_at;
  }
  ++frames_;
  if (scene_ && (frames_ == 1 || (sampled && scene_->Samples() >= limit_)))
    OH_LOG_Print(LOG_APP, LOG_INFO, 0, "LongMarchGPU",
                 "Scene %{public}s frame=%{public}llu spp=%{public}u duration=%{public}.3f presented=%{public}d",
                 scene_id_.c_str(), static_cast<unsigned long long>(frames_), static_cast<unsigned>(scene_->Samples()),
                 duration, presented);
  dirty_ = false;
  delay_ = Game() ? Game()->NextFrameDelay() : 1.0 / 60;
  // Shared games use infinity, not a negative number, to report idle.
  if (!std::isfinite(delay_) || delay_ < 0)
    delay_ = -1;
  if (scene_ && (paused_ || scene_->Samples() >= limit_))
    delay_ = -1;
  if (demo_ && (selection_ == "nbody_cs" || selection_ == "graphics_hello_resize") && paused_)
    delay_ = -1;
  // A replaced swapchain needs another presentation even when sampling is paused.
  if (!presented)
    delay_ = 1.0 / 60;
  deadline_ = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                          std::chrono::duration<double>(std::max(1.0 / 60, delay_)));
}

void Host::Publish() {
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> w(buffer);
  w.StartObject();
  auto text = [&](const char *key, const std::string &value) {
    w.Key(key);
    w.String(value.c_str());
  };
  auto number = [&](const char *key, double value) {
    w.Key(key);
    w.Double(std::isfinite(value) ? value : 0);
  };
  w.Key("ready");
  w.Bool(ready_);
  w.Key("hdr");
  w.Bool(surface_ && surface_->HDR());
  text("error", error_);
  text("demo", selection_);
  text("scene", scene_id_);
  text("resources", resources_.string());
  w.Key("scenes");
  w.RawValue(catalog_.c_str(), catalog_.size(), rapidjson::kArrayType);
  number("fps", fps_);
  number("frames", frames_);
  number("seconds", elapsed_);
  number("spp", scene_ ? scene_->Samples() : 0);
  number("width", scene_ ? scene_->Width() : demo_ ? demo_->Image()->Extent().width : 0);
  number("height", scene_ ? scene_->Height() : demo_ ? demo_->Image()->Extent().height : 0);
  number("bounces", scene_ ? scene_->MaxBounces() : 0);
  number("gpuMs", demo_ ? demo_->GPUMilliseconds() : 0);
  number("fileRequest", Game() ? Game()->FileRequest() : 0);
  number("fileRevision", file_revision_);
  number("sizeAxis", size_axis_);
  number("sizeValue", size_value_);
  text("device", scene_ ? scene_->Device() : demo_ ? demo_->Core()->DeviceName() : "");
  text("pipeline", scene_ ? (scene_->ComputeFallback() ? "Compute fallback" : "Ray Query") : "Raster / Compute");
  w.EndObject();
  std::lock_guard<std::mutex> lock(status_mutex_);
  status_ = buffer.GetString();
}

std::string Host::Status() {
  std::lock_guard<std::mutex> lock(status_mutex_);
  return status_;
}

void Host::Run() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (!quit_) {
    bool renderable = active_ && window_ && width_ && height_ && (demo_ || scene_) && error_.empty();
    if (commands_.empty() &&
        !(renderable && (dirty_ || (delay_ >= 0 && std::chrono::steady_clock::now() >= deadline_)))) {
      if (renderable && delay_ >= 0)
        changed_.wait_until(lock, deadline_);
      else
        changed_.wait(lock);
      continue;
    }
    std::function<void()> command;
    if (!commands_.empty()) {
      command = std::move(commands_.front());
      commands_.pop_front();
    }
    lock.unlock();
    try {
      if (command)
        command();
      else
        Frame();
    } catch (const std::exception &error) {
      error_ = error.what();
      dirty_ = false;
    }
    Publish();
    lock.lock();
  }
  lock.unlock();
  Stop();
  if (window_)
    OH_NativeWindow_NativeObjectUnreference(window_);
}
}  // namespace longmarch::harmony
