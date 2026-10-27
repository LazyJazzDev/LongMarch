#pragma once
#include <memory>
#include <string>
#include <string_view>

#include "grassland/rhi/rhi.h"

struct ModuleOptions {
  int frames = 0;
  bool headless = false;
  std::string capture_directory;
  const char *font_file = nullptr;
  float font_size = 20.0f;
};

class ModuleBase {
 public:
  ModuleBase(grassland::rhi::ObjPtr<grassland::rhi::ICore> core,
             grassland::rhi::ObjPtr<grassland::rhi::IDevice> device);
  virtual ~ModuleBase() = default;
  ModuleBase(const ModuleBase &) = delete;
  ModuleBase &operator=(const ModuleBase &) = delete;

  int Run(const ModuleOptions &options);
  bool ShouldClose() const;

 protected:
  void RequestClose();
  int GetFrameIndex() const;
  void CompleteFrame();
  void SetWarmupFrames(int frames);

  grassland::rhi::ICore *GetCore() const;
  grassland::rhi::IDevice *GetDevice() const;
  grassland::rhi::BackendAPI GetBackendAPI() const;

 private:
  virtual int OnInit(const ModuleOptions &options) = 0;
  virtual int OnUpdate() = 0;
  virtual int OnRender() = 0;
  virtual int OnTerminate() = 0;

  bool should_close_ = false;
  int frame_index_ = 0;

  // Members are destroyed in reverse order: Device must be released before Core.
  grassland::rhi::ObjPtr<grassland::rhi::ICore> core_;
  grassland::rhi::ObjPtr<grassland::rhi::IDevice> device_;
};

std::unique_ptr<ModuleBase> CreateModule(std::string_view name,
                                         const grassland::rhi::ObjPtr<grassland::rhi::ICore> &core,
                                         const grassland::rhi::ObjPtr<grassland::rhi::IDevice> &device);
