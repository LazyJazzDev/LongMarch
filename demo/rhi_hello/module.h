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
  ModuleBase() = default;
  virtual ~ModuleBase() = default;
  ModuleBase(const ModuleBase &) = delete;
  ModuleBase &operator=(const ModuleBase &) = delete;

  int Run(grassland::rhi::BackendAPI api, int32_t device_index, const ModuleOptions &options);

 protected:
  grassland::rhi::ICore *GetCore() const;
  grassland::rhi::IDevice *GetDevice() const;
  grassland::rhi::BackendAPI GetBackendAPI() const;

 private:
  virtual int RunImpl(const ModuleOptions &options) = 0;

  grassland::rhi::BackendAPI api_ = grassland::rhi::BACKEND_API_DEFAULT;
  // Members are destroyed in reverse order: Device must be released before Core.
  grassland::rhi::ObjPtr<grassland::rhi::ICore> core_;
  grassland::rhi::ObjPtr<grassland::rhi::IDevice> device_;
};

class WindowModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class TriangleModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class CubeModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class TextureModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class BlendModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class NBodyModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class RayTracingModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class RayQueryModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class HDRModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class ImGuiModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

class ImGuiMultiWindowModule final : public ModuleBase {
 private:
  int RunImpl(const ModuleOptions &options) override;
};

std::unique_ptr<ModuleBase> CreateModule(std::string_view name);
