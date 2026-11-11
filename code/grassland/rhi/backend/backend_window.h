#pragma once
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_image.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_window.h"

struct ImGuiContext;

LONGMARCH_RHI_INTERFACE(grassland::rhi::backend::IBackendWindow,
                        grassland::rhi::IObject,
                        "093b830a-4e7b-4d76-8df4-9a2693735c9d")

namespace grassland::rhi::backend {
#ifdef __APPLE__
float CocoaHDRHeadroom(GLFWwindow *window);
#endif

class WindowImGui;
class ImagePresenter;

class IBackendWindow : public IObject {
 public:
  virtual IBackendQueue *OwnerQueue() const = 0;
  virtual ImGuiContext *GetImGuiContext() const = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands, rhi::IImage *image) = 0;
  virtual bool RecordImGui(rhi::ICommandContext *commands,
                           rhi::IImage *image,
                           ColorSpace color_space,
                           float white_scale) = 0;
  virtual bool RecordPresentImage(ICommandContext *commands, IImage *image, ColorSpace color_space) = 0;
  virtual const void *DeviceIdentity() const = 0;
  virtual bool UpdateNextImage() = 0;
  virtual void ReleaseNextImage() = 0;
  virtual ObjectPtr<IBackendImage> AcquireNativeFrame() = 0;
  virtual bool RebuildNativeSwapChain(PresentMode mode) = 0;
};

}  // namespace grassland::rhi::backend
