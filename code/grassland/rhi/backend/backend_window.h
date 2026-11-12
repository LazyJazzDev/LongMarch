#pragma once
#include <mutex>

#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/backend/backend_util.h"
#include "grassland/rhi/rhi_image.h"
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
  virtual ImGuiContext *GetImGuiContext() const = 0;
};

}  // namespace grassland::rhi::backend
