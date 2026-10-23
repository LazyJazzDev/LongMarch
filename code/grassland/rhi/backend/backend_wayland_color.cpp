#include "grassland/rhi/backend/backend_wayland_color.h"

#ifdef LONGMARCH_RHI_WAYLAND_COLOR
#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <unistd.h>
#include <wayland-client.h>

#include <cstring>

#include "color-management-v1-client-protocol.h"
#endif

namespace grassland::rhi::backend {

struct WaylandColor::State {
#ifdef LONGMARCH_RHI_WAYLAND_COLOR
  wl_registry *registry{};
  wl_surface *surface{};
  wp_color_manager_v1 *manager{};
  wp_color_management_surface_feedback_v1 *feedback{};
  wp_image_description_v1 *description{};
  wp_image_description_info_v1 *info{};
  uint32_t manager_id{};
  uint32_t pending_white{};
  uint32_t white{};

  void ClearDescription() {
    white = pending_white = 0;
    if (info) {
      wp_image_description_info_v1_destroy(info);
      info = nullptr;
    }
    if (description) {
      wp_image_description_v1_destroy(description);
      description = nullptr;
    }
  }

  void ClearManager() {
    ClearDescription();
    if (feedback) {
      wp_color_management_surface_feedback_v1_destroy(feedback);
      feedback = nullptr;
    }
    if (manager) {
      wp_color_manager_v1_destroy(manager);
      manager = nullptr;
    }
  }

  void Refresh() {
    ClearDescription();
    description = wp_color_management_surface_feedback_v1_get_preferred(feedback);
    static const wp_image_description_v1_listener listener{
        [](void *data, wp_image_description_v1 *, uint32_t, const char *) {
          static_cast<State *>(data)->ClearDescription();
        },
        [](void *data, wp_image_description_v1 *description, uint32_t) {
          auto &s = *static_cast<State *>(data);
          s.info = wp_image_description_v1_get_information(description);
          static const auto listener = [] {
            wp_image_description_info_v1_listener l{};
            l.done = [](void *data, wp_image_description_info_v1 *info) {
              auto &s = *static_cast<State *>(data);
              s.white = s.pending_white;
              wp_image_description_info_v1_destroy(info);
              s.info = nullptr;
            };
            l.icc_file = [](void *, wp_image_description_info_v1 *, int32_t fd, uint32_t) { close(fd); };
            l.primaries = [](void *, wp_image_description_info_v1 *, int32_t, int32_t, int32_t, int32_t, int32_t,
                             int32_t, int32_t, int32_t) {};
            l.primaries_named = [](void *, wp_image_description_info_v1 *, uint32_t) {};
            l.tf_power = l.primaries_named;
            l.tf_named = l.primaries_named;
            l.luminances = [](void *data, wp_image_description_info_v1 *, uint32_t, uint32_t, uint32_t reference) {
              static_cast<State *>(data)->pending_white = reference;
            };
            l.target_primaries = l.primaries;
            // Neither the primary-volume maximum nor target luminance proves physical display peak.
            l.target_luminance = [](void *, wp_image_description_info_v1 *, uint32_t, uint32_t) {};
            l.target_max_cll = l.primaries_named;
            l.target_max_fall = l.primaries_named;
            return l;
          }();
          wp_image_description_info_v1_add_listener(s.info, &listener, &s);
        }};
    wp_image_description_v1_add_listener(description, &listener, this);
  }

  explicit State(GLFWwindow *window) {
    surface = glfwGetWaylandWindow(window);
    registry = wl_display_get_registry(glfwGetWaylandDisplay());
    static const wl_registry_listener listener{
        [](void *data, wl_registry *registry, uint32_t id, const char *interface, uint32_t) {
          auto &s = *static_cast<State *>(data);
          if (s.manager || std::strcmp(interface, wp_color_manager_v1_interface.name))
            return;
          s.manager_id = id;
          s.manager =
              static_cast<wp_color_manager_v1 *>(wl_registry_bind(registry, id, &wp_color_manager_v1_interface, 1));
          static const wp_color_manager_v1_listener manager_listener{
              [](void *, wp_color_manager_v1 *, uint32_t) {}, [](void *, wp_color_manager_v1 *, uint32_t) {},
              [](void *, wp_color_manager_v1 *, uint32_t) {}, [](void *, wp_color_manager_v1 *, uint32_t) {},
              [](void *, wp_color_manager_v1 *) {}};
          wp_color_manager_v1_add_listener(s.manager, &manager_listener, nullptr);
          s.feedback = wp_color_manager_v1_get_surface_feedback(s.manager, s.surface);
          static const auto feedback_listener = [] {
            wp_color_management_surface_feedback_v1_listener l{};
            l.preferred_changed = [](void *data, wp_color_management_surface_feedback_v1 *, uint32_t) {
              static_cast<State *>(data)->Refresh();
            };
            return l;
          }();
          wp_color_management_surface_feedback_v1_add_listener(s.feedback, &feedback_listener, &s);
          s.Refresh();
        },
        [](void *data, wl_registry *, uint32_t id) {
          auto &s = *static_cast<State *>(data);
          if (id == s.manager_id)
            s.ClearManager();
        }};
    wl_registry_add_listener(registry, &listener, this);
  }

  ~State() {
    ClearManager();
    wl_registry_destroy(registry);
  }
#endif
};

WaylandColor::WaylandColor(GLFWwindow *window) {
#ifdef LONGMARCH_RHI_WAYLAND_COLOR
  if (glfwGetPlatform() == GLFW_PLATFORM_WAYLAND)
    state_ = std::make_unique<State>(window);
#endif
}

WaylandColor::~WaylandColor() = default;

bool WaylandColor::ReferenceWhite(float &nits) const {
#ifdef LONGMARCH_RHI_WAYLAND_COLOR
  if (state_ && state_->white > 0) {
    nits = float(state_->white);
    return true;
  }
#endif
  return false;
}

}  // namespace grassland::rhi::backend
