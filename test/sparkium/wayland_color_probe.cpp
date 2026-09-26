// Observe compositor-provided reference white without changing system settings
// or setting a color description on a surface owned by the Vulkan WSI driver.
#include <long_march.h>

#define GLFW_EXPOSE_NATIVE_WAYLAND
#include <GLFW/glfw3native.h>
#include <unistd.h>
#include <wayland-client.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#include "color-management-v1-client-protocol.h"

using namespace grassland;

namespace {
struct Description {
  bool complete{};
  bool ready{};
  bool has_luminances{};
  uint32_t reference{}, maximum{}, minimum{}, transfer{}, primaries{};
};

const wp_image_description_info_v1_listener &InfoListener() {
  static const auto listener = [] {
    wp_image_description_info_v1_listener l{};
    l.done = [](void *data, wp_image_description_info_v1 *info) {
      static_cast<Description *>(data)->complete = true;
      wp_image_description_info_v1_destroy(info);
    };
    l.icc_file = [](void *, wp_image_description_info_v1 *, int32_t fd, uint32_t) { close(fd); };
    l.primaries = [](void *, wp_image_description_info_v1 *, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t,
                     int32_t, int32_t) {};
    l.primaries_named = [](void *data, wp_image_description_info_v1 *, uint32_t value) {
      static_cast<Description *>(data)->primaries = value;
    };
    l.tf_power = [](void *, wp_image_description_info_v1 *, uint32_t) {};
    l.tf_named = [](void *data, wp_image_description_info_v1 *, uint32_t value) {
      static_cast<Description *>(data)->transfer = value;
    };
    l.luminances = [](void *data, wp_image_description_info_v1 *, uint32_t minimum, uint32_t maximum,
                      uint32_t reference) {
      auto &d = *static_cast<Description *>(data);
      d.has_luminances = true;
      d.minimum = minimum;
      d.maximum = maximum;
      d.reference = reference;
    };
    l.target_primaries = l.primaries;
    l.target_luminance = [](void *, wp_image_description_info_v1 *, uint32_t, uint32_t) {};
    l.target_max_cll = [](void *, wp_image_description_info_v1 *, uint32_t) {};
    l.target_max_fall = l.target_max_cll;
    return l;
  }();
  return listener;
}

const wp_image_description_v1_listener &DescriptionListener() {
  static const auto listener = [] {
    wp_image_description_v1_listener l{};
    l.failed = [](void *data, wp_image_description_v1 *, uint32_t cause, const char *message) {
      std::cout << "description unavailable: " << cause << " " << message << '\n';
      static_cast<Description *>(data)->complete = true;
    };
    l.ready = [](void *data, wp_image_description_v1 *description, uint32_t) {
      static_cast<Description *>(data)->ready = true;
      auto *info = wp_image_description_v1_get_information(description);
      wp_image_description_info_v1_add_listener(info, &InfoListener(), data);
    };
    return l;
  }();
  return listener;
}

struct Probe {
  wl_display *display{};
  wl_registry *registry{};
  wp_color_manager_v1 *manager{};
  wp_color_management_surface_feedback_v1 *feedback{};
  uint32_t advertised_version{};
  bool changed{true};

  explicit Probe(graphics::Window *window) {
    display = glfwGetWaylandDisplay();
    registry = wl_display_get_registry(display);
    static const wl_registry_listener listener{
        [](void *data, wl_registry *registry, uint32_t id, const char *interface, uint32_t version) {
          auto &p = *static_cast<Probe *>(data);
          if (std::strcmp(interface, wp_color_manager_v1_interface.name) != 0)
            return;
          p.advertised_version = version;
          // Version 1 suffices for luminances and preferred_changed. Binding
          // only v1 also permits building against the first protocol release.
          p.manager =
              static_cast<wp_color_manager_v1 *>(wl_registry_bind(registry, id, &wp_color_manager_v1_interface, 1));
          static const wp_color_manager_v1_listener manager_listener{
              [](void *, wp_color_manager_v1 *, uint32_t) {}, [](void *, wp_color_manager_v1 *, uint32_t) {},
              [](void *, wp_color_manager_v1 *, uint32_t) {}, [](void *, wp_color_manager_v1 *, uint32_t) {},
              [](void *, wp_color_manager_v1 *) {}};
          wp_color_manager_v1_add_listener(p.manager, &manager_listener, nullptr);
        },
        [](void *, wl_registry *, uint32_t) {}};
    wl_registry_add_listener(registry, &listener, this);
    Roundtrip();
    if (!manager)
      return;
    feedback = wp_color_manager_v1_get_surface_feedback(manager, glfwGetWaylandWindow(window->GLFWWindow()));
    static const auto feedback_listener = [] {
      wp_color_management_surface_feedback_v1_listener l{};
      l.preferred_changed = [](void *data, wp_color_management_surface_feedback_v1 *, uint32_t) {
        static_cast<Probe *>(data)->changed = true;
      };
      return l;
    }();
    wp_color_management_surface_feedback_v1_add_listener(feedback, &feedback_listener, this);
  }

  ~Probe() {
    if (feedback)
      wp_color_management_surface_feedback_v1_destroy(feedback);
    if (manager)
      wp_color_manager_v1_destroy(manager);
    wl_registry_destroy(registry);
  }

  void Roundtrip() {
    if (wl_display_roundtrip(display) < 0)
      throw std::runtime_error("Wayland connection failed");
  }

  void Report(const char *label) {
    if (!manager) {
      std::cout << label << ": color-management-v1 unavailable; reference white unknown\n";
      changed = false;
      return;
    }
    Description d;
    auto *description = wp_color_management_surface_feedback_v1_get_preferred(feedback);
    wp_image_description_v1_add_listener(description, &DescriptionListener(), &d);
    for (int i = 0; i < 4 && !d.complete; ++i)
      Roundtrip();
    // get_information is permitted on compositor-created preferred descriptions,
    // not necessarily on descriptions created by an application/WSI driver.
    wp_image_description_v1_destroy(description);
    if (!d.complete)
      throw std::runtime_error("No complete image-description reply");
    std::cout << label << ": protocol_version=" << advertised_version;
    if (d.ready && d.has_luminances) {
      std::cout << " preferred_reference_nits=" << d.reference << " preferred_max_nits=" << d.maximum
                << " preferred_min_nits=" << d.minimum / 10000.0 << " transfer=" << d.transfer
                << " primaries=" << d.primaries;
    } else {
      std::cout << " reference white unknown (no parametric luminance information)";
    }
    std::cout << std::endl;
    changed = false;
  }
};
}  // namespace

int main(int argc, char **argv) {
  try {
    int seconds = 2;
    if (argc == 3 && std::strcmp(argv[1], "--seconds") == 0)
      seconds = std::stoi(argv[2]);
    else if (argc != 1)
      throw std::invalid_argument("Usage: wayland_color_probe [--seconds N]");
    if (seconds < 1 || seconds > 3600)
      throw std::invalid_argument("seconds must be in [1, 3600]");
    std::unique_ptr<graphics::Core> core;
    if (graphics::CreateCore(graphics::BACKEND_API_VULKAN, {2, true}, &core) ||
        core->InitializeLogicalDeviceAutoSelect(false))
      throw std::runtime_error("Cannot initialize Vulkan");
    std::unique_ptr<graphics::Window> sdr, hdr;
    core->CreateWindowObject(480, 240, "SDR reference white", &sdr);
    if (glfwGetPlatform() != GLFW_PLATFORM_WAYLAND) {
      std::cout << "SKIP: native Wayland is required; X11 operation remains supported\n";
      return 77;
    }
    core->CreateWindowObject(480, 240, "HDR reference white (PQ/scRGB)", &hdr);
    if (hdr->SetHDR(true) != 0) {
      std::cout << "SKIP: HDR presentation is unavailable; see the application log\n";
      return 77;
    }
    Probe sdr_probe(sdr.get()), hdr_probe(hdr.get());
    std::unique_ptr<graphics::Image> sdr_image, hdr_image;
    core->CreateImage(480, 240, graphics::IMAGE_FORMAT_R8G8B8A8_UNORM, &sdr_image);
    core->CreateImage(480, 240, graphics::IMAGE_FORMAT_R32G32B32A32_SFLOAT, &hdr_image);
    std::cout << "These windows show SDR white and HDR reference white. Protocol values and visual comparison "
                 "are not a physical luminance measurement.\n";
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    int frames = 0;
    while (!sdr->ShouldClose() && !hdr->ShouldClose() && std::chrono::steady_clock::now() < end) {
      graphics::Window::PollEvents();
      std::unique_ptr<graphics::CommandContext> commands;
      core->CreateCommandContext(&commands);
      commands->CmdClearImage(sdr_image.get(), {{1, 1, 1, 1}});
      commands->CmdClearImage(hdr_image.get(), {{1, 1, 1, 1}});
      commands->CmdPresent(sdr.get(), sdr_image.get());
      commands->CmdPresent(hdr.get(), hdr_image.get());
      core->SubmitCommandContext(commands.get());
      core->WaitGPU();
      if (++frames >= 3) {
        if (sdr_probe.changed)
          sdr_probe.Report("SDR window");
        if (hdr_probe.changed)
          hdr_probe.Report("HDR window");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }
    core->WaitGPU();
    return sdr_probe.manager && hdr_probe.manager ? 0 : 77;
  } catch (const std::exception &error) {
    std::cerr << "wayland_color_probe: " << error.what() << '\n';
    return 1;
  }
}
