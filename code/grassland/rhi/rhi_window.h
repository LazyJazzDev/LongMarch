#pragma once
#ifndef GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include "grassland/rhi/rhi_object.h"

namespace grassland::rhi {
class IImage;
// Window operations and presentation run on the main thread.
bool PollEvents();

class IWindow : public IObject {
 public:
  // Borrowed handle; Window owns it. Use GLFW operations on the main thread.
  virtual GLFWwindow *GLFWWindow() const = 0;
  virtual bool ShouldClose() const = 0;
  virtual bool IsKeyDown(int key) const = 0;
  virtual bool SetTitle(const std::string &title) = 0;
  // Returns an owned reference to the same next image until Present submission
  // or swapchain replacement. Null while minimized, zero-sized or invalid.
  // This getter never acquires a native image or rebuilds the swapchain.
  // Release caller-held frames and recordings before swapchain replacement or
  // Window destruction. Window waits for GPU use; it is not retained by Device.
  virtual IImage *GetNextImage() = 0;
  // Known swapchain invalidation, not a native surface probe. OUT_OF_DATE
  // marks this state without rebuilding during command submission.
  virtual bool IsSwapChainInvalid() const = 0;
  // Call on the window thread outside recording, after releasing caller-held
  // frames and recordings. Success may remain invalid if the surface changes
  // again or is minimized; check IsSwapChainInvalid() before rendering.
  virtual bool RebuildSwapChain() = 0;
  virtual bool SetPresentMode(PresentMode mode) = 0;
  virtual PresentMode GetPresentMode() const = 0;
  // Explicit replacement, with the same recording/lifetime rules as RebuildSwapChain.
  // Unsupported requests return false without replacing the current swapchain.
  // A native failure can invalidate it; inspect IsSwapChainInvalid(). No SDR fallback.
  virtual bool SetColorSpace(ColorSpace color_space) = 0;
  // Encoding of the last successfully configured, drawable swapchain.
  virtual ColorSpace GetColorSpace() const = 0;
  // Query reference white in cd/m^2 on the window thread. Windows requires HDR enabled.
  // Unsupported platforms (including Linux) and failed queries return false.
  // On failure nits is unchanged. This does not change presentation encoding.
  virtual bool GetHDRReferenceWhiteNits(float &nits) const = 0;
  // Multiply relative linear RGB by this to align 1.0 with SDR white.
  // Windows: reference white / 80; macOS EDR and Linux: 1. Unavailable: 1.
  // Linux uses fixed source encoding parameters without an output-white query.
  virtual float GetHDRReferenceWhiteScale() const = 0;
  // Peak / reference white, in relative linear units, at least 1.
  // Unavailable display information returns 10, not a measured capability.
  virtual float GetHDRHeadroom() const = 0;
  // Currently implemented combinations supported by this window's surface.
  // Formats are selected within the requested color space at creation/rebuild.
  // This is presentation capability, not a measurement of display luminance. Empty on failure.
  virtual std::vector<WindowPresentationOption> GetSupportedPresentationOptions() const = 0;

 protected:
  ~IWindow() override;
};

}  // namespace grassland::rhi
