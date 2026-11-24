#pragma once

#include <Metal/Metal.hpp>
#include <QuartzCore/CAMetalDrawable.hpp>

namespace grassland::rhi {
class IDevice;
class IImage;
class ISubmissionToken;

// Available when the Metal backend is built. The borrowed MTL::Device remains
// valid until device destruction. nullptr for null or non-Metal devices.
MTL::Device *GetNativeMetalDevice(IDevice *device);

// drawable is a borrowed CA::MetalDrawable. The host owns its view/layer and
// acquisition on the UI thread; no UIKit/SwiftUI type enters this interface.
// Copy without scaling or color conversion: both textures must have the same
// size/format, and the drawable's layer must have framebufferOnly = false.
// rendered must describe the last write to source. This synchronous operation
// encodes a GPU timeline wait before copying, then waits for copy completion and
// render-result publication before returning, so the
// host may safely reuse/release source afterwards. Serialize calls with source
// writes and device destruction. False means validation, render or copy failure.
bool PresentMetalDrawable(IDevice *device, IImage *source, ISubmissionToken *rendered, CA::MetalDrawable *drawable);
}  // namespace grassland::rhi
