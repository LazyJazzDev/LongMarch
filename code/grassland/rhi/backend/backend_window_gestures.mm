#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/rhi_window.h"

#import <Cocoa/Cocoa.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include <algorithm>

namespace grassland::rhi::backend::glfw {
void DispatchCocoaMagnifyEvent(GLFWwindow *window, NSEvent *event) {
  if (!window || event.type != NSEventTypeMagnify)
    return;
  NSWindow *native_window = glfwGetCocoaWindow(window);
  if (event.window != native_window)
    return;
  NSView *view = native_window.contentView;
  const NSPoint point = [view convertPoint:event.locationInWindow fromView:nil];
  const NSRect bounds = view.bounds;
  MagnifyPhase phase = MagnifyPhase::kUpdate;
  if (event.phase & NSEventPhaseCancelled)
    phase = MagnifyPhase::kCancel;
  else if (event.phase & NSEventPhaseEnded)
    phase = MagnifyPhase::kEnd;
  else if (event.phase & NSEventPhaseBegan)
    phase = MagnifyPhase::kBegin;
  // Cocoa reports an incremental fractional change, not a cumulative scale.
  const double scale = phase == MagnifyPhase::kCancel ? 1.0 : std::max(0.001, 1.0 + double(event.magnification));
  DispatchMagnifyEvent(window,
                       MagnifyGesture{scale, point.x - bounds.origin.x,
                                      view.isFlipped ? point.y - bounds.origin.y : NSMaxY(bounds) - point.y, phase});
}

void *InstallMagnifyEvents(GLFWwindow *window) {
  id monitor = [NSEvent addLocalMonitorForEventsMatchingMask:NSEventMaskMagnify
                                                     handler:^NSEvent *(NSEvent *event) {
                                                       DispatchCocoaMagnifyEvent(window, event);
                                                       return event;
                                                     }];
  return [monitor retain];
}

void RemoveMagnifyEvents(void *monitor) {
  if (monitor) {
    [NSEvent removeMonitor:(id)monitor];
    [(id)monitor release];
  }
}
}  // namespace grassland::rhi::backend::glfw
