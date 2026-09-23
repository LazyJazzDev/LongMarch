#pragma once

#ifdef __OBJC__
@class NSEvent;
#endif

namespace grassland::graphics {
class Window;

namespace detail {
// Owned by Window; removed before the GLFW window is destroyed.
void *InstallMagnifyEvents(Window *window);
void RemoveMagnifyEvents(void *monitor);
#ifdef __OBJC__
void DispatchMagnifyEvent(Window *window, NSEvent *event);
#endif
}  // namespace detail
}  // namespace grassland::graphics
