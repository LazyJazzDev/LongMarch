#include "trackpad_gestures.h"

#import <Cocoa/Cocoa.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

void *InstallTrackpadGestures(GLFWwindow *window, std::function<bool(float)> on_magnify) {
  NSWindow *native_window = glfwGetCocoaWindow(window);
  id monitor = [NSEvent
      addLocalMonitorForEventsMatchingMask:NSEventMaskMagnify
                                   handler:^NSEvent *(NSEvent *event) {
                                     if (event.window == native_window && on_magnify(float(event.magnification)))
                                       return nil;
                                     return event;
                                   }];
  return [monitor retain];
}

void RemoveTrackpadGestures(void *monitor) {
  if (monitor) {
    [NSEvent removeMonitor:(id)monitor];
    [(id)monitor release];
  }
}
