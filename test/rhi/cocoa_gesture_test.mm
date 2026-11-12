#include <gtest/gtest.h>

#include <cstdlib>
#include "grassland/rhi/backend/backend_glfw_helper.h"
#include "grassland/rhi/rhi.h"
#define GLFW_EXPOSE_NATIVE_COCOA
#import <Cocoa/Cocoa.h>
#include <GLFW/glfw3native.h>

// Synthetic event payload; native NSWindow/NSView coordinate conversion and
// the RHI registry/lifetime path below are real. No trackpad is required.
@interface RHIMagnifyTestEvent : NSObject
@property NSEventType type;
@property(assign) NSWindow *window;
@property NSPoint locationInWindow;
@property NSEventPhase phase;
@property CGFloat magnification;
@end
@implementation RHIMagnifyTestEvent
@end

namespace {
void CheckMagnify(grassland::rhi::BackendAPI api) {
  if (!std::getenv("LONGMARCH_RHI_WINDOW_TESTS"))
    GTEST_SKIP() << "Requires a desktop.";
  using namespace grassland;
  using namespace grassland::rhi;
  ObjPtr<ICore> core = rhi::CreateCore(api);
  ASSERT_TRUE(core);
  ObjPtr<IDevice> device = core->CreateDeviceByAutoSelection();
  ASSERT_TRUE(device);
  ObjPtr<IWindow> window = device->CreateGLFWWindow({320, 240, "RHI magnify test"});
  ObjPtr<IWindow> other = device->CreateGLFWWindow({320, 240, "RHI gesture isolation"});
  ASSERT_TRUE(window);
  ASSERT_TRUE(other);
  ASSERT_TRUE(window->SupportsMagnifyGestures());
  ASSERT_TRUE(other->SupportsMagnifyGestures());
  ASSERT_TRUE(rhi::PollEvents());
  auto native = window->GLFWWindow();
  NSWindow *cocoa = glfwGetCocoaWindow(native);
  std::vector<MagnifyGesture> events;
  int other_events = 0, scroll_events = 0;
  auto callback = window->MagnifyEvent().RegisterCallback([&](const auto &event) { events.push_back(event); });
  other->MagnifyEvent().RegisterCallback([&](const auto &) { ++other_events; });
  window->ScrollEvent().RegisterCallback([&](double, double) { ++scroll_events; });
  @autoreleasepool {
    auto event = [[[RHIMagnifyTestEvent alloc] init] autorelease];
    event.type = NSEventTypeMagnify;
    event.window = cocoa;
    NSView *view = cocoa.contentView;
    auto bounds = view.bounds;
    const NSPoint point =
        NSMakePoint(bounds.origin.x + 40, view.isFlipped ? bounds.origin.y + 60 : NSMaxY(bounds) - 60);
    event.locationInWindow = [view convertPoint:point toView:nil];
    const NSEventPhase phases[] = {NSEventPhaseBegan, NSEventPhaseChanged, NSEventPhaseEnded, NSEventPhaseCancelled};
    const MagnifyPhase expected[] = {MagnifyPhase::kBegin, MagnifyPhase::kUpdate, MagnifyPhase::kEnd,
                                     MagnifyPhase::kCancel};
    for (int i = 0; i < 4; ++i) {
      event.phase = phases[i];
      event.magnification = 0.25;
      rhi::backend::glfw::DispatchCocoaMagnifyEvent(other->GLFWWindow(), (NSEvent *)event);
      rhi::backend::glfw::DispatchCocoaMagnifyEvent(native, (NSEvent *)event);
      ASSERT_EQ(events.size(), i + 1);
      EXPECT_EQ(events.back().phase, expected[i]);
      EXPECT_DOUBLE_EQ(events.back().scale, i == 3 ? 1.0 : 1.25);
      EXPECT_DOUBLE_EQ(events.back().x, 40);
      EXPECT_DOUBLE_EQ(events.back().y, 60);
    }
    event.phase = NSEventPhaseChanged;
    event.magnification = -0.5;
    rhi::backend::glfw::DispatchCocoaMagnifyEvent(native, (NSEvent *)event);
    EXPECT_DOUBLE_EQ(events.back().scale, 0.5);
    event.type = NSEventTypeScrollWheel;
    rhi::backend::glfw::DispatchCocoaMagnifyEvent(native, (NSEvent *)event);
    EXPECT_EQ(events.size(), 5);
    EXPECT_EQ(other_events, 0);
    EXPECT_EQ(scroll_events, 0);
    window->MagnifyEvent().UnregisterCallback(callback);
    event.type = NSEventTypeMagnify;
    rhi::backend::glfw::DispatchCocoaMagnifyEvent(native, (NSEvent *)event);
    EXPECT_EQ(events.size(), 5);
    // Dropping the last external reference must not destroy the registry while
    // callbacks are running. The dispatcher's temporary retain covers both.
    bool continued = false;
    window->MagnifyEvent().RegisterCallback([&](const auto &) { window.Release(); }, 200);
    window->MagnifyEvent().RegisterCallback([&](const auto &) { continued = true; }, 100);
    rhi::backend::glfw::DispatchCocoaMagnifyEvent(native, (NSEvent *)event);
    EXPECT_FALSE(window);
    EXPECT_TRUE(continued);
    // A stale registry key is ignored without dereferencing its old GLFW handle.
    rhi::backend::glfw::DispatchMagnifyEvent(native, {});
  }
  ObjPtr<IWindow> replacement = device->CreateGLFWWindow({320, 240, "RHI gesture recreation"});
  ASSERT_TRUE(replacement);
  EXPECT_TRUE(replacement->SupportsMagnifyGestures());
  EXPECT_TRUE(rhi::PollEvents());
}
}  // namespace

#ifdef LONGMARCH_METAL_ENABLED
TEST(RhiCocoaGesture, MetalCoordinatesPhasesIsolationAndLifetime) {
  CheckMagnify(grassland::rhi::BACKEND_API_METAL);
}
#endif
#ifdef LONGMARCH_VULKAN_ENABLED
TEST(RhiCocoaGesture, VulkanCoordinatesPhasesIsolationAndLifetime) {
  CheckMagnify(grassland::rhi::BACKEND_API_VULKAN);
}
#endif
