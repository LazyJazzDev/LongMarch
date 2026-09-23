#import <Cocoa/Cocoa.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <gtest/gtest.h>

#include "grassland/graphics/window.h"
#include "grassland/graphics/window_gestures.h"

@interface TestMagnifyEvent : NSEvent
@property(nonatomic, assign) NSWindow *target;
@property(nonatomic) NSEventPhase testPhase;
@property(nonatomic) CGFloat delta;
@property(nonatomic) NSPoint point;
@end
@implementation TestMagnifyEvent

- (NSEventType)type {
  return NSEventTypeMagnify;
}

- (NSWindow *)window {
  return self.target;
}

- (NSInteger)windowNumber {
  return self.target.windowNumber;
}

- (NSEventPhase)phase {
  return self.testPhase;
}

- (CGFloat)magnification {
  return self.delta;
}

- (NSPoint)locationInWindow {
  return self.point;
}

@end

namespace {
using namespace grassland::graphics;

class TestWindow : public Window {
 public:
  TestWindow() : Window(320, 240, "Window gesture test", false, true, false) {
  }

  void InitImGui(const char *, float) override {
  }

  void TerminateImGui() override {
  }

  void BeginImGuiFrame() override {
  }

  void EndImGuiFrame() override {
  }

  ImGuiContext *GetImGuiContext() const override {
    return nullptr;
  }
};

void Send(Window &first, Window &second, NSWindow *window, NSEventPhase phase, double delta) {
  TestMagnifyEvent *event = [[TestMagnifyEvent alloc] init];
  event.target = window;
  event.testPhase = phase;
  event.delta = delta;
  // A point 40 units right and 30 units down from the content's top-left.
  NSView *view = window.contentView;
  event.point = [view convertPoint:NSMakePoint(40, view.isFlipped ? 30 : view.bounds.size.height - 30) toView:nil];
  detail::DispatchMagnifyEvent(&first, event);
  detail::DispatchMagnifyEvent(&second, event);
  [event release];
}

TEST(WindowGestures, AdapterRoutingScaleCoordinatesPhasesAndCleanup) {
  @autoreleasepool {
    TestWindow first;
    TestWindow second;
    ASSERT_TRUE(first.SupportsMagnifyGestures());
    ASSERT_TRUE(second.SupportsMagnifyGestures());
    std::vector<MagnifyGesture> received;
    int other_events = 0;
    auto id = first.MagnifyEvent().RegisterCallback([&](const MagnifyGesture &event) { received.push_back(event); });
    second.MagnifyEvent().RegisterCallback([&](const MagnifyGesture &) { ++other_events; });
    NSWindow *native = glfwGetCocoaWindow(first.GLFWWindow());
    Send(first, second, native, NSEventPhaseBegan, 0.0);
    Send(first, second, native, NSEventPhaseChanged, 0.25);
    Send(first, second, native, NSEventPhaseEnded, -0.1);
    Send(first, second, native, NSEventPhaseCancelled, 0.5);
    ASSERT_EQ(received.size(), 4);
    EXPECT_EQ(other_events, 0);
    EXPECT_EQ(received[0].phase, MagnifyPhase::kBegin);
    EXPECT_EQ(received[1].phase, MagnifyPhase::kUpdate);
    EXPECT_EQ(received[2].phase, MagnifyPhase::kEnd);
    EXPECT_EQ(received[3].phase, MagnifyPhase::kCancel);
    EXPECT_DOUBLE_EQ(received[1].scale, 1.25);
    EXPECT_DOUBLE_EQ(received[2].scale, 0.9);
    EXPECT_DOUBLE_EQ(received[3].scale, 1.0);
    EXPECT_DOUBLE_EQ(received[1].x, 40);
    EXPECT_DOUBLE_EQ(received[1].y, 30);
    first.MagnifyEvent().UnregisterCallback(id);
    Send(first, second, native, NSEventPhaseChanged, 0.1);
    EXPECT_EQ(received.size(), 4);
    first.CloseWindow();
    EXPECT_FALSE(first.SupportsMagnifyGestures());
    first.CloseWindow();
    Send(first, second, glfwGetCocoaWindow(second.GLFWWindow()), NSEventPhaseChanged, 0.1);
    EXPECT_EQ(other_events, 1);
  }
}
}  // namespace
