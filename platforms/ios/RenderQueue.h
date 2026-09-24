#pragma once
#import <Foundation/Foundation.h>

// Shader-cache configuration and graphics resource lifetimes are serialized across demos.
inline dispatch_queue_t LongMarchRenderQueue() {
  static dispatch_queue_t queue;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    // This queue feeds visible frames and input feedback. Give it foreground
    // QoS; demand-driven scheduling still leaves it asleep between updates.
    auto attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INTERACTIVE, 0);
    queue = dispatch_queue_create("dev.lazyjazz.longmarch.render", attributes);
  });
  return queue;
}
