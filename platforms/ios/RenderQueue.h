#pragma once
#import <Foundation/Foundation.h>
// Shader-cache configuration and graphics resource lifetimes are serialized across demos.
inline dispatch_queue_t LongMarchRenderQueue() {
  static dispatch_queue_t queue;
  static dispatch_once_t once;
  dispatch_once(&once, ^{
    queue = dispatch_queue_create("dev.lazyjazz.longmarch.render", DISPATCH_QUEUE_SERIAL);
  });
  return queue;
}
