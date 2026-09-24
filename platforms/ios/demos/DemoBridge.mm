#import "DemoBridge.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "DemoSession.h"
#include "RenderQueue.h"
#include "grassland/graphics/backend/metal/metal_core.h"
#include "grassland/graphics/backend/metal/metal_image.h"

@implementation DemoRenderer {
  MTKView *_view;
  DemoProgress _progress;
  std::atomic<uint64_t> _generation;
  BOOL _active, _busy, _failed, _filePending;
  NSString *_demo;
  NSInteger _particles, _galaxies, _reset;
  float _deltaTime, _yaw, _pitch, _resolutionScale;
  BOOL _simulate, _hdr;
  CFTimeInterval _statsTime;
  NSInteger _statsFrames;
  // Owned and used only on LongMarchRenderQueue.
  std::unique_ptr<DemoSession> _session;
  id<MTLRenderPipelineState> _presentPipeline;
  NSInteger _frames;
}

- (instancetype)init {
  if ((self = [super init])) {
    _generation = 0;
    _particles = 4096;
    _galaxies = 10;
    _deltaTime = .03f;
    _simulate = YES;
    _hdr = YES;
    _resolutionScale = 1;
  }
  return self;
}

- (void)startView:(MTKView *)view resources:(NSURL *)resources demo:(NSString *)demo progress:(DemoProgress)progress {
  NSAssert(NSThread.isMainThread, @"Configure demo views on main");
  uint64_t generation = ++_generation;
  _view = view;
  _progress = [progress copy];
  _demo = [demo copy];
  _failed = NO;
  _filePending = NO;
  _statsTime = 0;
  _statsFrames = 0;
  view.device = MTLCreateSystemDefaultDevice();
  BOOL hdrDemo = [demo isEqualToString:@"graphics_hello_hdr"];
  view.colorPixelFormat = hdrDemo ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
  if (hdrDemo) {
    CAMetalLayer *layer = (CAMetalLayer *)view.layer;
    CGColorSpaceRef space = CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
    layer.colorspace = space;
    CGColorSpaceRelease(space);
    layer.wantsExtendedDynamicRangeContent = YES;
  }
  view.framebufferOnly = YES;
  view.preferredFramesPerSecond = 60;
  view.delegate = self;
  view.paused = !_active;
  dispatch_async(LongMarchRenderQueue(), ^{
    @autoreleasepool {
      if (self->_generation != generation)
        return;
      self->_session.reset();
      self->_presentPipeline = nil;
      self->_frames = 0;
      try {
        self->_session = std::make_unique<DemoSession>(resources.fileSystemRepresentation, demo.UTF8String);
        auto core = static_cast<grassland::graphics::backend::MetalCore *>(self->_session->Core());
        id<MTLDevice> device = (__bridge id<MTLDevice>)core->Device();
        NSString *source =
            @"#include <metal_stdlib>\nusing namespace metal;\n"
             "struct V{float4 p [[position]];float2 uv;};\n"
             "vertex V present_vertex(uint i [[vertex_id]]){float2 p=float2((i<<1)&2,i&2);return "
             "{float4(p*float2(2,-2)+float2(-1,1),0,1),p};}\n"
             "fragment float4 present_fragment(V v [[stage_in]],texture2d<float> image [[texture(0)]],constant uint &clampSDR [[buffer(0)]]){"
             "constexpr sampler s(coord::normalized,address::clamp_to_edge,filter::linear);return "
             "float4(clampSDR ? min(image.sample(s,v.uv).rgb,float3(1)) : image.sample(s,v.uv).rgb,1);}";
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (!library)
          throw std::runtime_error(error.localizedDescription.UTF8String);
        MTLRenderPipelineDescriptor *descriptor = [MTLRenderPipelineDescriptor new];
        descriptor.vertexFunction = [library newFunctionWithName:@"present_vertex"];
        descriptor.fragmentFunction = [library newFunctionWithName:@"present_fragment"];
        descriptor.colorAttachments[0].pixelFormat = hdrDemo ? MTLPixelFormatRGBA16Float : MTLPixelFormatBGRA8Unorm;
        self->_presentPipeline = [device newRenderPipelineStateWithDescriptor:descriptor error:&error];
        if (!self->_presentPipeline)
          throw std::runtime_error(error.localizedDescription.UTF8String);
      } catch (const std::exception &error) {
        NSString *failure = [NSString stringWithUTF8String:error.what()];
        self->_session.reset();
        dispatch_async(dispatch_get_main_queue(), ^{
          if (self->_generation == generation) {
            self->_failed = YES;
            progress(0, 0, 0, @"—", 0, 0, 0, failure);
          }
        });
      }
    }
  });
}

- (void)setHDR:(BOOL)hdr {
  _hdr = hdr;
  if ([_demo isEqualToString:@"graphics_hello_hdr"])
    ((CAMetalLayer *)_view.layer).wantsExtendedDynamicRangeContent = hdr;
}

- (void)input:(NSInteger)kind x:(double)x y:(double)y value:(double)value {
  uint64_t generation = _generation;
  dispatch_async(LongMarchRenderQueue(), ^{
    if (self->_generation != generation || !self->_session || !self->_session->Game())
      return;
    auto *window = self->_session->Game()->Window();
    auto size = window->GetFramebufferSize();
    if (kind <= 3 || kind == 6)
      window->SendPointer(x * size.x, y * size.y);
    if (kind == 1) {
      window->CursorEnterEvent().InvokeCallbacks(true);
      window->SendMouseButton(int(value), GLFW_PRESS);
    }
    if (kind == 2) {
      window->SendMouseButton(int(value), GLFW_RELEASE);
      window->CursorEnterEvent().InvokeCallbacks(false);
    }
    if (kind == 4) {
      window->SendKey(int(value), GLFW_PRESS);
      window->SendKey(int(value), GLFW_RELEASE);
    }
    if (kind == 5) {
      window->SendFocus(value != 0);
      if (value)
        self->_session->Game()->ResetClock();
    }
    if (kind == 6)
      window->MagnifyEvent().InvokeCallbacks(grassland::graphics::MagnifyGesture{
          value, x * size.x, y * size.y, grassland::graphics::MagnifyPhase::kUpdate});
  });
}

- (void)completeFile:(NSString *)path completion:(void (^)(NSString *))completion {
  uint64_t generation = _generation;
  dispatch_async(LongMarchRenderQueue(), ^{
    std::string error;
    if (self->_generation == generation && self->_session && self->_session->Game())
      error = self->_session->Game()->CompleteFile(path.UTF8String);
    NSString *message = error.empty() ? nil : [NSString stringWithUTF8String:error.c_str()];
    dispatch_async(dispatch_get_main_queue(), ^{
      self->_filePending = NO;
      completion(message);
    });
  });
}

- (void)setActive:(BOOL)active {
  if (_active != active)
    [self input:5 x:0 y:0 value:active];
  if (_active != active)
    _statsTime = 0;
  _active = active;
  _view.paused = !active;
}

- (void)configureParticles:(NSInteger)particles
                  galaxies:(NSInteger)galaxies
                 deltaTime:(float)deltaTime
                  simulate:(BOOL)simulate
                       yaw:(float)yaw
                     pitch:(float)pitch
                     reset:(NSInteger)reset
           resolutionScale:(float)scale {
  _particles = particles;
  _galaxies = galaxies;
  _deltaTime = deltaTime;
  _simulate = simulate;
  _yaw = yaw;
  _pitch = pitch;
  _reset = reset;
  _resolutionScale = scale;
}

- (void)mtkView:(MTKView *)view drawableSizeWillChange:(CGSize)size {
}

- (void)drawInMTKView:(MTKView *)view {
  if (!_active || _busy || _failed)
    return;
  id<CAMetalDrawable> drawable = view.currentDrawable;
  if (!drawable)
    return;
  _busy = YES;
  uint64_t generation = _generation;
  auto particles = _particles, galaxies = _galaxies, reset = _reset;
  float dt = _deltaTime, yaw = _yaw, pitch = _pitch;
  BOOL simulate = _simulate;
  uint32_t clampSDR = !_hdr;
  int width = 1280, height = 720;
  if ([_demo isEqualToString:@"graphics_hello_resize"] || [_demo isEqualToString:@"nbody_cs"] ||
      [_demo isEqualToString:@"gol"] || [_demo isEqualToString:@"2048"]) {
    width = std::max(1, int(view.drawableSize.width * _resolutionScale));
    height = std::max(1, int(view.drawableSize.height * _resolutionScale));
  }
  DemoProgress progress = _progress;
  dispatch_async(LongMarchRenderQueue(), ^{
    @autoreleasepool {
      NSString *failure = nil, *deviceName = @"—";
      double seconds = 0, gpuMS = 0;
      NSInteger frames = 0;
      try {
        if (self->_generation == generation && self->_session) {
          auto start = std::chrono::steady_clock::now();
          self->_session->Resize(width, height);
          self->_session->Configure(int(particles), int(galaxies), dt, simulate, yaw, pitch, int(reset));
          self->_session->Render();
          if (auto game = self->_session->Game()) {
            int action = game->FileRequest();
            dispatch_async(dispatch_get_main_queue(), ^{
              if (self->_generation == generation && action && !self->_filePending && self.fileRequest) {
                self->_filePending = YES;
                self.fileRequest(action);
              }
            });
          }
          auto core = static_cast<grassland::graphics::backend::MetalCore *>(self->_session->Core());
          auto image = static_cast<grassland::graphics::backend::MetalImage *>(self->_session->Image());
          id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)core->Queue();
          id<MTLCommandBuffer> command = [queue commandBuffer];
          MTLRenderPassDescriptor *pass = [MTLRenderPassDescriptor renderPassDescriptor];
          pass.colorAttachments[0].texture = drawable.texture;
          pass.colorAttachments[0].loadAction = MTLLoadActionClear;
          pass.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
          pass.colorAttachments[0].storeAction = MTLStoreActionStore;
          id<MTLRenderCommandEncoder> encoder = [command renderCommandEncoderWithDescriptor:pass];
          double fit = std::min(double(drawable.texture.width) / width, double(drawable.texture.height) / height);
          [encoder
              setViewport:MTLViewport{(drawable.texture.width - width * fit) / 2,
                                      (drawable.texture.height - height * fit) / 2, width * fit, height * fit, 0, 1}];
          [encoder setRenderPipelineState:self->_presentPipeline];
          [encoder setFragmentBytes:&clampSDR length:sizeof(clampSDR) atIndex:0];
          [encoder setFragmentTexture:(__bridge id<MTLTexture>)image->Handle() atIndex:0];
          [encoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [encoder endEncoding];
          [command presentDrawable:drawable];
          [command commit];
          [command waitUntilCompleted];
          if (command.error)
            throw std::runtime_error(command.error.localizedDescription.UTF8String);
          seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          gpuMS = self->_session->GPUMilliseconds();
          frames = ++self->_frames;
          deviceName = [NSString stringWithUTF8String:core->DeviceName().c_str()];
          if (frames == 3 && NSProcessInfo.processInfo.environment[@"LONGMARCH_SMOKE_DEMO"]) {
            NSDictionary *result = @{
              @"demo" : self->_demo,
              @"width" : @(width),
              @"height" : @(height),
              @"frames" : @(frames),
              @"device" : deviceName,
              @"gpu_ms" : @(gpuMS),
              @"frame_seconds" : @(seconds)
            };
            NSURL *url = [NSFileManager.defaultManager URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask]
                             .firstObject;
            [[NSJSONSerialization dataWithJSONObject:result options:NSJSONWritingPrettyPrinted error:nil]
                writeToURL:[url URLByAppendingPathComponent:@"DemoSmokeResult.json"]
                atomically:YES];
          }
        }
      } catch (const std::exception &error) {
        failure = [NSString stringWithUTF8String:error.what()];
        self->_session.reset();
      }
      dispatch_async(dispatch_get_main_queue(), ^{
        self->_busy = NO;
        if (self->_generation == generation) {
          if (failure)
            self->_failed = YES;
          if ((frames > 0 && (frames <= 3 || frames % 15 == 0)) || failure) {
            CFTimeInterval now = CACurrentMediaTime();
            double fps = self->_statsTime > 0 ? (frames - self->_statsFrames) / (now - self->_statsTime) : 0;
            self->_statsTime = now;
            self->_statsFrames = frames;
            progress(seconds, gpuMS, fps, deviceName, width, height, frames, failure);
          }
        }
      });
    }
  });
}

- (void)stop {
  ++_generation;
  _active = NO;
  _view.paused = YES;
  _view.delegate = nil;
  _view = nil;
  _progress = nil;
  dispatch_async(LongMarchRenderQueue(), ^{
    self->_session.reset();
    self->_presentPipeline = nil;
  });
}

@end
