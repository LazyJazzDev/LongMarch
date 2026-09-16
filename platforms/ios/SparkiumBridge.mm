#import "SparkiumBridge.h"
#include "RenderSession.h"
#include <atomic>
#include <chrono>
#include <cmath>

@implementation SparkiumRenderer {
  dispatch_queue_t _queue;
  std::atomic<bool> _cancelled;
  BOOL _busy;  // Accessed only on the main queue.
}
- (instancetype)init {
  if ((self = [super init])) {
    _queue = dispatch_queue_create("dev.lazyjazz.sparkium.render", DISPATCH_QUEUE_SERIAL);
    _cancelled = false;
  }
  return self;
}
- (void)cancel { _cancelled = true; }
- (void)renderResources:(NSURL *)resources scene:(NSString *)scene dimension:(NSInteger)dimension
           aspectRatio:(double)aspectRatio
               samples:(NSInteger)samples
              progress:(void (^)(CGImageRef, NSInteger, double, NSString *))progress
            completion:(void (^)(NSString * _Nullable, BOOL))completion {
  NSAssert([NSThread isMainThread], @"Start rendering on the main thread");
  if (_busy) { completion(@"A render is already running", NO); return; }
  if (!std::isfinite(aspectRatio) || aspectRatio < 0.25 || aspectRatio > 4.0 || dimension < 64 || dimension > 16384 ||
      samples < 1 || samples > 4096 ||
      [scene containsString:@"/"] || [scene containsString:@".."] || !resources.isFileURL) {
    completion(@"Invalid render settings", NO); return;
  }
  _busy = YES;
  _cancelled = false;
  dispatch_async(_queue, ^{
    @autoreleasepool {
      NSString *failure = nil;
      try {
        const auto start = std::chrono::steady_clock::now();
        RenderSession session(resources.fileSystemRepresentation, scene.UTF8String, static_cast<int>(dimension),
                              false, aspectRatio);
        for (NSInteger i = 0; i < samples && !self->_cancelled; ++i) {
          @autoreleasepool {
            auto pixels = session.Step();
            NSData *data = [NSData dataWithBytes:pixels.data() length:pixels.size()];
            CGDataProviderRef provider = CGDataProviderCreateWithCFData((__bridge CFDataRef)data);
            CGColorSpaceRef colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            CGImageRef image = CGImageCreate(session.Width(), session.Height(), 8, 32, session.Width() * 4,
                colorSpace, kCGBitmapByteOrderDefault | kCGImageAlphaLast, provider, nullptr, false,
                kCGRenderingIntentDefault);
            CGColorSpaceRelease(colorSpace);
            CGDataProviderRelease(provider);
            if (!image) throw std::runtime_error("Cannot create display image");
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            NSString *device = [NSString stringWithUTF8String:session.Device().c_str()];
            NSInteger spp = session.Samples();
            // Back pressure limits preview memory to one image. The main thread never waits for the GPU.
            dispatch_sync(dispatch_get_main_queue(), ^{ progress(image, spp, elapsed, device); });
            CGImageRelease(image);
          }
        }
      } catch (const std::exception &error) {
        failure = [NSString stringWithUTF8String:error.what()];
      } catch (...) {
        failure = @"Unexpected rendering failure";
      }
      BOOL cancelled = self->_cancelled;
      dispatch_async(dispatch_get_main_queue(), ^{
        self->_busy = NO;
        completion(failure, cancelled);
      });
    }
  });
}
@end
