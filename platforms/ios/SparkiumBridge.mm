#import "SparkiumBridge.h"
#include <atomic>
#include <chrono>
#include "DisplayImage.h"
#include "RenderQueue.h"
#include "RenderSession.h"

@implementation SparkiumRenderer {
  dispatch_queue_t _queue;
  std::atomic<uint64_t> _generation;
  std::atomic<NSInteger> _sampleLimit;
  std::atomic<bool> _paused;
  std::atomic<bool> _hdr;
  std::atomic<float> _exposure;
  std::atomic<uint64_t> _displayRevision;
  uint64_t _displayedRevision;
  // Main-queue state. Each queued operation captures its own callback pair.
  SparkiumProgress _progress;
  SparkiumCompletion _completion;
  uint64_t _sceneEpoch;
  // Rendering-queue state. Preserve the film when reaching/changing the sample limit.
  std::unique_ptr<RenderSession> _session;
  uint64_t _sessionGeneration;
  uint64_t _sessionSceneEpoch;
  double _renderSeconds, _lastFrameSeconds;
}

- (instancetype)init {
  if ((self = [super init])) {
    _queue = LongMarchRenderQueue();
    _generation = 0;
    _sampleLimit = 32;
    _paused = false;
    _hdr = true;
    _exposure = 0;
    _displayRevision = 1;
  }
  return self;
}

- (void)stop {
  NSAssert([NSThread isMainThread], @"Stop rendering on main");
  ++_generation;
  _paused = true;
  _progress = nil;
  _completion = nil;
  dispatch_async(_queue, ^{
    self->_session.reset();
  });
}

- (void)notifyMetadata:(uint64_t)generation progress:(SparkiumProgress)progress {
  NSString *device = [NSString stringWithUTF8String:_session->Device().c_str()];
  NSInteger width = _session->Width(), height = _session->Height(), bounces = _session->MaxBounces();
  dispatch_sync(dispatch_get_main_queue(), ^{
    if (self->_generation == generation)
      progress(nil, 0, 0, 0, device, width, height, bounces);
  });
}

- (void)pump:(uint64_t)generation progress:(SparkiumProgress)progress completion:(SparkiumCompletion)completion {
  @autoreleasepool {
    if (_generation != generation || !_session || _sessionGeneration != generation)
      return;
    NSString *failure = nil;
    try {
      while (_generation == generation && ((!_paused && _session->Samples() < _sampleLimit) ||
                                           (_session->Samples() > 0 && _displayedRevision != _displayRevision))) {
        @autoreleasepool {
          const auto start = std::chrono::steady_clock::now();
          const bool rendered = !_paused && _session->Samples() < _sampleLimit;
          if (rendered)
            _session->Render();
          const uint64_t displayRevision = _displayRevision;
          const bool hdr = _hdr;
          auto pixels = _session->Display(hdr, _exposure);
          _displayedRevision = displayRevision;
          if (rendered) {
            _lastFrameSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            _renderSeconds += _lastFrameSeconds;
          }
          double frameSeconds = _lastFrameSeconds;
          if (_generation != generation)
            return;
          CGImageRef image = CreateDisplayImage(pixels, _session->Width(), _session->Height(), hdr);
          if (!image)
            throw std::runtime_error("Cannot create display image");
          NSString *device = [NSString stringWithUTF8String:_session->Device().c_str()];
          NSInteger spp = _session->Samples(), width = _session->Width(), height = _session->Height();
          NSInteger bounces = _session->MaxBounces();
          double elapsed = _renderSeconds;
          // One preview in flight; never accumulate full-resolution images on the main queue.
          dispatch_sync(dispatch_get_main_queue(), ^{
            if (self->_generation == generation)
              progress(image, spp, elapsed, frameSeconds, device, width, height, bounces);
          });
          CGImageRelease(image);
        }
      }
    } catch (const std::exception &error) {
      failure = [NSString stringWithUTF8String:error.what()];
      _session.reset();
    } catch (...) {
      failure = @"Unexpected rendering failure";
      _session.reset();
    }
    dispatch_async(dispatch_get_main_queue(), ^{
      if (self->_generation == generation)
        completion(failure, self->_paused);
    });
  }
}

- (void)schedulePump {
  NSAssert([NSThread isMainThread], @"Control rendering on the main thread");
  if (!_progress || !_completion)
    return;
  uint64_t generation = _generation;
  SparkiumProgress progress = _progress;
  SparkiumCompletion completion = _completion;
  dispatch_async(_queue, ^{
    [self pump:generation progress:progress completion:completion];
  });
}

- (void)setHDR:(BOOL)hdr exposure:(float)exposure {
  NSAssert([NSThread isMainThread], @"Control display on main");
  _hdr = hdr;
  _exposure = std::clamp(exposure, -10.f, 10.f);
  ++_displayRevision;
  [self schedulePump];
}

- (void)setSampleLimit:(NSInteger)samples {
  NSAssert([NSThread isMainThread], @"Control rendering on the main thread");
  _sampleLimit = std::clamp<NSInteger>(samples, 1, 4096);
  [self schedulePump];
}

- (void)setPaused:(BOOL)paused {
  NSAssert([NSThread isMainThread], @"Control rendering on the main thread");
  _paused = paused;
  if (!paused)
    [self schedulePump];
}

- (void)loadResources:(NSURL *)resources
                scene:(NSString *)scene
              samples:(NSInteger)samples
             progress:(SparkiumProgress)progress
           completion:(SparkiumCompletion)completion {
  NSAssert([NSThread isMainThread], @"Load scenes on the main thread");
  uint64_t generation = ++_generation;  // Cancel old work before enqueuing a replacement.
  uint64_t sceneEpoch = ++_sceneEpoch;
  _progress = [progress copy];
  _completion = [completion copy];
  _sampleLimit = std::clamp<NSInteger>(samples, 1, 4096);
  if ([scene containsString:@"/"] || [scene containsString:@".."] || !resources.isFileURL) {
    dispatch_async(_queue, ^{
      self->_session.reset();
    });
    completion(@"Invalid scene resource path", NO);
    return;
  }
  dispatch_async(_queue, ^{
    @autoreleasepool {
      if (self->_generation != generation)
        return;
      self->_session.reset();  // Release old textures/geometry before loading another scene.
      try {
        self->_session = std::make_unique<RenderSession>(resources.fileSystemRepresentation, scene.UTF8String, 0);
        self->_sessionGeneration = generation;
        self->_sessionSceneEpoch = sceneEpoch;
        self->_renderSeconds = 0;
        self->_lastFrameSeconds = 0;
        if (self->_generation != generation)
          return;
        [self notifyMetadata:generation progress:progress];
        [self pump:generation progress:progress completion:completion];
      } catch (const std::exception &error) {
        NSString *failure = [NSString stringWithUTF8String:error.what()];
        self->_session.reset();
        dispatch_async(dispatch_get_main_queue(), ^{
          if (self->_generation == generation)
            completion(failure, self->_paused);
        });
      }
    }
  });
}

- (void)resetFilm {
  NSAssert([NSThread isMainThread], @"Reset the film on the main thread");
  uint64_t previous = _generation++;
  uint64_t generation = previous + 1;
  uint64_t sceneEpoch = _sceneEpoch;
  SparkiumProgress progress = _progress;
  SparkiumCompletion completion = _completion;
  dispatch_async(_queue, ^{
    @autoreleasepool {
      if (self->_generation != generation || !self->_session || self->_sessionSceneEpoch != sceneEpoch)
        return;
      try {
        self->_session->ResetFilm();
        self->_sessionGeneration = generation;
        self->_renderSeconds = 0;
        self->_lastFrameSeconds = 0;
        [self notifyMetadata:generation progress:progress];
        [self pump:generation progress:progress completion:completion];
      } catch (const std::exception &error) {
        NSString *failure = [NSString stringWithUTF8String:error.what()];
        self->_session.reset();
        dispatch_async(dispatch_get_main_queue(), ^{
          if (self->_generation == generation)
            completion(failure, self->_paused);
        });
      }
    }
  });
}

@end
