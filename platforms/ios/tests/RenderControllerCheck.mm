#import "SparkiumBridge.h"
#include <cstdlib>
#include <iostream>

static SparkiumRenderer *renderer;
static NSURL *resources;
static int phase = 0, lastSample = 0, metadata = 0;
static NSData *firstFrame;
static double lastSeconds = 0;
static bool finished = false;
static void Check(bool condition, const char *message) {
  if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
static void CheckError(NSString *error) {
  if (error) { std::cerr << error.UTF8String << '\n'; std::exit(1); }
}
static void Finish() {
  finished = true;
  std::cout << "PASS preset resolution, limit increase/decrease, pause/resume, reset, latest-scene selection\n";
  std::exit(0);
}
static void CheckLatestScene() {
  [renderer loadResources:resources scene:@"specular" samples:1
    progress:^(CGImageRef, NSInteger, double, double, NSString *, NSInteger, NSInteger, NSInteger) {
      Check(false, "superseded scene delivered progress");
    } completion:^(NSString *, BOOL) { Check(false, "superseded scene delivered completion"); }];
  [renderer loadResources:resources scene:@"texture" samples:1
    progress:^(CGImageRef image, NSInteger spp, double seconds, double frameSeconds, NSString *, NSInteger w, NSInteger h, NSInteger) {
      Check(w == 2048 && h == 1024, "texture does not use its original film dimensions");
      if (image) { Check(spp == 1 && seconds > 0 && frameSeconds > 0, "invalid texture progress"); lastSample = spp; }
    } completion:^(NSString *error, BOOL paused) { CheckError(error); Check(!paused && lastSample == 1, "texture failed"); Finish(); }];
}
int main(int argc, char **argv) {
  @autoreleasepool {
    Check(argc == 2, "usage: RenderControllerCheck <prepared resources>");
    resources = [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
    renderer = [SparkiumRenderer new];
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 120 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
      if (!finished) { std::cerr << "Render controller timed out\n"; std::exit(1); }
    });
    [renderer loadResources:resources scene:@"cornell_box" samples:2
      progress:^(CGImageRef image, NSInteger spp, double seconds, double frameSeconds, NSString *device, NSInteger w, NSInteger h, NSInteger bounces) {
        Check(w == 1024 && h == 1024 && bounces == 32, "scene presets changed");
        Check(device.length > 0, "missing device name");
        if (!image) { metadata++; lastSample = 0; lastSeconds = 0; return; }
        Check(spp == lastSample + 1, "limit change discarded or skipped accumulated samples");
        Check(seconds > lastSeconds && frameSeconds > 0, "invalid timing statistics");
        Check(CGImageGetWidth(image) == 1024 && CGImageGetHeight(image) == 1024, "image extent mismatch");
        if (spp == 1) {
          NSData *pixels = CFBridgingRelease(CGDataProviderCopyData(CGImageGetDataProvider(image)));
          if (phase == 0) firstFrame = pixels;
          if (phase == 5) Check([firstFrame isEqualToData:pixels], "reset film does not reproduce the first sample");
        }
        lastSample = spp; lastSeconds = seconds;
      }
      completion:^(NSString *error, BOOL paused) {
        CheckError(error);
        switch (phase++) {
          case 0:
            Check(lastSample == 2 && metadata == 1, "initial cap not reached");
            [renderer setSampleLimit:4];
            break;
          case 1:
            Check(lastSample == 4 && metadata == 1, "raising cap reloaded the scene");
            [renderer setSampleLimit:1];
            break;
          case 2:
            Check(lastSample == 4 && metadata == 1, "lowering cap reset the film");
            [renderer setPaused:YES];
            [renderer setSampleLimit:8];
            break;
          case 3:
            Check(paused && lastSample == 4, "paused controller rendered samples");
            [renderer setPaused:NO];
            break;
          case 4:
            Check(!paused && lastSample == 8 && metadata == 1, "resume lost accumulation");
            [renderer setSampleLimit:1];
            [renderer resetFilm];
            [renderer resetFilm]; // A superseded reset must not strand the latest reset.
            break;
          case 5:
            Check(lastSample == 1 && metadata == 2, "reset film did not restart sampling");
            CheckLatestScene();
            break;
          default: Check(false, "stale completion delivered after switching scenes");
        }
      }];
    // Keep the actual main thread alive, as UIApplication does on iOS.
    // dispatch_main() can execute the main dispatch queue on a worker thread on macOS.
    [NSTimer scheduledTimerWithTimeInterval:125 repeats:NO block:^(NSTimer *) {
      Check(false, "render controller timed out");
    }];
    [[NSRunLoop mainRunLoop] run];
  }
}
