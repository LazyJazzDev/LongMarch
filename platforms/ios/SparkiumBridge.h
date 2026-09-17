#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>
#import <TargetConditionals.h>

NS_ASSUME_NONNULL_BEGIN
// Callbacks run on the main queue; progress applies synchronous back pressure.
// A nil image announces loaded/reset metadata.
typedef void (^SparkiumProgress)(CGImageRef _Nullable image, NSInteger spp, double seconds,
                                double frameSeconds, NSString *device, NSInteger width,
                                NSInteger height, NSInteger maxBounces);
typedef void (^SparkiumCompletion)(NSString * _Nullable error, BOOL paused);

// Call public methods on the main thread. GPU resources live exclusively on the rendering queue.
@interface SparkiumRenderer : NSObject
- (void)loadResources:(NSURL *)resources
               scene:(NSString *)scene
             samples:(NSInteger)samples
            progress:(SparkiumProgress)progress
          completion:(SparkiumCompletion)completion
    NS_SWIFT_NAME(load(resources:scene:samples:progress:completion:));
- (void)setSampleLimit:(NSInteger)samples;
- (void)setPaused:(BOOL)paused;
- (void)resetFilm;
- (void)stop;
@end
NS_ASSUME_NONNULL_END

#if TARGET_OS_IOS
#import "demos/DemoBridge.h"
#endif
