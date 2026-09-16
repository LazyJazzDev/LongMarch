#import <Foundation/Foundation.h>
#import <CoreGraphics/CoreGraphics.h>

NS_ASSUME_NONNULL_BEGIN
// Objective-C++ contains all C++ exceptions and owns the serial rendering queue.
@interface SparkiumRenderer : NSObject
- (void)renderResources:(NSURL *)resources
                 scene:(NSString *)scene
             dimension:(NSInteger)dimension
           aspectRatio:(double)aspectRatio
               samples:(NSInteger)samples
              progress:(void (^)(CGImageRef image, NSInteger spp, double seconds, NSString *device))progress
            completion:(void (^)(NSString * _Nullable error, BOOL cancelled))completion
    NS_SWIFT_NAME(render(resources:scene:dimension:aspectRatio:samples:progress:completion:));
- (void)cancel;
@end
NS_ASSUME_NONNULL_END
