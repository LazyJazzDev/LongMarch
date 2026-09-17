#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
NS_ASSUME_NONNULL_BEGIN
typedef void (^DemoProgress)(double frameSeconds, double gpuMilliseconds, double fps, NSString *device, NSInteger width,
                             NSInteger height, NSInteger frames, NSString *_Nullable error);
@interface DemoRenderer : NSObject <MTKViewDelegate>
- (void)startView:(MTKView *)view
        resources:(NSURL *)resources
             demo:(NSString *)demo
         progress:(DemoProgress)progress NS_SWIFT_NAME(start(view:resources:demo:progress:));
- (void)setActive:(BOOL)active;
- (void)configureParticles:(NSInteger)particles
                  galaxies:(NSInteger)galaxies
                 deltaTime:(float)deltaTime
                  simulate:(BOOL)simulate
                       yaw:(float)yaw
                     pitch:(float)pitch
                     reset:(NSInteger)reset
           resolutionScale:(float)scale
    NS_SWIFT_NAME(configure(particles:galaxies:deltaTime:simulate:yaw:pitch:reset:resolutionScale:));
- (void)stop;
@end
NS_ASSUME_NONNULL_END
