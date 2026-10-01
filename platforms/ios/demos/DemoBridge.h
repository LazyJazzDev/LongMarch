#import <Foundation/Foundation.h>
#import <MetalKit/MetalKit.h>
NS_ASSUME_NONNULL_BEGIN
typedef void (^DemoProgress)(double frameSeconds,
                             double gpuMilliseconds,
                             double fps,
                             NSString *device,
                             NSInteger width,
                             NSInteger height,
                             NSInteger frames,
                             NSString *_Nullable error);
@interface DemoRenderer : NSObject <MTKViewDelegate>
- (void)startView:(MTKView *)view
        resources:(NSURL *)resources
             demo:(NSString *)demo
         progress:(DemoProgress)progress NS_SWIFT_NAME(start(view:resources:demo:progress:));
@property(nonatomic, copy, nullable) void (^fileRequest)(NSInteger action);
// bounds is the tapped slider as fractions of the view.
@property(nonatomic, copy, nullable) void (^sizeRequest)(NSInteger axis, NSInteger value, CGRect bounds);
- (void)setGridDimension:(NSInteger)axis value:(NSInteger)value;
- (void)completeFile:(NSString *)path completion:(void (^)(NSString *_Nullable error))completion;
- (void)input:(NSInteger)kind x:(double)x y:(double)y value:(double)value;
- (void)setHDR:(BOOL)hdr;
- (void)setGameIconRotation:(float)radians;
- (void)setGameBottomControlInset:(float)heightFraction;
- (void)setGameCutoutInsetsLeft:(float)left top:(float)top right:(float)right;
- (void)setGameControlExtentLimit:(float)heightFraction;
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
