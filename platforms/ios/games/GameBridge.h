#import <Foundation/Foundation.h>
NS_ASSUME_NONNULL_BEGIN
@interface GameController : NSObject
@property(nonatomic, readonly) NSDictionary *snapshot;
- (void)tick:(double)elapsed life:(BOOL)life;
- (void)action:(NSString *)action value:(NSInteger)value;
- (void)resizeWidth:(NSInteger)width height:(NSInteger)height;
- (void)toggleCell:(NSInteger)index;
- (nullable NSString *)loadPattern:(NSString *)text;
- (NSString *)savePattern;
@end
NS_ASSUME_NONNULL_END
