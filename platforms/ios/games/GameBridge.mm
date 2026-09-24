#import "GameBridge.h"
#include "GameSession.h"
@implementation GameController {
  std::unique_ptr<MobileGames> _game;
}

- (instancetype)init {
  if ((self = [super init]))
    _game = std::make_unique<MobileGames>();
  return self;
}

- (NSDictionary *)snapshot {
  auto &g = *_game;
  auto glider = g.glider.ProjectedCells();
  NSMutableArray *tiles = [NSMutableArray array];
  for (int y = 3; y >= 0; --y)
    for (int x = 0; x < 4; ++x) {
      int rank = g.board[BoardCell(x, y)];
      [tiles addObject:@(rank ? 1 << rank : 0)];
    }
  return @{
    @"glider" : [NSData dataWithBytes:glider.data() length:glider.size()],
    @"cells" : [NSData dataWithBytes:g.grid.cells.data() length:g.grid.cells.size()],
    @"width" : @(g.grid.width),
    @"height" : @(g.grid.height),
    @"playing" : @(g.playing),
    @"periodic" : @(g.periodic),
    @"speed" : @(g.speed),
    @"generation" : @(g.generation),
    @"tiles" : tiles,
    @"score" : @(g.score),
    @"ai" : @(g.autoplay),
    @"won" : @(g.Won() && !g.continued),
    @"over" : @(g.Over())
  };
}

- (void)tick:(double)elapsed life:(BOOL)life {
  _game->Tick(elapsed, life);
}

- (void)resizeWidth:(NSInteger)width height:(NSInteger)height {
  _game->Resize(int(width), int(height));
}

- (void)toggleCell:(NSInteger)index {
  if (index >= 0 && index < _game->grid.cells.size())
    _game->grid.cells[index] ^= 1;
}

- (void)action:(NSString *)action value:(NSInteger)value {
  auto &g = *_game;
  if ([action isEqual:@"play"])
    g.playing = !g.playing;
  if ([action isEqual:@"clear"])
    g.Clear();
  if ([action isEqual:@"random"])
    g.Randomize();
  if ([action isEqual:@"boundary"]) {
    g.periodic = !g.periodic;
    g.glider.Start(g.periodic);
  }
  if ([action isEqual:@"speed"])
    g.speed = (g.speed + 1) % 4;
  if ([action isEqual:@"new"])
    g.NewPuzzle();
  if ([action isEqual:@"continue"])
    g.continued = true;
  if ([action isEqual:@"ai"]) {
    g.autoplay = !g.autoplay;
    g.ai.Reset();
    ++g.revision;
  }
  if ([action isEqual:@"move"] && value >= 0 && value < 4)
    g.Move(static_cast<Direction>(value));
}

- (NSString *)loadPattern:(NSString *)text {
  try {
    _game->Load(text.UTF8String);
    return nil;
  } catch (const std::exception &e) {
    return [NSString stringWithUTF8String:e.what()];
  }
}

- (NSString *)savePattern {
  return [NSString stringWithUTF8String:_game->Save().c_str()];
}

@end
