#include "2048_lib.h"

#include <algorithm>
#include <vector>

void update_step(int num_blocks, Block *blocks, Direction direction) {
  // Each line parallel to the movement is processed from the block nearest to
  // the destination wall. A block either slides next to the previous one or
  // merges into it, and a merged block is locked for the rest of the move.
  // Both merged blocks end up with the merged position and number.
  auto line_of = [direction](const Block &block) {
    return (direction == Direction::kLeft || direction == Direction::kRight) ? block.y : block.x;
  };
  // Distance from the destination wall.
  auto depth_of = [direction](const Block &block) {
    switch (direction) {
      case Direction::kUp:
        return 3 - block.y;
      case Direction::kDown:
        return block.y;
      case Direction::kLeft:
        return block.x;
      case Direction::kRight:
      default:
        return 3 - block.x;
    }
  };
  auto place = [direction](Block &block, int line, int depth) {
    switch (direction) {
      case Direction::kUp:
        block.x = line;
        block.y = 3 - depth;
        break;
      case Direction::kDown:
        block.x = line;
        block.y = depth;
        break;
      case Direction::kLeft:
        block.x = depth;
        block.y = line;
        break;
      case Direction::kRight:
        block.x = 3 - depth;
        block.y = line;
        break;
    }
  };

  for (int line = 0; line < 4; line++) {
    std::vector<int> order;
    for (int i = 0; i < num_blocks; i++) {
      if (line_of(blocks[i]) == line) {
        order.push_back(i);
      }
    }
    std::sort(order.begin(), order.end(), [&](int a, int b) { return depth_of(blocks[a]) < depth_of(blocks[b]); });

    int next_depth = 0;
    int last = -1;
    bool last_locked = false;
    for (int index : order) {
      Block &block = blocks[index];
      if (last >= 0 && !last_locked && blocks[last].number == block.number) {
        blocks[last].number *= 2;
        block = blocks[last];
        last_locked = true;
      } else {
        place(block, line, next_depth++);
        last = index;
        last_locked = false;
      }
    }
  }
}
