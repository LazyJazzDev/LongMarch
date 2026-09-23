#include "ai_benchmark.h"

#include <algorithm>
#include <optional>
#include <random>

#include "ai_player.h"

namespace {

// A game over this many moves is stuck rather than unlucky.
constexpr int kMoveLimit = 100000;

// The interactive game without the drawing: blocks are moved by update_step,
// settled the way the application settles them between two turns, and new
// blocks come from the very same spawn rule.
class SimulatedGame {
 public:
  explicit SimulatedGame(std::mt19937 &random) : random_(random) {
    Spawn();
    Spawn();
  }

  bool Move(Direction direction) {
    std::vector<Block> moved = blocks_;
    update_step(int(moved.size()), moved.data(), direction);

    bool changed = false;
    for (size_t i = 0; i < blocks_.size(); i++) {
      if (moved[i].x != blocks_[i].x || moved[i].y != blocks_[i].y || moved[i].number != blocks_[i].number) {
        changed = true;
        break;
      }
    }
    if (!changed) {
      return false;
    }

    for (size_t i = 0; i < blocks_.size(); i++) {
      if (blocks_[i].number != moved[i].number) {
        score_ += blocks_[i].number;
      }
    }

    // The block that was merged into another one disappears, leaving one block
    // per cell like the interactive game does before the next turn.
    blocks_.clear();
    for (const Block &block : moved) {
      auto kept = std::find_if(blocks_.begin(), blocks_.end(),
                               [&block](const Block &other) { return other.x == block.x && other.y == block.y; });
      if (kept == blocks_.end()) {
        blocks_.push_back(block);
      } else {
        kept->number = std::max(kept->number, block.number);
      }
    }

    for (const Block &block : blocks_) {
      max_number_ = std::max(max_number_, block.number);
    }

    moves_++;
    Spawn();
    return true;
  }

  [[nodiscard]] AiPlayer::Board Snapshot() const {
    AiPlayer::Board board{};
    for (const Block &block : blocks_) {
      board[BoardCell(block.x, block.y)] = uint8_t(AiPlayer::RankOf(block.number));
    }
    return board;
  }

  [[nodiscard]] int max_number() const {
    return max_number_;
  }

  [[nodiscard]] int score() const {
    return score_;
  }

  [[nodiscard]] int moves() const {
    return moves_;
  }

 private:
  void Spawn() {
    CellFlags occupied{};
    for (const Block &block : blocks_) {
      occupied[BoardCell(block.x, block.y)] = true;
    }

    int number = 2;
    const auto cell = PickSpawnCell(random_, occupied, &number);
    if (cell.has_value()) {
      blocks_.push_back(Block{cell->first, cell->second, number});
      max_number_ = std::max(max_number_, number);
    }
  }

  std::mt19937 &random_;
  std::vector<Block> blocks_;
  int max_number_{0};
  int score_{0};
  int moves_{0};
};

}  // namespace

std::vector<AiBenchmarkGame> RunAiBenchmark(const AiBenchmarkSettings &settings) {
  std::vector<AiBenchmarkGame> results;
  results.reserve(size_t(std::max(settings.games, 0)));

  for (int index = 0; index < settings.games; index++) {
    std::mt19937 random(settings.seed + uint32_t(index));
    SimulatedGame game(random);

    double depth_sum = 0.0;
    double node_sum = 0.0;
    int searches = 0;
    while (game.moves() < kMoveLimit) {
      AiPlayer::SearchStats stats;
      const std::optional<Direction> move = AiPlayer::SearchBestMove(game.Snapshot(), settings.budget_ms, &stats);
      depth_sum += stats.depth;
      node_sum += double(stats.nodes);
      searches++;
      if (!move.has_value() || !game.Move(*move)) {
        break;
      }
    }

    AiBenchmarkGame result;
    result.max_number = game.max_number();
    result.score = game.score();
    result.moves = game.moves();
    if (searches > 0) {
      result.average_depth = depth_sum / searches;
      result.average_nodes = node_sum / searches;
    }
    results.push_back(result);
    if (settings.on_finished) {
      settings.on_finished(result, index);
    }
  }
  return results;
}
