#include "ai_player.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Bits = uint64_t;

// The search works on a bitboard: every cell is one nibble holding the tile
// exponent, so a merge is a nibble increment and a whole board is one word.
// A nibble saturates at the tallest tile the model holds, which is far past the
// end of a game in practice.
constexpr int kMaxRank = 15;
constexpr int kRowCount = 1 << 16;
constexpr Bits kCellMask = 0xFULL;
constexpr Bits kRowValueMask = 0xFFFFULL;

// Heuristic weights in the shape the classic expectimax players use. The score
// rewards free cells and merge potential, penalizes lines that are not
// monotonic, and prefers large tiles, which keeps the board ordered around one
// corner instead of chasing the current score.
constexpr float kMonotonicityPower = 4.0f;
constexpr float kMonotonicityWeight = 47.0f;
constexpr float kSumPower = 3.5f;
constexpr float kSumWeight = 11.0f;
constexpr float kMergeWeight = 700.0f;
constexpr float kEmptyWeight = 270.0f;

// Spawn branches that carry less probability than this are replaced by the
// heuristic: what they could contribute is irrelevant next to the number of
// branches they would add.
// A position without a legal move is over, and no heuristic value of a tidy
// full board may make it look attractive.
constexpr float kDeadBoardValue = -1.0e9f;

constexpr float kProbabilityCutoff = 1.0e-6f;
constexpr float kTwoProbability = 0.9f;
constexpr float kFourProbability = 0.1f;
constexpr int kMaxDepth = 16;

constexpr Direction kDirections[4] = {Direction::kUp, Direction::kDown, Direction::kLeft, Direction::kRight};

uint16_t g_move_left[kRowCount];
uint16_t g_reverse_row[kRowCount];
float g_row_heuristic[kRowCount];
uint8_t g_empty_cells_in_byte[256];
uint8_t g_empty_cells_in_row[kRowCount];
uint8_t g_row_has_merge[kRowCount];
float g_rank_sum[kBoardCells + 1];
float g_rank_monotonicity[kBoardCells + 1];
std::once_flag g_tables_once;

Bits ToBits(const AiPlayer::Board &board) {
  Bits bits = 0;
  for (int cell = 0; cell < kBoardCells; cell++) {
    bits |= Bits(board[cell]) << (4 * cell);
  }
  return bits;
}

AiPlayer::Board ToBoard(Bits bits) {
  AiPlayer::Board board{};
  for (int cell = 0; cell < kBoardCells; cell++) {
    board[cell] = uint8_t((bits >> (4 * cell)) & kCellMask);
  }
  return board;
}

inline uint16_t RowAt(Bits board, int index) {
  return uint16_t((board >> (16 * index)) & kRowValueMask);
}

inline Bits MoveLeft(Bits board) {
  Bits moved = 0;
  for (int index = 0; index < kBoardSize; index++) {
    moved |= Bits(g_move_left[RowAt(board, index)]) << (16 * index);
  }
  return moved;
}

inline Bits MoveRight(Bits board) {
  Bits moved = 0;
  for (int index = 0; index < kBoardSize; index++) {
    const uint16_t row = RowAt(board, index);
    moved |= Bits(g_reverse_row[g_move_left[g_reverse_row[row]]]) << (16 * index);
  }
  return moved;
}

inline Bits Transpose(Bits board) {
  const Bits a1 = board & 0xF0F00F0FF0F00F0FULL;
  const Bits a2 = board & 0x0000F0F00000F0F0ULL;
  const Bits a3 = board & 0x0F0F00000F0F0000ULL;
  const Bits a = a1 | (a2 << 12) | (a3 >> 12);
  const Bits b1 = a & 0xFF00FF0000FF00FFULL;
  const Bits b2 = a & 0x00FF00FF00000000ULL;
  const Bits b3 = a & 0x00000000FF00FF00ULL;
  return b1 | (b2 >> 24) | (b3 << 24);
}

// Vertical moves are horizontal moves on the transposed board, where the column
// of the board becomes the row of the transposed board.
inline Bits MoveBoard(Bits board, Direction direction) {
  switch (direction) {
    case Direction::kUp:
      return Transpose(MoveRight(Transpose(board)));
    case Direction::kDown:
      return Transpose(MoveLeft(Transpose(board)));
    case Direction::kLeft:
      return MoveLeft(board);
    case Direction::kRight:
    default:
      return MoveRight(board);
  }
}

inline int CountEmptyCells(Bits board) {
  int empty = 0;
  for (int byte = 0; byte < 8; byte++) {
    empty += g_empty_cells_in_byte[(board >> (8 * byte)) & 0xFFULL];
  }
  return empty;
}

inline int RowsEmptyCells(Bits board) {
  return g_empty_cells_in_row[RowAt(board, 0)] + g_empty_cells_in_row[RowAt(board, 1)] +
         g_empty_cells_in_row[RowAt(board, 2)] + g_empty_cells_in_row[RowAt(board, 3)];
}

inline int RowsWithMerge(Bits board) {
  return g_row_has_merge[RowAt(board, 0)] + g_row_has_merge[RowAt(board, 1)] + g_row_has_merge[RowAt(board, 2)] +
         g_row_has_merge[RowAt(board, 3)];
}

inline float RowsHeuristic(Bits board) {
  return g_row_heuristic[RowAt(board, 0)] + g_row_heuristic[RowAt(board, 1)] + g_row_heuristic[RowAt(board, 2)] +
         g_row_heuristic[RowAt(board, 3)];
}

inline float BoardHeuristic(Bits board) {
  return RowsHeuristic(board) + RowsHeuristic(Transpose(board));
}

// The board is over when it is full and neither rows nor columns hold a pair of
// equal tiles.
inline bool IsDeadBoard(Bits board) {
  return RowsEmptyCells(board) == 0 && RowsWithMerge(board) + RowsWithMerge(Transpose(board)) == 0;
}

// A search leaf: a finished board is worthless, a playable one is judged by the
// heuristic.
inline float LeafValue(Bits board) {
  return IsDeadBoard(board) ? kDeadBoardValue : BoardHeuristic(board);
}

void InitTables() {
  for (int byte = 0; byte < 256; byte++) {
    g_empty_cells_in_byte[byte] = uint8_t((byte & 0xF ? 0 : 1) + (byte >> 4 ? 0 : 1));
  }
  for (int rank = 0; rank <= kBoardCells; rank++) {
    g_rank_sum[rank] = std::pow(float(rank), kSumPower);
    g_rank_monotonicity[rank] = std::pow(float(rank), kMonotonicityPower);
  }

  for (int row = 0; row < kRowCount; row++) {
    const int rank[4] = {row & 0xF, (row >> 4) & 0xF, (row >> 8) & 0xF, (row >> 12) & 0xF};
    g_reverse_row[row] = uint16_t((rank[0] << 12) | (rank[1] << 8) | (rank[2] << 4) | rank[3]);

    int merged[4] = {0, 0, 0, 0};
    int count = 0;
    bool locked = false;
    for (int i = 0; i < 4; i++) {
      if (rank[i] == 0) {
        continue;
      }
      if (count > 0 && merged[count - 1] == rank[i] && !locked) {
        merged[count - 1] = rank[i] + 1;
        locked = true;
      } else {
        merged[count++] = rank[i];
        locked = false;
      }
    }
    g_move_left[row] = uint16_t(merged[0] | (merged[1] << 4) | (merged[2] << 8) | (merged[3] << 12));

    int empty = 0;
    int merges = 0;
    float sum = 0.0f;
    float rising = 0.0f;
    float falling = 0.0f;
    int previous = 0;
    int run = 0;
    for (int i = 0; i < 4; i++) {
      const int tile = rank[i];
      sum += g_rank_sum[tile];
      if (tile == 0) {
        empty++;
      } else {
        if (previous == tile) {
          run++;
        } else if (run > 0) {
          merges += 1 + run;
          run = 0;
        }
        previous = tile;
        if (i > 0 && rank[i - 1] == tile) {
          g_row_has_merge[row] = 1;
        }
      }
      // Empty cells take part in the comparison as a tile of rank zero: that is
      // what pushes the largest tile towards a corner instead of the middle.
      if (i > 0) {
        if (rank[i - 1] > tile) {
          falling += g_rank_monotonicity[rank[i - 1]] - g_rank_monotonicity[tile];
        } else {
          rising += g_rank_monotonicity[tile] - g_rank_monotonicity[rank[i - 1]];
        }
      }
    }
    if (run > 0) {
      merges += 1 + run;
    }
    g_empty_cells_in_row[row] = uint8_t(empty);
    // The sum enters the score with a negative sign: merging two tiles of rank
    // r into one of rank r + 1 lowers r ^ 3.5 + r ^ 3.5 to (r + 1) ^ 3.5, so
    // consolidating large tiles is what this term rewards.
    g_row_heuristic[row] = kEmptyWeight * float(empty) + kMergeWeight * float(merges) -
                           kMonotonicityWeight * std::min(rising, falling) - kSumWeight * sum;
  }
}

// Expectimax over the two kinds of nodes of the puzzle: the player picks a
// move, then the game drops a 2 or a 4 into one of the free cells. Depths count
// spawn layers, and an exhausted budget aborts the walk.
class Expectimax {
 public:
  Expectimax(Clock::time_point begin, float budget_ms)
      : deadline_(begin +
                  std::chrono::duration_cast<Clock::duration>(std::chrono::duration<float, std::milli>(budget_ms))) {
  }

  float MaxValue(Bits board, int depth, float probability) {
    if (depth <= 0 || probability < kProbabilityCutoff) {
      return LeafValue(board);
    }
    ChargeNode();
    if (timed_out_) {
      return 0.0f;
    }

    float best = -std::numeric_limits<float>::infinity();
    for (Direction direction : kDirections) {
      const Bits moved = MoveBoard(board, direction);
      if (moved == board) {
        continue;
      }
      best = std::max(best, ChanceValue(moved, depth, probability));
      if (timed_out_) {
        return 0.0f;
      }
    }
    return best == -std::numeric_limits<float>::infinity() ? kDeadBoardValue : best;
  }

  float ChanceValue(Bits board, int depth, float probability) {
    if (depth <= 0 || probability < kProbabilityCutoff) {
      return LeafValue(board);
    }
    ChargeNode();
    if (timed_out_) {
      return 0.0f;
    }

    const int empty = CountEmptyCells(board);
    if (empty == 0) {
      return LeafValue(board);
    }

    const float two_probability = kTwoProbability / float(empty);
    const float four_probability = kFourProbability / float(empty);
    float total = 0.0f;
    for (int cell = 0; cell < kBoardCells; cell++) {
      if ((board >> (4 * cell)) & kCellMask) {
        continue;
      }
      const Bits occupied = board | (Bits(1) << (4 * cell));
      total += kTwoProbability * MaxValue(occupied, depth - 1, probability * two_probability);
      if (timed_out_) {
        return 0.0f;
      }
      total +=
          kFourProbability * MaxValue(occupied | (Bits(1) << (4 * cell)), depth - 1, probability * four_probability);
      if (timed_out_) {
        return 0.0f;
      }
    }
    return total / float(empty);
  }

  [[nodiscard]] bool OutOfTime() const {
    return timed_out_;
  }

  [[nodiscard]] uint64_t Nodes() const {
    return nodes_;
  }

 private:
  void ChargeNode() {
    nodes_++;
    if ((nodes_ & 0x3F) == 0 && Clock::now() >= deadline_) {
      timed_out_ = true;
    }
  }

  Clock::time_point deadline_;
  bool timed_out_{false};
  uint64_t nodes_{0};
};

// A cheap one-ply choice, used when the budget does not even cover one full
// expectimax layer, so the strategy always has something to play.
std::optional<Direction> GreedyMove(Bits board) {
  std::optional<Direction> best;
  float best_value = -std::numeric_limits<float>::infinity();
  for (Direction direction : kDirections) {
    const Bits moved = MoveBoard(board, direction);
    if (moved == board) {
      continue;
    }
    const float value = BoardHeuristic(moved);
    if (value > best_value) {
      best_value = value;
      best = direction;
    }
  }
  return best;
}

// The move as the game computes it: update_step on the very same blocks.
bool ReferenceMove(const AiPlayer::Board &board, Direction direction, AiPlayer::Board *result) {
  std::vector<Block> blocks;
  for (int cell = 0; cell < kBoardCells; cell++) {
    if (board[cell] == 0) {
      continue;
    }
    blocks.push_back(Block{cell % kBoardSize, cell / kBoardSize, 1 << board[cell]});
  }
  update_step(int(blocks.size()), blocks.data(), direction);

  AiPlayer::Board moved{};
  for (const Block &block : blocks) {
    const int cell = BoardCell(block.x, block.y);
    moved[cell] = uint8_t(std::max<int>(moved[cell], AiPlayer::RankOf(block.number)));
  }
  *result = moved;
  return moved != board;
}

}  // namespace

AiPlayer::AiPlayer() {
  worker_ = std::thread([this] { WorkerLoop(); });
}

AiPlayer::~AiPlayer() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    quit_ = true;
  }
  condition_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
}

void AiPlayer::RequestMove(const Board &board, uint64_t revision, float budget_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (searching_ && searching_revision_ == revision) {
    return;
  }
  if (pending_.has_value() && pending_->revision == revision) {
    return;
  }
  if (result_.has_value() && result_->first == revision) {
    return;
  }
  pending_ = Request{board, revision, budget_ms};
  condition_.notify_all();
}

std::optional<Direction> AiPlayer::TakeMove(uint64_t revision) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!result_.has_value() || result_->first != revision) {
    return std::nullopt;
  }
  const Direction move = result_->second;
  result_.reset();
  return move;
}

void AiPlayer::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_.reset();
  result_.reset();
}

void AiPlayer::WorkerLoop() {
  std::unique_lock<std::mutex> lock(mutex_);
  while (true) {
    condition_.wait(lock, [this] { return quit_ || pending_.has_value(); });
    if (quit_) {
      return;
    }
    const Request request = *pending_;
    pending_.reset();
    searching_ = true;
    searching_revision_ = request.revision;
    lock.unlock();

    const std::optional<Direction> move = SearchBestMove(request.board, request.budget_ms);

    lock.lock();
    searching_ = false;
    if (move.has_value()) {
      result_ = std::make_pair(request.revision, *move);
    }
  }
}

std::optional<Direction> AiPlayer::SearchBestMove(const Board &board, float budget_ms, SearchStats *stats) {
  std::call_once(g_tables_once, InitTables);

  const Bits root = ToBits(board);
  const Clock::time_point begin = Clock::now();
  Expectimax search(begin, budget_ms);

  // Iterative deepening: every completed layer replaces the previous answer, so
  // an interrupted search still leaves a decision that is at least as deep as
  // the greedy one.
  std::optional<Direction> best = GreedyMove(root);
  int best_depth = 0;
  for (int depth = 1; depth <= kMaxDepth; depth++) {
    std::optional<Direction> candidate;
    float candidate_value = -std::numeric_limits<float>::infinity();
    for (Direction direction : kDirections) {
      const Bits moved = MoveBoard(root, direction);
      if (moved == root) {
        continue;
      }
      const float value = search.ChanceValue(moved, depth, 1.0f);
      if (search.OutOfTime()) {
        break;
      }
      if (value > candidate_value) {
        candidate_value = value;
        candidate = direction;
      }
    }
    if (search.OutOfTime() || !candidate.has_value()) {
      break;
    }
    best = candidate;
    best_depth = depth;
  }

  if (stats) {
    stats->depth = best_depth;
    stats->nodes = search.Nodes();
    stats->seconds = std::chrono::duration<double>(Clock::now() - begin).count();
  }
  return best;
}

bool AiPlayer::ApplyMove(const Board &board, Direction direction, Board *result) {
  std::call_once(g_tables_once, InitTables);

  const Bits bits = ToBits(board);
  const Bits moved = MoveBoard(bits, direction);
  *result = ToBoard(moved);
  return moved != bits;
}

bool AiPlayer::ValidateModel(uint32_t seed, int trials, std::string *error) {
  std::call_once(g_tables_once, InitTables);
  std::mt19937 random(seed);

  // The transposed board is what turns horizontal rows into vertical ones; it
  // has to agree with the plain cell by cell transposition.
  for (int trial = 0; trial < trials; trial++) {
    Bits bits = 0;
    for (int cell = 0; cell < kBoardCells; cell++) {
      bits |= Bits(random() % (kBoardCells + 1)) << (4 * cell);
    }
    Bits expected = 0;
    for (int cell = 0; cell < kBoardCells; cell++) {
      const int x = cell % kBoardSize;
      const int y = cell / kBoardSize;
      expected |= ((bits >> (4 * cell)) & kCellMask) << (4 * BoardCell(y, x));
    }
    if (Transpose(bits) != expected) {
      *error = "Transpose disagrees with the cell by cell transposition";
      return false;
    }
  }

  for (int trial = 0; trial < trials; trial++) {
    Board board{};
    const int blocks = 1 + int(random() % kBoardCells);
    for (int i = 0; i < blocks; i++) {
      board[random() % kBoardCells] = uint8_t(1 + random() % (kBoardCells - 2));
    }

    for (Direction direction : kDirections) {
      Board reference{};
      const bool reference_moved = ReferenceMove(board, direction, &reference);
      Board computed{};
      const bool computed_moved = ApplyMove(board, direction, &computed);
      if (reference_moved != computed_moved || reference != computed) {
        *error = "The search moves the board differently from update_step";
        return false;
      }
    }
  }
  return true;
}

int AiPlayer::RankOf(int number) {
  int rank = 0;
  while ((1 << rank) < number && rank < kMaxRank) {
    rank++;
  }
  return rank;
}
