#pragma once

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>

#include "2048_lib.h"
#include "game_rules.h"

// Autoplay for the puzzle: an expectimax search over the rules of the game,
// guided by the heuristic the classic 2048 solvers use and driven by a small
// worker thread so a search never blocks a frame. The strategy only ever
// produces a Direction, which the application pushes through the very same
// input buffer as the arrow keys; it never reads or touches the random block
// generator, and its budget only limits how deep the search looks.
class AiPlayer {
 public:
  // One entry per cell holding log2 of the tile value, so 0 marks an empty
  // cell: merging two tiles adds one to the entry, exactly like the game
  // doubles them. A position is therefore one nibble per cell, and the tallest
  // tile that fits is 32768.
  using Board = std::array<uint8_t, kBoardCells>;

  // What a single search cost, used by the offline benchmark.
  struct SearchStats {
    int depth{0};
    uint64_t nodes{0};
    double seconds{0.0};
  };

  AiPlayer();
  ~AiPlayer();

  AiPlayer(const AiPlayer &) = delete;
  AiPlayer &operator=(const AiPlayer &) = delete;

  // Asks the worker for a move on the position identified by `revision`.
  // Requests are deduplicated by that revision, so calling this every frame
  // starts a single search per position.
  void RequestMove(const Board &board, uint64_t revision, float budget_ms);

  // Returns the move computed for `revision` once the worker finished it.
  std::optional<Direction> TakeMove(uint64_t revision);

  // Drops the pending request together with any computed move.
  void Reset();

  // The same search without the worker, blocking until the budget runs out.
  static std::optional<Direction> SearchBestMove(const Board &board, float budget_ms, SearchStats *stats = nullptr);

  // Applies one move to the board the way update_step does, reporting whether
  // anything moved.
  static bool ApplyMove(const Board &board, Direction direction, Board *result);

  // Checks the move model of the search against update_step, the function the
  // game itself moves blocks with.
  static bool ValidateModel(uint32_t seed, int trials, std::string *error);

  // log2 of a tile, capped at the tallest tile the board holds.
  static int RankOf(int number);

 private:
  struct Request {
    Board board{};
    uint64_t revision{0};
    float budget_ms{0.0f};
  };

  void WorkerLoop();

  std::mutex mutex_;
  std::condition_variable condition_;
  std::thread worker_;
  std::optional<Request> pending_;
  std::optional<std::pair<uint64_t, Direction>> result_;
  bool searching_{false};
  uint64_t searching_revision_{0};
  bool quit_{false};
};
