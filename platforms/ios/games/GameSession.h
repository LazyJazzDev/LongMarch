#pragma once
#include <algorithm>
#include <sstream>

#include "demo/2048/ai_player.h"
#include "demo/gol/boundary_glider.h"
#include "demo/gol/cells_pattern.h"
#include "demo/gol/game_of_life_lib/game_of_life_lib.h"
#include "demo/gol/simulation_clock.h"

// The native UI owns presentation; rules, file format, clock and AI are shared with desktop.
class MobileGames {
 public:
  MobileGames() {
    NewPuzzle();
  }

  CellsPattern grid{32, 32, std::vector<uint8_t>(1024)};
  bool playing = false, periodic = true, autoplay = false, continued = false;
  int speed = 0, score = 0;
  uint64_t generation = 0, revision = 0;
  AiPlayer::Board board{};
  SimulationClock clock;
  BoundaryGlider glider;
  AiPlayer ai;
  std::mt19937 random{std::random_device{}()};

  void Resize(int width, int height) {
    width = std::clamp(width, 2, 200);
    height = std::clamp(height, 2, 200);
    CellsPattern next{width, height, std::vector<uint8_t>(width * height)};
    for (int y = 0; y < std::min(height, grid.height); ++y)
      for (int x = 0; x < std::min(width, grid.width); ++x)
        next.cells[y * width + x] = grid.cells[y * grid.width + x];
    grid = std::move(next);
  }

  void Randomize() {
    for (auto &cell : grid.cells)
      cell = std::bernoulli_distribution(.25)(random);
    generation = 0;
  }

  void Clear() {
    std::fill(grid.cells.begin(), grid.cells.end(), 0);
    generation = 0;
    playing = false;
  }

  void Load(const std::string &text) {
    if (text.size() > 1024 * 1024)
      throw std::invalid_argument("Pattern file exceeds 1 MiB");
    std::istringstream input(text);
    grid = FitCellsPattern(ParseCellsPattern(input), grid.width, grid.height);
    playing = false;
    generation = 0;
  }

  std::string Save() const {
    std::ostringstream out;
    WriteCellsPattern(out, grid);
    return out.str();
  }

  void Tick(double elapsed, bool life) {
    glider.Update(float(elapsed));
    if (life)
      clock.Advance(std::clamp(elapsed, 0.0, .1), playing, speed, [&] {
        update_step(grid.width, grid.height, grid.cells.data(),
                    periodic ? BoundaryMode::kPeriodic : BoundaryMode::kFixed);
        ++generation;
      });
    else if (autoplay && !Over() && (!Won() || continued)) {
      ai.RequestMove(board, revision, 100);
      if (auto move = ai.TakeMove(revision))
        Move(*move);
    }
  }

  bool Won() const {
    return *std::max_element(board.begin(), board.end()) >= 11;
  }

  bool Over() const {
    AiPlayer::Board next;
    for (auto d : {Direction::kUp, Direction::kDown, Direction::kLeft, Direction::kRight})
      if (AiPlayer::ApplyMove(board, d, &next))
        return false;
    return true;
  }

  void Spawn() {
    CellFlags occupied{};
    for (int i = 0; i < 16; ++i)
      occupied[i] = board[i] != 0;
    int number;
    if (auto cell = PickSpawnCell(random, occupied, &number))
      board[BoardCell(cell->first, cell->second)] = AiPlayer::RankOf(number);
  }

  void NewPuzzle() {
    ai.Reset();
    ++revision;
    board.fill(0);
    score = 0;
    continued = false;
    Spawn();
    Spawn();
  }

  void Move(Direction direction) {
    if (Won() && !continued)
      return;
    AiPlayer::Board next;
    if (!AiPlayer::ApplyMove(board, direction, &next))
      return;
    for (int i = 0; i < 16; ++i)
      score += int(next[i]) * (1 << next[i]) - int(board[i]) * (1 << board[i]);
    board = next;
    Spawn();
    ++revision;
  }
};
