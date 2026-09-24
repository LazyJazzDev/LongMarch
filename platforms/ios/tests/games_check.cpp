#include <iostream>

#include "games/GameSession.h"

static void Check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main() {
  try {
    MobileGames game;
    game.Resize(5, 7);
    game.Load("OO\n.O\n");
    Check(game.grid.width == 5 && game.grid.height == 7 && game.grid.cells[11] && game.grid.cells[12] &&
              game.grid.cells[17],
          "Pattern is not centered per dimension");
    auto saved = game.Save();
    game.Clear();
    game.Load(saved);
    Check(game.Save() == saved && !game.playing, "Full-grid pattern roundtrip failed");
    game.Resize(4, 4);
    game.Clear();
    game.grid.cells[0] = game.grid.cells[3] = game.grid.cells[12] = 1;
    game.playing = true;
    game.speed = SimulationClock::kLightning;
    game.Tick(0, true);
    Check(game.generation == 1 && game.grid.cells[15], "Periodic lightning step did not wrap");
    game.Clear();
    game.periodic = false;
    game.playing = true;
    game.grid.cells[0] = game.grid.cells[3] = game.grid.cells[12] = 1;
    game.Tick(2, true);
    Check(game.generation == 1 && std::count(game.grid.cells.begin(), game.grid.cells.end(), 1) == 0,
          "Fixed boundaries or one step per frame failed");
    game.playing = false;
    game.Tick(10, true);
    Check(game.generation == 1, "Paused Life advanced");
    auto previous = game.Save();
    bool rejected = false;
    try {
      game.Load("OX\n");
    } catch (...) {
      rejected = true;
    }
    Check(rejected && previous == game.Save(), "Invalid import changed the grid");
    game.Resize(1, 300);
    Check(game.grid.width == 2 && game.grid.height == 200, "Grid range is incorrect");
    game.board.fill(0);
    game.board[0] = game.board[1] = game.board[2] = game.board[3] = 1;
    game.score = 0;
    game.Move(Direction::kLeft);
    Check(game.board[0] == 2 && game.board[1] == 2 && game.score == 8, "2048 merge score or double-merge rule failed");
    game.board.fill(0);
    game.board[0] = 11;
    game.continued = false;
    auto board = game.board;
    game.Move(Direction::kRight);
    Check(board == game.board && game.Won(), "Win did not pause the puzzle");
    game.continued = true;
    game.Move(Direction::kRight);
    Check(board != game.board, "Keep going did not resume");
    game.NewPuzzle();
    Check(game.score == 0 && std::count(game.board.begin(), game.board.end(), 0) == 14,
          "New game must spawn two tiles");
    std::string error;
    Check(AiPlayer::ValidateModel(42, 1000, &error), error.c_str());
    std::cout
        << "PASS mobile games: import/export, boundaries, clock, dimensions, merges, scoring, win, reset and AI model\n";
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
