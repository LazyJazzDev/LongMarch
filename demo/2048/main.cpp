#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "2048.h"
#include "ai_benchmark.h"
#include "ai_player.h"

namespace {

graphics::BackendAPI ParseBackend(const std::string &name) {
  if (name == "auto") {
    return graphics::BACKEND_API_DEFAULT;
  }
  if (name == "vulkan") {
    return graphics::BACKEND_API_VULKAN;
  }
  if (name == "d3d12") {
    return graphics::BACKEND_API_D3D12;
  }
  if (name == "metal") {
    return graphics::BACKEND_API_METAL;
  }
  throw std::invalid_argument("Unknown backend: " + name);
}

void PrintHelp(const char *executable) {
  std::cout << "Usage: " << executable
            << " [--backend auto|vulkan|d3d12|metal] [--frames N] [--screenshot FILE] [--ai]\n"
            << "       " << executable << " --ai-benchmark GAMES [--ai-budget MS] [--ai-seed SEED]\n"
            << "  --frames N         Exit after N rendered frames\n"
            << "  --screenshot FILE  Save the last frame as PNG on exit\n"
            << "  --ai               Start with the autoplay running\n"
            << "  --ai-stop-at N     With --ai, close once a block of N has been built\n"
            << "  --ai-benchmark N   Let the strategy play N games without a window and report the results\n"
            << "  --ai-budget MS     Search budget of one autoplay move in milliseconds (default 150)\n"
            << "  --ai-seed SEED     Spawn seed of the benchmark games (default 1)\n"
            << "Use the arrow keys to move the blocks.\n";
}

// Plays the game with the autoplay strategy on the rules of the game and
// reports how far it gets, which is how the strength of the strategy is
// checked without watching a window.
int RunBenchmark(int games, float budget_ms, uint32_t seed) {
  std::string error;
  if (!AiPlayer::ValidateModel(seed, 512, &error)) {
    std::cout << "AI model check failed: " << error << '\n';
    return 1;
  }
  std::cout << "AI model check: 512 random boards, all 4 directions and the transposed board match update_step\n";
  std::cout << fmt::format("AI benchmark: {} games, {:.0f} ms per move, seed {}\n", games, budget_ms, seed);
  std::cout.flush();

  AiBenchmarkSettings settings;
  settings.games = games;
  settings.seed = seed;
  settings.budget_ms = budget_ms;
  settings.on_finished = [](const AiBenchmarkGame &game, int index) {
    std::cout << fmt::format("  game {:3}: max {:5}, score {}, {} moves, depth {:.1f}, nodes {:.0f}\n", index + 1,
                             game.max_number, game.score, game.moves, game.average_depth, game.average_nodes);
    // A benchmark game can take minutes, so every finished game has to reach a
    // redirected log immediately instead of sitting in the stream buffer.
    std::cout.flush();
  };

  const std::vector<AiBenchmarkGame> results = RunAiBenchmark(settings);

  int reached_2048 = 0;
  int reached_4096 = 0;
  int best = 0;
  double score_sum = 0.0;
  double move_sum = 0.0;
  double depth_sum = 0.0;
  for (const AiBenchmarkGame &game : results) {
    reached_2048 += game.max_number >= 2048 ? 1 : 0;
    reached_4096 += game.max_number >= 4096 ? 1 : 0;
    best = std::max(best, game.max_number);
    score_sum += game.score;
    move_sum += game.moves;
    depth_sum += game.average_depth;
  }

  const double games_count = double(results.size());
  std::cout << fmt::format(
      "  summary: best tile {}, 2048 reached {:.0f}% ({}/{}), 4096 reached {:.0f}% ({}/{}), average score {:.0f}, "
      "average moves {:.0f}, average search depth {:.1f}\n",
      best, 100.0 * reached_2048 / games_count, reached_2048, results.size(), 100.0 * reached_4096 / games_count,
      reached_4096, results.size(), score_sum / games_count, move_sum / games_count, depth_sum / games_count);
  std::cout.flush();
  return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
  try {
    int frames = 0;
    auto api = graphics::BACKEND_API_DEFAULT;
    std::string screenshot;
    int benchmark_games = 0;
    float ai_budget_ms = 150.0f;
    uint32_t seed = 1;
    bool autoplay = false;
    int stop_tile = 0;

    for (int i = 1; i < argc; i++) {
      const std::string option = argv[i];
      if (option == "--help") {
        PrintHelp(argv[0]);
        return 0;
      } else if (option == "--backend" && i + 1 < argc) {
        api = ParseBackend(argv[++i]);
      } else if (option == "--frames" && i + 1 < argc) {
        frames = std::stoi(argv[++i]);
      } else if (option == "--screenshot" && i + 1 < argc) {
        screenshot = argv[++i];
      } else if (option == "--ai") {
        autoplay = true;
      } else if (option == "--ai-stop-at" && i + 1 < argc) {
        stop_tile = std::stoi(argv[++i]);
      } else if (option == "--ai-benchmark" && i + 1 < argc) {
        benchmark_games = std::stoi(argv[++i]);
      } else if (option == "--ai-budget" && i + 1 < argc) {
        ai_budget_ms = std::stof(argv[++i]);
      } else if (option == "--ai-seed" && i + 1 < argc) {
        seed = uint32_t(std::stoul(argv[++i]));
      } else {
        throw std::invalid_argument("Unknown or incomplete option: " + option);
      }
    }

    if (benchmark_games > 0) {
      return RunBenchmark(benchmark_games, ai_budget_ms, seed);
    }

    TwentyFourEight application("2048", 720, 960, api);
    application.SetScreenshotPath(screenshot);
    application.SetInitialAiEnabled(autoplay);
    application.SetAiStopTile(stop_tile);
    application.Run(frames);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
