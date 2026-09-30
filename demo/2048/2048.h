#pragma once
#include <array>

#include "2048_lib.h"
#include "ai_player.h"
#include "application/application.h"
#include "block_renderer.h"
#include "number_block.h"
#include "random"
#include "snowberg/gui/gui.h"

enum class GameStage { kGameGoing, kMenu, kGameOver };

class TwentyFourEight : public Application {
 public:
  TwentyFourEight(const std::string &title, int width, int height, graphics::BackendAPI api);
  void TransitStage(GameStage stage);
  void ResetGame();

  // Enables the autoplay before the application runs, which is how the `--ai`
  // option starts the demo with the strategy already playing.
  void SetInitialAiEnabled(bool enabled) {
    initial_ai_enabled_ = enabled;
  }

  // Closes the window once the autoplay has built a block of this value, which
  // is how a screenshot run stops on a board worth capturing. Zero never stops.
  void SetAiStopTile(int tile) {
    ai_stop_tile_ = tile;
  }

 private:
  void CustomOnInit() override;
  void CustomOnUpdate() override;
  void CustomOnClose() override;
  void OnFramebufferResize() override;
  graphics::Image *ComposeUI(graphics::CommandContext *commands, graphics::Image *scene) override;
  void OnUpdate(float t);
  void OnDraw();

  void OnWindowSize();
  void OnTransitStage();
  void SetAiEnabled(bool enabled);
  [[nodiscard]] AiPlayer::Board SnapshotBoard() const;
  void OnMove(Direction direction);
  void AddBlock(int x, int y, int number);
  void GenRandomBlock();
  bool IsGameOver();

  std::unique_ptr<font::Factory> font_factory_;
  std::unique_ptr<snowberg::gui::Context> ui_;

  std::unique_ptr<BlockRenderer> block_renderer_;

  std::unique_ptr<DeviceModel> background_model_;
  std::unique_ptr<DeviceModel> slot_model_;
  int32_t program_transformation_index_{};
  std::vector<NumberBlock> number_blocks_;
  glm::mat4 board_to_world_{1.0f};
  float alpha_{0.0f};
  std::mt19937 random_device_{0};
  int score_{0};

  // The autoplay strategy and the position it was asked about: the revision
  // counts how often the board changed, so a result for an older position is
  // never played.
  AiPlayer ai_player_;
  bool ai_enabled_{false};
  bool initial_ai_enabled_{false};
  int ai_stop_tile_{0};
  uint64_t board_revision_{0};

  int program_texture_uniform_alpha_{0};
  GameStage game_stage_{GameStage::kGameGoing};
  GameStage target_game_stage_{GameStage::kGameGoing};
  std::optional<Direction> operation_buffer_;
};
