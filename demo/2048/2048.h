#pragma once
#include <array>

#include "2048_lib.h"
#include "ai_player.h"
#include "application/application.h"
#include "application/button.h"
#include "application/text_bar.h"
#include "block_renderer.h"
#include "notice_board.h"
#include "number_block.h"
#include "random"
#include "text_button.h"

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
  void OnUpdate(float t);
  void OnDraw();

  // Five consecutive clicks on the score board start the autoplay, and one
  // click on a running autoplay stops it. The gesture is a click target over
  // the score board, not a control of its own: it has no hover look and draws
  // nothing, so the board keeps the layout it always had.
  class ScoreBoardGesture : public Button {
   public:
    ScoreBoardGesture(Application *app, TwentyFourEight *owner) : Button(app, 0.0f, 0.0f, 1.0f, 1.0f), owner_(owner) {
    }

    void OnClick() override {
      owner_->OnScoreBoardClick();
    }

    void OnStateChange(int state) override {
    }

   private:
    TwentyFourEight *owner_;
  };

  void OnWindowSize();
  void OnTransitStage();
  void SetAiEnabled(bool enabled);
  void OnScoreBoardClick();
  void UpdateScoreBoardTheme();
  [[nodiscard]] AiPlayer::Board SnapshotBoard() const;
  void OnMove(Direction direction);
  void AddBlock(int x, int y, int number);
  void GenRandomBlock();
  void UpdateScoreBoard();
  bool IsGameOver();

  std::unique_ptr<font::Factory> font_factory_;

  std::unique_ptr<BlockRenderer> block_renderer_;

  std::unique_ptr<DeviceModel> background_model_;
  std::unique_ptr<DeviceModel> slot_model_;
  int32_t program_transformation_index_{};
  std::unique_ptr<NoticeBoard> score_board_;
  std::vector<NumberBlock> number_blocks_;
  glm::mat4 board_to_world_{1.0f};
  glm::mat4 logo_to_world_{1.0f};
  float alpha_{0.0f};
  std::mt19937 random_device_{0};
  int score_{0};

  std::unique_ptr<TextBar> game_over_bar_;
  std::unique_ptr<NoticeBoard> game_over_score_board_;
  std::unique_ptr<TextButton> game_over_button_;

  std::unique_ptr<TextBar> menu_bar_;
  std::unique_ptr<TextButton> menu_keep_going_button_;
  std::unique_ptr<TextButton> menu_new_game_button_;

  std::unique_ptr<TextButton> menu_button_;
  std::unique_ptr<ScoreBoardGesture> score_board_gesture_;

  // The autoplay strategy and the position it was asked about: the revision
  // counts how often the board changed, so a result for an older position is
  // never played.
  AiPlayer ai_player_;
  bool ai_enabled_{false};
  bool initial_ai_enabled_{false};
  int ai_stop_tile_{0};
  uint64_t board_revision_{0};

  // Clicks on the score board that are still close enough together to count as
  // one gesture, and how far the score board has shifted to the autoplay
  // theme: zero keeps the original colors, one is fully red.
  int score_board_clicks_{0};
  float score_board_click_timer_{0.0f};
  float ai_theme_mix_{0.0f};

  int program_texture_uniform_alpha_{0};
  GameStage game_stage_{GameStage::kGameGoing};
  GameStage target_game_stage_{GameStage::kGameGoing};
  std::optional<Direction> operation_buffer_;
};
