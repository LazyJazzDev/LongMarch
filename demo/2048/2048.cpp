#include "2048.h"

#include <algorithm>

namespace {

// The score board in its normal state, and the same board while the autoplay
// owns the game. The autoplay palette is a muted brick red of about the same
// lightness as the warm grey it replaces, with a tinted title, so the board
// reads as the same object in a different mood instead of a warning banner.
const glm::vec3 kScoreBoardColor{184.0f / 255.0f, 173.0f / 255.0f, 161.0f / 255.0f};
const glm::vec3 kScoreBoardTitleColor{236.0f / 255.0f, 228.0f / 255.0f, 214.0f / 255.0f};
const glm::vec3 kAiScoreBoardColor{174.0f / 255.0f, 70.0f / 255.0f, 56.0f / 255.0f};
const glm::vec3 kAiScoreBoardTitleColor{240.0f / 255.0f, 196.0f / 255.0f, 178.0f / 255.0f};

// How long the score board takes to fade between the two palettes, and how
// long one autoplay move may think. The search runs on its own thread, so the
// budget only decides how deep it looks, not how smoothly the board moves.
constexpr float kAiThemeSeconds = 0.45f;
constexpr float kAiSearchBudgetMs = 150.0f;

// The clicks that start the autoplay have to be consecutive: a pause longer
// than this window starts the count over.
constexpr float kScoreBoardClickWindow = 1.2f;
constexpr int kScoreBoardClicksToStartAi = 5;

}  // namespace

TwentyFourEight::TwentyFourEight(const std::string &title, int width, int height, graphics::BackendAPI api)
    : Application(title, width, height, api) {
  GetWindow()->KeyEvent().RegisterCallback([this](int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) {
      return;
    }
    // The autoplay takes the game over completely while it runs, so the arrow
    // keys stop answering until the score board turns it off again.
    if (ai_enabled_) {
      return;
    }
    switch (key) {
      case GLFW_KEY_LEFT:
        operation_buffer_ = Direction::kLeft;
        break;
      case GLFW_KEY_RIGHT:
        operation_buffer_ = Direction::kRight;
        break;
      case GLFW_KEY_UP:
        operation_buffer_ = Direction::kUp;
        break;
      case GLFW_KEY_DOWN:
        operation_buffer_ = Direction::kDown;
        break;
      default:
        break;
    }
  });
}

void TwentyFourEight::OnFramebufferResize() {
  OnWindowSize();
}

void TwentyFourEight::CustomOnInit() {
  Application::CustomOnInit();
  clear_color_ = {250.0f / 255.0f, 248.0f / 255.0f, 240.0f / 255.0f, 1.0f};
  float arc_size = 1.0f / 16.0f;
  auto model = GenerateRoundedRectangle(-arc_size, -arc_size, 4.0f + arc_size, 4.0f + arc_size, arc_size,
                                        glm::vec3{184.0f, 173.0f, 161.0f} / 255.0f);
  background_model_ = std::make_unique<DeviceModel>(this, model);
  model = GenerateRoundedRectangle(1.0f / 16.0f, 1.0f / 16.0f, 15.0f / 16.0f, 15.0f / 16.0f, 1.0f / 32.0f,
                                   glm::vec3{202.0f, 192.0f, 181.0f} / 255.0f);
  slot_model_ = std::make_unique<DeviceModel>(this, model);

  font_factory_ = std::make_unique<font::Factory>(FindAssetFile("fonts/ClearSans-Bold-webfont.woff"));
  block_renderer_ = std::make_unique<BlockRenderer>(this, font_factory_.get());
  score_board_ = std::make_unique<NoticeBoard>(this, font_factory_.get(), kScoreBoardTitleColor, glm::vec3{1.0f},
                                               kScoreBoardColor, L"SCORE", L"0");
  score_board_gesture_ = std::make_unique<ScoreBoardGesture>(this, this);
  game_over_bar_ = std::make_unique<TextBar>(this, font_factory_.get(), L"Game Over!", 1.0f,
                                             glm::vec3{117.0f, 110.0f, 102.0f} / 255.0f, glm::vec2{0.0f, 0.0f},
                                             TextBar::AlignMode::kMid);
  game_over_score_board_ =
      std::make_unique<NoticeBoard>(this, font_factory_.get(), glm::vec3{236.0f, 228.0f, 214.0f} / 255.0f,
                                    glm::vec3{1.0f}, glm::vec3{184.0f, 173.0f, 161.0f} / 255.0f, L"SCORE", L"0");
  game_over_button_ = std::make_unique<TextButton>(
      this, font_factory_.get(), glm::vec3{1.0f}, glm::vec3{184.0f, 173.0f, 161.0f} / 255.0f, L"TRY AGAIN",
      [](Application *app) {
        auto app_instance = dynamic_cast<TwentyFourEight *>(app);
        if (app_instance) {
          app_instance->ResetGame();
        } else {
          LogError("The app is not a 2048 instance");
        }
      },
      0.5f);
  menu_bar_ =
      std::make_unique<TextBar>(this, font_factory_.get(), L"Menu", 1.0f, glm::vec3{117.0f, 110.0f, 102.0f} / 255.0f,
                                glm::vec2{0.0f, 0.0f}, TextBar::AlignMode::kMid);
  menu_keep_going_button_ = std::make_unique<TextButton>(
      this, font_factory_.get(), glm::vec3{1.0f}, glm::vec3{184.0f, 173.0f, 161.0f} / 255.0f, L"KEEP GOING",
      [](Application *app) {
        auto app_instance = dynamic_cast<TwentyFourEight *>(app);
        if (app_instance) {
          app_instance->TransitStage(GameStage::kGameGoing);
        } else {
          LogError("The app is not a 2048 instance");
        }
      },
      0.5f);
  menu_new_game_button_ = std::make_unique<TextButton>(
      this, font_factory_.get(), glm::vec3{1.0f}, glm::vec3{184.0f, 173.0f, 161.0f} / 255.0f, L"NEW GAME",
      [](Application *app) {
        auto app_instance = dynamic_cast<TwentyFourEight *>(app);
        if (app_instance) {
          app_instance->ResetGame();
        } else {
          LogError("The app is not a 2048 instance");
        }
      },
      0.5f);

  menu_button_ = std::make_unique<TextButton>(
      this, font_factory_.get(), glm::vec3{1.0f}, glm::vec3{225.0f, 156.0f, 102.0f} / 255.0f, L"MENU",
      [](Application *app) {
        auto app_instance = dynamic_cast<TwentyFourEight *>(app);
        if (app_instance) {
          app_instance->TransitStage(GameStage::kMenu);
        } else {
          LogError("The app is not a 2048 instance");
        }
      },
      0.618f);
  OnWindowSize();
  ResetGame();
  SetAiEnabled(initial_ai_enabled_);
  // A run started with `--ai` shows the theme it would have faded into.
  ai_theme_mix_ = ai_enabled_ ? 1.0f : 0.0f;
  UpdateScoreBoardTheme();
}

void TwentyFourEight::CustomOnUpdate() {
  Application::CustomOnUpdate();

  static auto last_step_time_point = std::chrono::steady_clock::now();
  auto time_point = std::chrono::steady_clock::now();
  auto time_last_turn = static_cast<float>((time_point - last_step_time_point) / std::chrono::microseconds(1)) * 1e-6f;
  last_step_time_point = time_point;
  OnUpdate(time_last_turn);
  OnDraw();
}

void TwentyFourEight::CustomOnClose() {
  // Release resources in reverse order of creation.
  ai_player_.Reset();
  score_board_gesture_.reset();
  menu_button_.reset();
  menu_new_game_button_.reset();
  menu_keep_going_button_.reset();
  menu_bar_.reset();
  game_over_button_.reset();
  game_over_score_board_.reset();
  game_over_bar_.reset();
  score_board_.reset();

  background_model_.reset();
  slot_model_.reset();
  number_blocks_.clear();

  block_renderer_.reset();
  font_factory_.reset();
  Application::CustomOnClose();
}

void TwentyFourEight::SetAiEnabled(bool enabled) {
  ai_enabled_ = enabled;
  // The revision marks the position the strategy is answering about, so a move
  // that was computed before this change is dropped instead of played. A move
  // the player queued by hand is dropped with it: from here the strategy owns
  // the board.
  board_revision_++;
  ai_player_.Reset();
  operation_buffer_.reset();
  score_board_clicks_ = 0;
  score_board_click_timer_ = 0.0f;
  score_board_->UpdateTitleText(ai_enabled_ ? L"AI" : L"SCORE");
  LogInfo("2048 autoplay {}", ai_enabled_ ? "started" : "stopped");
}

void TwentyFourEight::OnScoreBoardClick() {
  if (ai_enabled_) {
    SetAiEnabled(false);
    return;
  }

  score_board_clicks_++;
  score_board_click_timer_ = kScoreBoardClickWindow;
  if (score_board_clicks_ >= kScoreBoardClicksToStartAi) {
    SetAiEnabled(true);
  }
}

// The score board fades between the two palettes instead of switching, so the
// autoplay taking over reads as the board itself changing.
void TwentyFourEight::UpdateScoreBoardTheme() {
  const glm::vec3 title_color = glm::mix(kScoreBoardTitleColor, kAiScoreBoardTitleColor, ai_theme_mix_);
  const glm::vec3 background_color = glm::mix(kScoreBoardColor, kAiScoreBoardColor, ai_theme_mix_);
  score_board_->UpdateTitleColor(title_color);
  score_board_->UpdateBackgroundColor(background_color);
}

AiPlayer::Board TwentyFourEight::SnapshotBoard() const {
  // A position may still hold both blocks of a merge until the motion settles:
  // the larger one is the tile that survives, and the strategy only ever sees
  // the settled board.
  AiPlayer::Board board{};
  for (const auto &number_block : number_blocks_) {
    if (number_block.IsDead() || number_block.x < 0 || number_block.x >= kBoardSize || number_block.y < 0 ||
        number_block.y >= kBoardSize) {
      continue;
    }
    const int cell = BoardCell(number_block.x, number_block.y);
    board[cell] = std::max(board[cell], uint8_t(AiPlayer::RankOf(number_block.number)));
  }
  return board;
}

void TwentyFourEight::OnWindowSize() {
  auto window_width = float(FramebufferSize().x), window_height = float(FramebufferSize().y);
  float title_scale = 1.0f / 4.0f;
  float ui_unit = std::min(window_width, window_height / (1.0f + title_scale)) * 1e-2f;
  float block_size = ui_unit * 20.0f;
  float left = window_width * 0.5f - ui_unit * 50.0f;
  float top = window_height * 0.5f - ui_unit * (1.0f + title_scale) * 50.0f;

  board_to_world_ = glm::mat4{block_size,
                              0.0f,
                              0.0f,
                              0.0f,
                              0.0f,
                              -block_size,
                              0.0f,
                              0.0f,
                              0.0f,
                              0.0f,
                              1.0f,
                              0.0f,
                              window_width * 0.5f - block_size * 2.0f,
                              window_height * 0.5f + 50.0f * title_scale * ui_unit + block_size * 2.0f,
                              0.0f,
                              1.0f};
  logo_to_world_ = glm::mat4{block_size,
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             -block_size,
                             0.0f,
                             0.0f,
                             0.0f,
                             0.0f,
                             1.0f,
                             0.0f,
                             window_width * 0.5f - block_size * 2.125f,
                             top + 50.0f * title_scale * ui_unit + block_size * 0.5f,
                             0.0f,
                             1.0f};
  score_board_->Resize(window_width * 0.5f + ui_unit * 50.0f - ui_unit * 38.75f,
                       top + 50.0f * title_scale * ui_unit + block_size * 0.5f - block_size * (1.0f - 1.0f / 16.0f),
                       window_width * 0.5f + ui_unit * 50.0f - ui_unit * 8.75f,
                       top + 50.0f * title_scale * ui_unit + block_size * 0.5f - block_size * (5.0f / 16.0f),
                       block_size / 32.0f);
  menu_button_->Resize(window_width * 0.5f + ui_unit * 50.0f - ui_unit * 38.75f,
                       top + 50.0f * title_scale * ui_unit + block_size * 0.5f - block_size * (4.0f / 16.0f),
                       window_width * 0.5f + ui_unit * 50.0f - ui_unit * 8.75f,
                       top + 50.0f * title_scale * ui_unit + block_size * 0.5f - block_size * (1.0f / 16.0f),
                       block_size / 32.0f);
  // The click target of the score board gesture is the score board itself.
  score_board_gesture_->Resize(
      window_width * 0.5f + ui_unit * 50.0f - ui_unit * 38.75f,
      top + 50.0f * title_scale * ui_unit + block_size * 0.5f - block_size * (1.0f - 1.0f / 16.0f),
      window_width * 0.5f + ui_unit * 50.0f - ui_unit * 8.75f,
      top + 50.0f * title_scale * ui_unit + block_size * 0.5f - block_size * (5.0f / 16.0f));

  game_over_bar_->Resize(ui_unit * 7.0f, glm::vec2{window_width * 0.5f, window_height * 0.5f - ui_unit * 30.0f});
  game_over_score_board_->Resize(window_width * 0.5f - ui_unit * 15.0f, window_height * 0.5f - ui_unit * 25.0f,
                                 window_width * 0.5f + ui_unit * 15.0f, window_height * 0.5f - ui_unit * 5.0f,
                                 block_size / 32.0f);
  game_over_button_->Resize(window_width * 0.5f - ui_unit * 15.0f, window_height * 0.5f - ui_unit * 3.0f,
                            window_width * 0.5f + ui_unit * 15.0f, window_height * 0.5f + ui_unit * 5.0f,
                            block_size / 32.0f);
  menu_new_game_button_->Resize(window_width * 0.5f - ui_unit * 15.0f, window_height * 0.5f - ui_unit * 0.0f,
                                window_width * 0.5f + ui_unit * 15.0f, window_height * 0.5f + ui_unit * 8.0f,
                                block_size / 32.0f);
  menu_keep_going_button_->Resize(window_width * 0.5f - ui_unit * 15.0f, window_height * 0.5f - ui_unit * 13.0f,
                                  window_width * 0.5f + ui_unit * 15.0f, window_height * 0.5f - ui_unit * 5.0f,
                                  block_size / 32.0f);
  menu_bar_->Resize(ui_unit * 7.0f, glm::vec2{window_width * 0.5f, window_height * 0.5f - ui_unit * 30.0f});
}

void TwentyFourEight::OnTransitStage() {
  alpha_ = 0.0f;
  game_stage_ = target_game_stage_;
  switch (game_stage_) {
    case GameStage::kGameGoing:
      menu_button_->Activate();
      score_board_gesture_->Activate();
      game_over_button_->Deactivate();
      menu_keep_going_button_->Deactivate();
      menu_new_game_button_->Deactivate();
      break;
    case GameStage::kMenu:
      menu_button_->Deactivate();
      score_board_gesture_->Deactivate();
      game_over_button_->Deactivate();
      menu_keep_going_button_->Activate();
      menu_new_game_button_->Activate();
      break;
    case GameStage::kGameOver:
      [&]() {
        auto number_str = std::to_string(score_);
        std::wstring number_str32;
        for (auto c : number_str) {
          number_str32 += char32_t(c);
        }
        game_over_score_board_->UpdateContentText(number_str32);
      }();
      menu_button_->Deactivate();
      score_board_gesture_->Deactivate();
      game_over_button_->Activate();
      menu_keep_going_button_->Deactivate();
      menu_new_game_button_->Deactivate();
      break;
  }
}

void TwentyFourEight::TransitStage(GameStage stage) {
  target_game_stage_ = stage;
}

void TwentyFourEight::ResetGame() {
  // A new game always returns control to the player and cancels pending AI work.
  SetAiEnabled(false);
  number_blocks_.clear();
  random_device_ = std::mt19937(int(std::time(nullptr)));
  GenRandomBlock();
  GenRandomBlock();
  score_ = 0;
  alpha_ = 0.0f;
  UpdateScoreBoard();
  TransitStage(GameStage::kGameGoing);
  operation_buffer_.reset();
}

void TwentyFourEight::OnMove(Direction direction) {
  int dir_x = 0, dir_y = 0;
  switch (direction) {
    case Direction::kUp:
      dir_y = 1;
      break;
    case Direction::kDown:
      dir_y = -1;
      break;
    case Direction::kLeft:
      dir_x = -1;
      break;
    case Direction::kRight:
      dir_x = 1;
      break;
  }
  std::vector<Block> source;
  for (auto &number_block : number_blocks_) {
    source.push_back(number_block.GetBlock());
  }
  std::vector<Block> dest = source;
  update_step(int(dest.size()), dest.data(), direction);

  bool different = false;
  for (size_t i = 0; i < number_blocks_.size(); i++) {
    if (dest[i].x != source[i].x || dest[i].y != source[i].y || dest[i].number != source[i].number) {
      different = true;
      break;
    }
  }

  if (different) {
    for (size_t i = 0; i < number_blocks_.size(); i++) {
      if (source[i].number != dest[i].number) {
        score_ += source[i].number;
        bool is_passive = false;
        for (size_t j = 0; j < number_blocks_.size(); j++) {
          if (i == j) {
            continue;
          }
          if (dest[i].x == dest[j].x && dest[i].y == dest[j].y) {
            if (source[i].x * dir_x + source[i].y * dir_y > source[j].x * dir_x + source[j].y * dir_y) {
              is_passive = true;
            }
          }
        }
        if (is_passive) {
          number_blocks_[i].Move(dest[i].x, dest[i].y);
        } else {
          number_blocks_[i].Merge(dest[i].x, dest[i].y, dest[i].number);
        }
      } else {
        number_blocks_[i].Move(dest[i].x, dest[i].y);
      }
    }
    alpha_ = 0.0f;
    GenRandomBlock();
    board_revision_++;
  }
  UpdateScoreBoard();
}

void TwentyFourEight::AddBlock(int x, int y, int number) {
  number_blocks_.emplace_back(x, y, number);
}

void TwentyFourEight::GenRandomBlock() {
  CellFlags occupied{};
  for (const NumberBlock &number_block : number_blocks_) {
    occupied[BoardCell(number_block.x, number_block.y)] = true;
  }

  // The one spawn rule of the game, shared with the offline autoplay benchmark
  // through game_rules.h: a uniform empty cell, then a 2 or a 4.
  int number = 2;
  const auto cell = PickSpawnCell(random_device_, occupied, &number);
  if (cell.has_value()) {
    AddBlock(cell->first, cell->second, number);
  }
}

void TwentyFourEight::UpdateScoreBoard() {
  auto number_str = std::to_string(score_);
  std::wstring number_str32;
  for (auto c : number_str) {
    number_str32 += char32_t(c);
  }
  score_board_->UpdateContentText(number_str32);
}

bool TwentyFourEight::IsGameOver() {
  if (number_blocks_.size() < 16) {
    return false;
  }
  int number[4][4] = {};
  auto legal_pos = [](int x, int y) { return 0 <= x && x < 4 && 0 <= y && y < 4; };
  for (auto &number_block : number_blocks_) {
    if (!legal_pos(number_block.x, number_block.y)) {
      return true;
    }
    if (number[number_block.x][number_block.y]) {
      return true;
    }
    number[number_block.x][number_block.y] = number_block.number;
  }
  for (int x = 0; x < 3; x++) {
    for (int y = 0; y < 4; y++) {
      int nx = x + 1;
      if (number[x][y] == number[nx][y]) {
        return false;
      }
    }
  }

  for (auto &x : number) {
    for (int y = 0; y < 3; y++) {
      int ny = y + 1;
      if (x[y] == x[ny]) {
        return false;
      }
    }
  }
  return true;
}

void TwentyFourEight::OnUpdate(float t) {
  // The gesture window and the theme fade run on the frame clock; the board
  // animation below runs on its own faster one.
  if (score_board_click_timer_ > 0.0f) {
    score_board_click_timer_ -= t;
    if (score_board_click_timer_ <= 0.0f) {
      score_board_click_timer_ = 0.0f;
      score_board_clicks_ = 0;
    }
  }
  const float theme_target = ai_enabled_ ? 1.0f : 0.0f;
  if (ai_theme_mix_ != theme_target) {
    const float step = t / kAiThemeSeconds;
    ai_theme_mix_ = theme_target > ai_theme_mix_ ? std::min(theme_target, ai_theme_mix_ + step)
                                                 : std::max(theme_target, ai_theme_mix_ - step);
    UpdateScoreBoardTheme();
  }

  if (target_game_stage_ != game_stage_) {
    OnTransitStage();
  }
  if (game_stage_ == GameStage::kGameGoing) {
    t *= 6.0f;
    bool alpha_eq_one = (alpha_ == 1.0f);
    if (1.0f - alpha_ < t) {
      alpha_ = 1.0f;
    } else {
      alpha_ += t;
    }

    if (alpha_ == 1.0f) {
      for (auto &number_block : number_blocks_) {
        number_block.FinishTurn();
      }

      for (size_t i = 0; i < number_blocks_.size(); i++) {
        for (size_t j = 0; j < number_blocks_.size(); j++) {
          if (j == i)
            continue;
          if (number_blocks_[i].x == number_blocks_[j].x && number_blocks_[i].y == number_blocks_[j].y) {
            if (number_blocks_[i].number < number_blocks_[j].number) {
              number_blocks_[i].Kill();
              break;
            } else if (number_blocks_[i].number == number_blocks_[j].number) {
              if (i < j) {
                number_blocks_[i].Kill();
                break;
              }
            }
          }
        }
      }
    }

    size_t live_size = 0;
    for (auto &number_block : number_blocks_) {
      if (!number_block.IsDead()) {
        number_blocks_[live_size++] = number_block;
      }
    }

    while (number_blocks_.size() > live_size) {
      number_blocks_.pop_back();
    }

    if (ai_enabled_) {
      // The strategy plays through the same buffer as the arrow keys: it only
      // ever answers with a direction, and the blocks move through the very
      // same update_step call. The random block generator stays untouched.
      ai_player_.RequestMove(SnapshotBoard(), board_revision_, kAiSearchBudgetMs);
      if (!operation_buffer_.has_value()) {
        if (const auto ai_move = ai_player_.TakeMove(board_revision_)) {
          operation_buffer_ = ai_move;
        }
      }
    }

    while (alpha_ == 1.0f && operation_buffer_.has_value()) {
      auto op = operation_buffer_.value();
      operation_buffer_.reset();
      OnMove(op);
    }

    if (ai_enabled_ && ai_stop_tile_ > 0 && alpha_ == 1.0f) {
      const auto reached = std::max_element(
          number_blocks_.begin(), number_blocks_.end(),
          [](const NumberBlock &left, const NumberBlock &right) { return left.number < right.number; });
      if (reached != number_blocks_.end() && reached->number >= ai_stop_tile_) {
        LogInfo("2048 autoplay reached {}, closing for the screenshot", reached->number);
        glfwSetWindowShouldClose(GetWindow()->GLFWWindow(), GLFW_TRUE);
      }
    }

    if (!alpha_eq_one && alpha_ == 1.0f) {
      if (IsGameOver()) {
        TransitStage(GameStage::kGameOver);
      }
    }
  } else {
    operation_buffer_.reset();
    if (game_stage_ == GameStage::kGameOver) {
      t *= 0.5f;
      if (1.0f - alpha_ < t) {
        alpha_ = 1.0f;
      } else {
        alpha_ += t;
      }
    } else if (game_stage_ == GameStage::kMenu) {
      t *= 2.0f;
      if (1.0f - alpha_ < t) {
        alpha_ = 1.0f;
      } else {
        alpha_ += t;
      }
    }
  }
}

void TwentyFourEight::OnDraw() {
  if (game_stage_ == GameStage::kGameOver) {
    game_over_bar_->Draw();
    game_over_score_board_->Draw();
    game_over_button_->Draw();

    float alpha = std::max((alpha_ - 0.5f) * 2.0f, 0.0f);

    if (alpha == 1.0f) {
      return;
    } else {
      // Fade the overlay in over the game board.
      CaptureSecondFrame(alpha);
    }
  }

  if (game_stage_ == GameStage::kMenu) {
    menu_bar_->Draw();
    menu_keep_going_button_->Draw();
    menu_new_game_button_->Draw();

    if (alpha_ == 1.0f) {
      return;
    } else {
      // Fade the overlay in over the game board.
      CaptureSecondFrame(alpha_);
    }
  }

  DrawModel(background_model_.get(), InstanceInfo{
                                         glm::translate(board_to_world_, glm::vec3{0.0f, 0.0f, 0.8f}),
                                         glm::vec4{1.0f},
                                         glm::uvec4{0},
                                     });

  for (int x = 0; x < 4; x++) {
    for (int y = 0; y < 4; y++) {
      DrawModel(slot_model_.get(), InstanceInfo{
                                       board_to_world_ * glm::mat4{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                                                                   0.0f, 1.0f, 0.0f, x, y, 0.6f, 1.0f},
                                       glm::vec4{1.0f},
                                       glm::uvec4{0},
                                   });
    }
  }
  score_board_->Draw();

  std::sort(number_blocks_.begin(), number_blocks_.end(),
            [](const NumberBlock &num_block0, const NumberBlock &num_block1) {
              return num_block0.number < num_block1.number;
            });

  block_renderer_->SetBoardToWorld(logo_to_world_);
  block_renderer_->Render(2048, 0.0f, 0.0f, 1.0f, 1.0f, 0.2f);
  block_renderer_->SetBoardToWorld(board_to_world_);

  float depth_offset = 0.2f + 0.2f / 16.0f;
  for (auto number_block : number_blocks_) {
    number_block.Render(block_renderer_.get(), alpha_, depth_offset);
    depth_offset -= 0.2f / 16.0f;
  }

  menu_button_->Draw();
}
