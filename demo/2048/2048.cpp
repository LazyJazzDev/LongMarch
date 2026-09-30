#include "2048.h"

#include <algorithm>

#include "rounded_rectangle.h"

namespace {
constexpr float kAiSearchBudgetMs = 150.0f;
}  // namespace

TwentyFourEight::TwentyFourEight(const std::string &title, int width, int height, graphics::BackendAPI api)
    : Application(title, width, height, api) {
  GetWindow()->KeyEvent().RegisterCallback([this](int key, int scancode, int action, int mods) {
    if (action != GLFW_PRESS) {
      return;
    }
    // The autoplay takes the game over completely while it runs, so the arrow
    // keys stop answering until the GUI control turns it off again.
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
  ui_ = std::make_unique<snowberg::gui::Context>(Core(), GetWindow(), snowberg::gui::DefaultFont());
  OnWindowSize();
  ResetGame();
  SetAiEnabled(initial_ai_enabled_);
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
  ui_.reset();
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
  LogInfo("2048 autoplay {}", ai_enabled_ ? "started" : "stopped");
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
  if (window_width < window_height) {
    const float board_top = board_to_world_[3].y - block_size * 4.0f;
    board_to_world_[3].y += std::max(0.0f, 238.0f * window_width / 720.0f - board_top);
  }
}

void TwentyFourEight::OnTransitStage() {
  alpha_ = 0.0f;
  game_stage_ = target_game_stage_;
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
        GetWindow()->RequestClose();
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

  std::sort(number_blocks_.begin(), number_blocks_.end(),
            [](const NumberBlock &num_block0, const NumberBlock &num_block1) {
              return num_block0.number < num_block1.number;
            });

  block_renderer_->SetBoardToWorld(board_to_world_);

  float depth_offset = 0.2f + 0.2f / 16.0f;
  for (auto number_block : number_blocks_) {
    number_block.Render(block_renderer_.get(), alpha_, depth_offset);
    depth_offset -= 0.2f / 16.0f;
  }
}

graphics::Image *TwentyFourEight::ComposeUI(graphics::CommandContext *commands, graphics::Image *scene) {
  ui_->BeginFrame();
  const auto size = GetWindow()->GetSize();
  const float margin = 24.0f;
  const bool portrait = size.x < size.y;
  const float width =
      portrait ? std::min(float(size.x) - 2.0f * margin, 600.0f) : std::min(280.0f, std::max(220.0f, size.x * 0.24f));
  ui_->BeginPanel("game", portrait ? (size.x - width) * 0.5f : margin, margin, width, "2048");
  ui_->Heading("Score  " + std::to_string(score_));
  ui_->Text(ai_enabled_ ? "Autoplay is running" : "Use arrow keys to move tiles");
  if (portrait)
    ui_->BeginRow(3);
  if (ui_->Button(ai_enabled_ ? "Stop autoplay" : "Autoplay"))
    SetAiEnabled(!ai_enabled_);
  if (ui_->Button("New game"))
    ResetGame();
  if (ui_->Button("Menu"))
    TransitStage(GameStage::kMenu);
  if (portrait)
    ui_->EndRow();
  ui_->EndPanel();

  if (game_stage_ == GameStage::kMenu) {
    ui_->BeginPanel("menu", std::max(margin, size.x * 0.5f - 160.0f), std::max(margin, size.y * 0.5f - 100.0f), 320.0f,
                    "Game menu");
    if (ui_->Button("Keep going"))
      TransitStage(GameStage::kGameGoing);
    if (ui_->Button("Start again"))
      ResetGame();
    ui_->EndPanel();
  } else if (game_stage_ == GameStage::kGameOver) {
    ui_->BeginPanel("game_over", std::max(margin, size.x * 0.5f - 160.0f), std::max(margin, size.y * 0.5f - 100.0f),
                    320.0f, "Game over");
    ui_->Text("Score  " + std::to_string(score_));
    if (ui_->Button("Try again"))
      ResetGame();
    ui_->EndPanel();
  }
  return ui_->EndFrame(commands, scene);
}
