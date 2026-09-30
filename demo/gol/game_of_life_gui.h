#pragma once
#include "application/animation_var.h"
#include "application/application.h"
#include "application/model.h"
#include "cell_grid.h"
#include "game_of_life_lib/game_of_life_lib.h"
#include "grid_size.h"
#include "grid_view.h"
#include "simulation_clock.h"
#include "snowberg/gui/gui.h"

class GameOfLife : public Application {
 public:
  GameOfLife(const char *title,
             int width,
             int height,
             int cell_grid_width,
             int cell_grid_height,
             graphics::BackendAPI api);

  ~GameOfLife() override;

  // Fills the grid with random live cells, each alive with the given probability.
  void RandomizeCells(float density, uint32_t seed);

  // Requests a random initial grid; applied when the cells are created.
  void SetRandomInitialCells(float density, uint32_t seed) {
    random_density_ = density;
    random_seed_ = seed;
  }

  void SetInitialCells(std::vector<uint8_t> cells);

  void SetInitialPlaying(bool playing) {
    initial_playing_ = playing;
  }

 private:
  void CustomOnInit() override;
  void CustomOnUpdate() override;
  void CustomOnClose() override;
  void OnFramebufferResize() override;
  graphics::Image *ComposeUI(graphics::CommandContext *commands, graphics::Image *scene) override;

  void OnWindowSize();

  void InitCells(int width, int height);
  void LayoutCells();
  void DrawCellGrid();
  void ResizeGrid(int width, int height);
  glm::vec2 CursorPosition() const;
  glm::vec2 FramePosition(glm::dvec2 position) const;
  bool CursorInGrid() const;
  void ZoomGrid(float factor);
  void ScrollGrid(double x, double y);
  enum class FileAction { kNone, kOpen, kSave };
  void RequestFileAction(FileAction action);
  void ProcessFileAction();

  SimulationClock simulation_clock_;
  GridView grid_view_;
  std::unique_ptr<snowberg::gui::Context> ui_;
  int requested_width_{};
  int requested_height_{};
  bool playing_{false};
  int speed_level_{0};
  BoundaryMode boundary_mode_{BoundaryMode::kPeriodic};
  uint32_t magnify_callback_{};
  uint32_t scroll_callback_{};
  uint32_t key_callback_{};
  uint32_t pan_button_callback_{};
  uint32_t pan_move_callback_{};
  uint32_t pan_focus_callback_{};
  bool panning_{false};
  glm::vec2 pan_cursor_{0.0f};

  FileAction file_action_{FileAction::kNone};
  float file_action_delay_{0.0f};
  std::string file_path_{"life.cells"};

  std::optional<DeviceModel> white_rect_model;

  std::vector<uint8_t> cell_grid_;
  std::vector<uint8_t> initial_cells_;
  std::unique_ptr<CellGrid> cell_input_;
  float time_total{0.0};
  double last_frame_time_{};
  float random_density_{0.0f};
  uint32_t random_seed_{0};
  bool initial_playing_{false};
  int cell_grid_width_{};
  int cell_grid_height_{};

  float playground_left_{};
  float playground_right_{};
  float playground_top_{};
  float playground_bottom_{};
};
