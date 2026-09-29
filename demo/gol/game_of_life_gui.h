#pragma once
#include "application/animation_var.h"
#include "application/application.h"
#include "application/model.h"
#include "boundary_toggle_button.h"
#include "cell_grid.h"
#include "file_button.h"
#include "grid_size.h"
#include "grid_view.h"
#include "pause_play_button.h"
#include "randomize_button.h"
#include "refresh_button.h"
#include "simulation_clock.h"
#include "size_slider.h"
#include "speed_toggle_button.h"

namespace life_demo {

class GameOfLife : public Application {
 public:
  GameOfLife(const char *title,
             int width,
             int height,
             int cell_grid_width,
             int cell_grid_height,
             graphics::BackendAPI api,
             bool hosted = false);

  ~GameOfLife() override;

  double NextFrameDelay() const override;
  void SetIconOrientation(float radians);
  void SetBottomControlInset(float height_fraction);
  void EnableNativeSizeControls();
  glm::ivec2 TakeSizeControlRequest();
  void SetGridDimension(int axis, int value);

  void ResetFrameClock() {
    last_frame_time_ = last_simulation_time_ = grassland::GetTimeSeconds();
  }

  void ResetAnimationClock() {
    const auto now = grassland::GetTimeSeconds();
    // A new interaction after a timed sleep starts its animation now. Do not
    // reset on every drag event or starve animations with high-rate input.
    if (now - last_frame_time_ > 1.0 / 30.0)
      last_frame_time_ = now;
  }

  // File pickers are asynchronous in native hosts. Buttons and feedback stay shared.
  int HostedFileRequest() const {
    return hosted_file_action_;
  }

  std::string CompleteHostedFile(const std::string &path);

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

  int hosted_file_action_{};
  int hosted_size_axis_{};
  bool hosted_was_playing_{};
  SimulationClock simulation_clock_;
  AnimationVar icon_rotation_{0.0f, AnimationStyle::kPower2};
  GridView grid_view_;
  std::unique_ptr<SizeSlider> width_slider_;
  std::unique_ptr<SizeSlider> height_slider_;
  int requested_width_{};
  int requested_height_{};
  bool sidebar_{true};
  float bottom_control_inset_{};
  bool sliders_were_dragging_{false};
  uint32_t magnify_callback_{};
  uint32_t scroll_callback_{};
  uint32_t key_callback_{};
  uint32_t pan_button_callback_{};
  uint32_t pan_move_callback_{};
  uint32_t pan_focus_callback_{};
  bool panning_{false};
  glm::vec2 pan_cursor_{0.0f};

  std::unique_ptr<PausePlayButton> pause_play_button_;
  std::unique_ptr<SpeedToggleButton> speed_toggle_button_;
  std::unique_ptr<BoundaryToggleButton> boundary_button_;
  std::unique_ptr<RefreshButton> refresh_button_;
  std::unique_ptr<RandomizeButton> randomize_button_;
  std::unique_ptr<FileButton> open_button_;
  std::unique_ptr<FileButton> save_button_;
  FileAction file_action_{FileAction::kNone};
  float file_action_delay_{0.0f};
  std::string file_path_{"life.cells"};

  std::optional<DeviceModel> white_icon_model;
  std::optional<DeviceModel> white_rect_model;

  std::vector<uint8_t> cell_grid_;
  std::vector<uint8_t> initial_cells_;
  std::unique_ptr<CellGrid> cell_input_;
  bool cells_animating_{false};
  float time_total{0.0};
  double last_frame_time_{};
  double last_simulation_time_{};
  float ui_scale_{1.0};
  float random_density_{0.0f};
  uint32_t random_seed_{0};
  bool initial_playing_{false};
  int cell_grid_width_{};
  int cell_grid_height_{};

  float panel_left_{};
  float panel_right_{};
  float panel_top_{};
  float panel_bottom_{};

  float playground_left_{};
  float playground_right_{};
  float playground_top_{};
  float playground_bottom_{};
};

}  // namespace life_demo
