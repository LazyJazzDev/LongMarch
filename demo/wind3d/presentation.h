#pragma once
#include <string>

namespace wind3d {
struct ViewSettings {
  float yaw{-0.85f}, pitch{0.40f}, distance{10.5f};
  bool flow{false}, loads{true}, volume{true}, aircraft{true}, domain{false};
  bool colored_volume{false};
  float volume_density{0.035f}, volume_threshold{6}, volume_step{1};
  std::string web_directory;
  int history_window{2}, history_skip{2};
  bool history{true}, history_demean{}, history_hide_changes{true};
  int streamline_density{2};
  float streamline_width{1.5f};
  float opaque_curvature{0.15f};
  float transparent_curvature{0.01f};
};

}  // namespace wind3d
