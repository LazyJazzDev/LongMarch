#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <ostream>
#include <thread>

#include "aircraft.h"
#include "presentation.h"

namespace wind3d::web {
inline bool Write(const std::filesystem::path &path, const std::string &bytes) {
  auto temporary = path.string() + ".tmp";
  std::ofstream file(temporary, std::ios::binary);
  file.write(bytes.data(), bytes.size());
  file.close();
  if (!file)
    return false;
  std::error_code error;
  for (int attempt = 0; attempt < 50; ++attempt) {
    std::filesystem::rename(temporary, path, error);
    if (!error)
      break;
    // Windows readers without FILE_SHARE_DELETE can briefly hold the old inode.
    if (error != std::errc::permission_denied)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(4));
  }
  if (error)
    std::cerr << "Snapshot publish " << path << ": " << error.message() << '\n';
  return !error;
}

inline void Vector(std::ostream &s, wind3d::Vec3 v) {
  s << '[' << v.x << ',' << v.y << ',' << v.z << ']';
}

inline void Controls(std::ostream &s, const wind3d::Controls &c) {
  s << "{\"speed\":" << c.speed << ",\"angle\":" << c.angle << ",\"flap\":" << c.flap << ",\"aileron\":" << c.aileron
    << ",\"elevator\":" << c.elevator << ",\"rudder\":" << c.rudder
    << ",\"tail_incidence\":" << c.airframe.tail_incidence << ",\"winglets\":" << int(c.airframe.winglets)
    << ",\"linked_flaps\":" << int(c.wing_controls.linked_flaps) << ",\"outer_flap\":" << c.wing_controls.outer_flap
    << ",\"cruise_camber\":" << c.wing_controls.cruise_camber << ",\"aileron_droop\":" << c.wing_controls.aileron_droop
    << ",\"spoiler\":" << c.wing_controls.spoiler << ",\"spoiler_roll\":" << c.wing_controls.spoiler_roll
    << ",\"linked_spoiler_roll\":" << int(c.wing_controls.linked_spoiler_roll) << ",\"slat\":" << c.wing_controls.slat
    << ",\"linked_slats\":" << int(c.wing_controls.linked_slats) << ",\"wing_incidence\":" << c.airframe.wing_incidence
    << '}';
}

inline bool Same(const wind3d::Controls &a, const wind3d::Controls &b) {
  return a.speed == b.speed && wind3d::SameAircraftGeometry(a, b);
}

struct Command {
  uint64_t sequence{}, reset{};
  bool paused{};
  int density{2};
  wind3d::Controls controls;
  ViewSettings view;
  bool has_view{};
};

inline bool ReadCommand(const std::filesystem::path &path, Command &command) {
  std::ifstream input(path);
  int pause = 0;
  auto &controls = command.controls;
  if (!(input >> command.sequence >> command.reset >> pause >> controls.speed >> controls.angle >> controls.flap >>
        controls.aileron >> controls.elevator >> controls.rudder))
    return false;
  command.paused = pause != 0;
  int &requested_density = command.density;
  input >> requested_density;
  bool extension_valid = true;
  std::string extension;
  if (input >> extension) {
    int winglets = 1, linked_flaps = 1, linked_roll = 0, linked_slats = 0;
    auto &w = controls.wing_controls;
    extension_valid =
        (extension == "wing-v1" || extension == "wing-v2") &&
        bool(input >> controls.airframe.tail_incidence >> winglets >> linked_flaps >> w.outer_flap >> w.cruise_camber >>
             w.aileron_droop >> w.spoiler >> w.spoiler_roll >> linked_roll >> w.slat >> linked_slats);
    if (extension == "wing-v2")
      extension_valid = extension_valid && bool(input >> controls.airframe.wing_incidence);
    auto flag = [](int value) { return value == 0 || value == 1; };
    extension_valid &= flag(winglets) && flag(linked_flaps) && flag(linked_roll) && flag(linked_slats);
    controls.airframe.winglets = winglets != 0;
    w.linked_flaps = linked_flaps != 0;
    w.linked_spoiler_roll = linked_roll != 0;
    w.linked_slats = linked_slats != 0;
  }
  if (input >> extension) {
    int flow, loads, aircraft, domain, volume, colored;
    auto &v = command.view;
    command.has_view = extension == "view-v1" &&
                       bool(input >> flow >> loads >> aircraft >> domain >> volume >> v.streamline_width >>
                            v.transparent_curvature >> v.opaque_curvature >> v.yaw >> v.pitch >> v.distance >>
                            colored >> v.volume_density >> v.volume_threshold >> v.volume_step);
    if (!command.has_view || flow < 0 || flow > 1 || loads < 0 || loads > 1 || aircraft < 0 || aircraft > 1 ||
        domain < 0 || domain > 1 || volume < 0 || volume > 1 || colored < 0 || colored > 1 ||
        !(v.streamline_width >= .75f && v.streamline_width <= 4 && v.transparent_curvature >= 0 &&
          v.opaque_curvature >= v.transparent_curvature + .001f && v.opaque_curvature <= 10 && v.yaw >= -3.15f &&
          v.yaw <= 3.15f && v.pitch >= -1.55f && v.pitch <= 1.55f && v.distance >= .6f && v.distance <= 40 &&
          v.volume_density >= .01f && v.volume_density <= .6f && v.volume_threshold >= 0 && v.volume_threshold <= 60 &&
          v.volume_step >= .5f && v.volume_step <= 2))
      return false;
    v.flow = flow;
    v.loads = loads;
    v.aircraft = aircraft;
    v.domain = domain;
    v.volume = volume;
    v.colored_volume = colored;
    if (input >> extension) {
      int history, demean, hide;
      if (extension != "charts-v1" || !(input >> v.history_window >> v.history_skip >> history >> demean >> hide) ||
          v.history_window < 0 || v.history_window > 5 || v.history_skip < 0 || v.history_skip > 4 || history < 0 ||
          history > 1 || demean < 0 || demean > 1 || hide < 0 || hide > 1)
        return false;
      v.history = history;
      v.history_demean = demean;
      v.history_hide_changes = hide;
    }
  }
  return extension_valid && wind3d::ValidAirframe(controls.airframe) &&
         wind3d::ValidWingControls(controls.wing_controls) && requested_density >= 1 && requested_density <= 4 &&
         (pause == 0 || pause == 1) && controls.speed >= 1 && controls.speed <= 30 && controls.angle >= -20 &&
         controls.angle <= 25 && controls.flap >= 0 && controls.flap <= 40 && std::abs(controls.aileron) <= 25 &&
         std::abs(controls.elevator) <= 25 && std::abs(controls.rudder) <= 25;
}

}  // namespace wind3d::web
