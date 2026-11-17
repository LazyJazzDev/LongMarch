#pragma once
#include <chrono>
#include <iomanip>
#include <sstream>

#include "presentation.h"
#include "simulation.h"
#include "visualization.h"
#include "web_protocol.h"

namespace wind3d::web {
// One solver owns both frontends. The relay owns commands; this publisher owns
// immutable frames and acknowledges only commands it has actually consumed.
class Session {
 public:
  explicit Session(const std::string &directory) : directory_(directory) {
    if (directory.empty())
      return;
    std::error_code error;
    std::filesystem::create_directories(directory_, error);
    valid_ = !error;
    id_ = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    Command command;
    if (ReadCommand(directory_ / "controls.txt", command)) {
      sequence_ = command.sequence;
      reset_ = command.reset;
    }
  }

  bool Enabled() const {
    return !directory_.empty();
  }

  const std::string &ID() const {
    return id_;
  }

  bool Poll(Simulation &simulation, wind3d::Controls &controls, bool &paused, ViewSettings &view) {
    if (!Enabled())
      return false;
    Command command;
    if (!ReadCommand(directory_ / "controls.txt", command) || command.sequence == sequence_)
      return false;
    sequence_ = command.sequence;
    requested_ = command.controls;
    requested_pause_ = command.paused;
    pending_ = true;
    controls = command.controls;
    paused = command.paused;
    if (command.has_view) {
      auto directory = view.web_directory;
      view = command.view;
      view.web_directory = directory;
    }
    view.streamline_density = command.density;
    simulation.Configure(controls, paused);
    if (command.reset != reset_) {
      reset_ = command.reset;
      const auto state = simulation.Latest();
      expected_generation_ = state ? state->generation + 1 : 1;
      simulation.Reset();
    }
    return true;
  }

  bool Publish(const Snapshot &s,
               const LoadHistory &history,
               const ViewSettings &view,
               bool paused,
               const std::vector<Vertex> &flow,
               const wind3d::Controls &requested_controls,
               bool requested_paused) {
    if (!Enabled())
      return true;
    if (!valid_)
      return false;
    auto now = std::chrono::steady_clock::now();
    if (now < next_)
      return true;
    next_ = now + std::chrono::milliseconds(500);
    if (!Write(directory_ / "heartbeat.txt", id_))
      return false;
    if (pending_ && Same(s.controls, requested_) && paused == requested_pause_ && s.generation >= expected_generation_)
      pending_ = false;
    std::string volume_id;
    if (view.volume && !s.vorticity.empty()) {
      volume_id = id_ + "-" + std::to_string(s.generation) + "-" + std::to_string(s.step);
      if (volume_id != volume_id_) {
        if (!Write(directory_ / ("volume-" + volume_id + ".bin"),
                   std::string(reinterpret_cast<const char *>(s.vorticity.data()), s.vorticity.size() * sizeof(float))))
          return false;
        if (!previous_volume_.empty()) {
          std::error_code error;
          std::filesystem::remove(directory_ / ("volume-" + previous_volume_ + ".bin"), error);
        }
        previous_volume_ = volume_id_;
        volume_id_ = volume_id;
      }
    }
    if (mesh_ != s.aircraft) {
      mesh_ = s.aircraft;
      mesh_id_ = id_ + std::to_string(++mesh_serial_);
      if (!Write(directory_ / ("mesh-" + mesh_id_ + ".bin"),
                 std::string(reinterpret_cast<const char *>(mesh_->data()), mesh_->size() * sizeof(Vertex))))
        return false;
    }
    std::ostringstream json;
    auto decorations = visualization::PresentationLines(s, view);
    json << std::setprecision(9) << "{\"version\":2,\"run\":\"" << id_ << "\""
         << ",\"step\":" << s.step << ",\"generation\":" << s.generation << ",\"mesh\":\"" << mesh_id_
         << "\",\"mesh_vertices\":" << mesh_->size() << ",\"line_vertices\":" << flow.size()
         << ",\"line_stride\":36,\"volume\":\"" << volume_id << "\""
         << ",\"decoration_vertices\":" << decorations.size() << ",\"time\":" << s.time << ",\"dt\":" << s.dt
         << ",\"step_seconds\":" << s.step_seconds << ",\"pressure_cycles\":" << s.pressure_cycles
         << ",\"pressure_residual\":" << s.pressure_residual
         << ",\"pressure_converged\":" << (s.pressure_converged ? "true" : "false")
         << ",\"divergence\":" << s.divergence_after << ",\"grid\":" << s.solver_nx
         << ",\"paused\":" << (paused ? "true" : "false") << ",\"ack\":" << (pending_ ? 0 : sequence_)
         << ",\"model_origin_body\":[0,0,0],\"construction_datum\":";
    Vector(json, ModelDatum());
    json << ",\"controls\":";
    Controls(json, s.controls);
    json << ",\"requested_controls\":";
    Controls(json, requested_controls);
    json << ",\"requested_paused\":" << (requested_paused ? "true" : "false");
    json << ",\"force\":";
    Vector(json, s.force);
    json << ",\"moment\":";
    Vector(json, s.moment);
    json << ",\"domain_min\":[-6,-4,-6],\"domain_max\":[10,4,6],\"streamline_density\":" << view.streamline_density
         << ",\"mean_ld\":";
    if (std::abs(s.impulse[0]) > 1e-9)
      json << s.impulse[1] / s.impulse[0];
    else
      json << "null";
    json << ",\"average_seconds\":" << s.time << ",\"mean_scope\":\"since-reset\",\"view\":{"
         << "\"flow\":" << int(view.flow) << ",\"loads\":" << int(view.loads) << ",\"aircraft\":" << int(view.aircraft)
         << ",\"domain\":" << int(view.domain) << ",\"volume\":" << int(view.volume)
         << ",\"width\":" << view.streamline_width << ",\"transparent_curvature\":" << view.transparent_curvature
         << ",\"opaque_curvature\":" << view.opaque_curvature << ",\"yaw\":" << view.yaw << ",\"pitch\":" << view.pitch
         << ",\"distance\":" << view.distance << ",\"colored_volume\":" << int(view.colored_volume)
         << ",\"volume_density\":" << view.volume_density << ",\"volume_threshold\":" << view.volume_threshold
         << ",\"volume_step\":" << view.volume_step << ",\"history_window\":" << view.history_window
         << ",\"history_skip\":" << view.history_skip << ",\"history\":" << int(view.history)
         << ",\"history_demean\":" << int(view.history_demean)
         << ",\"history_hide_changes\":" << int(view.history_hide_changes) << "}}";
    auto header = json.str();
    if (header == last_header_)
      return true;
    const auto signature = header;
    while (header.size() % 4)
      header += ' ';
    uint32_t length = uint32_t(header.size());
    std::string packet(reinterpret_cast<const char *>(&length), sizeof(length));
    packet += header;
    packet.append(reinterpret_cast<const char *>(flow.data()), flow.size() * sizeof(Vertex));
    packet.append(reinterpret_cast<const char *>(decorations.data()), decorations.size() * sizeof(Vertex));
    if (!Write(directory_ / "frame.bin", packet))
      return false;
    last_header_ = signature;
    if (history.samples.empty())
      return true;
    std::ostringstream h;
    h << std::setprecision(9) << "{\"version\":1,\"run\":\"" << id_ << "\",\"generation\":" << history.generation
      << ",\"count\":" << history.samples.size()
      << ",\"stride\":44,\"retention_seconds\":2,\"capacity_limited\":" << (history.capacity_limited ? "true" : "false")
      << ",\"controls_events\":[";
    bool first = true;
    wind3d::Controls previous;
    for (const auto &sample : history.samples) {
      if (first || !Same(previous, sample.controls)) {
        if (!first)
          h << ',';
        h << "{\"step\":" << sample.step << ",\"time\":" << sample.time << ",\"controls\":";
        Controls(h, sample.controls);
        h << '}';
        previous = sample.controls;
        first = false;
      }
    }
    h << "]}";
    header = h.str();
    length = uint32_t(header.size());
    packet.assign(reinterpret_cast<const char *>(&length), sizeof(length));
    packet += header;
    for (const auto &sample : history.samples) {
      packet.append(reinterpret_cast<const char *>(&sample.step), sizeof(sample.step));
      packet.append(reinterpret_cast<const char *>(&sample.time), sizeof(sample.time));
      packet.append(reinterpret_cast<const char *>(&sample.dt), sizeof(sample.dt));
      packet.append(reinterpret_cast<const char *>(&sample.force), sizeof(sample.force));
      packet.append(reinterpret_cast<const char *>(&sample.moment), sizeof(sample.moment));
    }
    return Write(directory_ / "loads.bin", packet);
  }

 private:
  std::filesystem::path directory_;
  bool valid_{};
  bool pending_{}, requested_pause_{};
  wind3d::Controls requested_;
  uint64_t expected_generation_{};
  std::string id_, mesh_id_, volume_id_, previous_volume_, last_header_;
  uint64_t sequence_{}, reset_{}, mesh_serial_{};
  std::chrono::steady_clock::time_point next_{};
  std::shared_ptr<const std::vector<Vertex>> mesh_;
};
}  // namespace wind3d::web
