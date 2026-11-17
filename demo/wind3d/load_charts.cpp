#include "load_charts.h"

#include <cstdio>

#include "imgui.h"

namespace wind3d::load_charts {
namespace {
const ImU32 colors[] = {IM_COL32(255, 80, 80, 255), IM_COL32(64, 223, 96, 255), IM_COL32(80, 128, 255, 255)};

void Plot(const LoadHistory &history, const Window &window, bool moment, bool demean, float height) {
  ImGui::PushID(moment ? "moment" : "force");
  ImGui::TextUnformatted(moment ? "Moment M / N m (world origin)" : "Force F / N (world axes)");
  ImVec2 origin = ImGui::GetCursorScreenPos();
  float width = std::max(100.f, ImGui::GetContentRegionAvail().x);
  ImGui::InvisibleButton("plot", {width, height});
  bool hovered = ImGui::IsItemHovered();
  auto *draw = ImGui::GetWindowDrawList();
  draw->AddRectFilled(origin, {origin.x + width, origin.y + height}, IM_COL32(12, 20, 30, 255), 4);
  if (window.first == history.samples.size()) {
    draw->AddText({origin.x + 12, origin.y + 20}, IM_COL32(150, 170, 190, 255), "Waiting for retained solver steps...");
    ImGui::PopID();
    return;
  }
  int offset = moment ? 3 : 0;
  double low = 1e300, high = -1e300;
  for (int k = 0; k < 3; ++k) {
    const auto &s = window.statistics[offset + k];
    double base = demean ? s.mean : 0;
    low = std::min(low, s.minimum - base);
    high = std::max(high, s.maximum - base);
  }
  if (!window.valid_count) {
    low = -1;
    high = 1;
  }
  double padding = high > low ? .08 * (high - low) : std::max(std::abs(low) * 1e-6, 1e-9);
  low -= padding;
  high += padding;
  char labels[5][48];
  float margin = 58;
  for (int i = 0; i < 5; ++i) {
    std::snprintf(labels[i], sizeof(labels[i]), "%.9g", high - (high - low) * i / 4);
    margin = std::max(margin, ImGui::CalcTextSize(labels[i]).x + 12);
  }
  margin = std::min(margin, width * .47f);
  float left = origin.x + margin, right = origin.x + width - 10, top = origin.y + 10, bottom = origin.y + height - 28;
  double span = window.end - window.start;
  if (span <= 0)
    span = std::max(window.dt_max, 1e-6);
  auto x = [&](double t) { return left + float((t - window.start) / span) * (right - left); };
  auto y = [&](double value) { return bottom - float((value - low) / (high - low)) * (bottom - top); };
  for (int i = 0; i < 5; ++i) {
    float yy = top + (bottom - top) * i / 4, xx = left + (right - left) * i / 4;
    draw->AddLine({left, yy}, {right, yy}, IM_COL32(36, 55, 70, 255));
    draw->AddText({left - ImGui::CalcTextSize(labels[i]).x - 6, yy - ImGui::GetFontSize() * .5f},
                  IM_COL32(157, 178, 194, 255), labels[i]);
    char label[48];
    std::snprintf(label, sizeof(label), "%.*f", span < .05 ? 5 : 3, window.start + span * i / 4);
    float tx =
        std::clamp(xx - ImGui::CalcTextSize(label).x * .5f, origin.x, origin.x + width - ImGui::CalcTextSize(label).x);
    draw->AddText({tx, bottom + 6}, IM_COL32(157, 178, 194, 255), label);
  }
  draw->PushClipRect({left, top}, {right, bottom}, true);
  for (size_t i : window.changes) {
    double t = history.samples[i].time;
    if (t < window.start || t > window.end)
      continue;
    for (float yy = top; yy < bottom; yy += 7)
      draw->AddLine({x(t), yy}, {x(t), std::min(yy + 3, bottom)}, IM_COL32(208, 171, 101, 200));
  }
  for (int k = 0; k < 3; ++k) {
    double base = demean ? window.statistics[offset + k].mean : 0;
    // Keep first, min, max and last samples in every pixel column. Hover and
    // statistics still use all solver steps, and subpixel spikes remain visible.
    std::vector<ImVec2> points;
    int column = -1;
    size_t first = 0, last = 0, minimum = 0, maximum = 0;
    bool bucket = false;
    auto flush = [&] {
      if (!bucket)
        return;
      size_t ids[] = {first, minimum, maximum, last};
      std::sort(ids, ids + 4);
      size_t previous = ~size_t(0);
      for (size_t i : ids)
        if (i != previous) {
          points.push_back({x(history.samples[i].time), y(Value(history.samples[i], offset + k) - base)});
          previous = i;
        }
      bucket = false;
    };
    auto stroke = [&] {
      flush();
      if (points.size() > 1)
        draw->AddPolyline(points.data(), int(points.size()), colors[k], 0, 1.3f);
      else if (points.size() == 1)
        draw->AddCircleFilled(points[0], 2, colors[k]);
      points.clear();
      column = -1;
    };
    for (size_t i = window.first; i < history.samples.size(); ++i) {
      if (!window.valid[i]) {
        stroke();
        continue;
      }
      if (i > window.first && history.samples[i].step != history.samples[i - 1].step + 1)
        stroke();
      int pixel = int(x(history.samples[i].time) - left);
      if (!bucket || pixel != column) {
        flush();
        column = pixel;
        first = last = minimum = maximum = i;
        bucket = true;
      } else {
        last = i;
        if (Value(history.samples[i], offset + k) < Value(history.samples[minimum], offset + k))
          minimum = i;
        if (Value(history.samples[i], offset + k) > Value(history.samples[maximum], offset + k))
          maximum = i;
      }
    }
    stroke();
  }
  if (hovered) {
    double target =
        window.start + std::clamp(double((ImGui::GetIO().MousePos.x - left) / (right - left)), 0.0, 1.0) * span;
    auto begin = history.samples.begin() + window.first, end = history.samples.end();
    auto found = std::lower_bound(begin, end, target, [](const LoadSample &s, double t) { return s.time < t; });
    if (found == end)
      --found;
    if (found != begin && target - (found - 1)->time < found->time - target)
      --found;
    size_t index = size_t(found - history.samples.begin());
    float xx = x(found->time);
    draw->AddLine({xx, top}, {xx, bottom}, IM_COL32(220, 235, 245, 220));
    ImGui::BeginTooltip();
    ImGui::Text("t %.6f s | step %llu | dt %.2f us", found->time, static_cast<unsigned long long>(found->step),
                found->dt * 1e6);
    if (!window.valid[index])
      ImGui::TextUnformatted("Hidden: first 20 steps after a control change");
    else
      for (int kind = 0; kind < 2; ++kind)
        ImGui::Text("%s X % .8g | Y % .8g | Z % .8g", kind ? "M [N m]" : "F [N]",
                    Value(*found, kind * 3) - (demean ? window.statistics[kind * 3].mean : 0),
                    Value(*found, kind * 3 + 1) - (demean ? window.statistics[kind * 3 + 1].mean : 0),
                    Value(*found, kind * 3 + 2) - (demean ? window.statistics[kind * 3 + 2].mean : 0));
    ImGui::EndTooltip();
  }
  draw->PopClipRect();
  if (ImGui::BeginTable("statistics", 4, ImGuiTableFlags_SizingStretchProp)) {
    ImGui::TableSetupColumn("Axis");
    ImGui::TableSetupColumn("Mean");
    ImGui::TableSetupColumn("Peak-to-peak");
    ImGui::TableSetupColumn("Std dev");
    ImGui::TableHeadersRow();
    for (int k = 0; k < 3; ++k) {
      ImGui::TableNextRow();
      ImGui::TableNextColumn();
      ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(colors[k]), "%c", "XYZ"[k]);
      const auto &s = window.statistics[offset + k];
      double values[] = {s.mean, s.maximum - s.minimum, s.deviation};
      for (double v : values) {
        ImGui::TableNextColumn();
        if (window.valid_count)
          ImGui::Text("%.5g", v);
        else
          ImGui::TextUnformatted("--");
      }
    }
    ImGui::EndTable();
  }
  ImGui::PopID();
}
}  // namespace

float Draw(const LoadHistory &history, Settings &settings, bool &visible, float panel_width) {
  if (!visible)
    return 0;
  auto size = ImGui::GetIO().DisplaySize;
  float height = std::min(390.f, size.y * .45f);
  ImGui::SetNextWindowPos({0, size.y - height});
  ImGui::SetNextWindowSize({std::max(250.f, size.x - panel_width), height});
  bool expanded = ImGui::Begin("Force and moment history", &visible,
                               ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);
  if (expanded) {
    const double windows[] = {.05, .1, .25, .5, 1, 2};
    const uint64_t skips[] = {0, 5, 20, 50, 100};
    ImGui::SetNextItemWidth(82);
    ImGui::Combo("Window (s)", &settings.window_index,
                 "0.05\0"
                 "0.1\0"
                 "0.25\0"
                 "0.5\0"
                 "1\0"
                 "2\0");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(65);
    ImGui::Combo("Skip startup steps", &settings.skip_index,
                 "0\0"
                 "5\0"
                 "20\0"
                 "50\0"
                 "100\0");
    ImGui::SameLine();
    ImGui::Checkbox("Subtract mean", &settings.demean);
    ImGui::Checkbox("Hide 20 steps after control changes", &settings.hide_changes);
    ImGui::SameLine();
    for (int k = 0; k < 3; ++k) {
      ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(colors[k]), "%c", "XYZ"[k]);
      if (k < 2)
        ImGui::SameLine();
    }
    auto window = Select(history, windows[settings.window_index], skips[settings.skip_index], settings.hide_changes);
    if (window.first < history.samples.size())
      ImGui::Text("Every solver step | %zu samples (%zu shown) | physical time %.4f - %.4f s%s%s",
                  history.samples.size() - window.first, window.valid_count, window.start, window.end,
                  history.capacity_limited ? " | capacity limit" : "", window.contiguous ? "" : " | STEP GAP");
    else
      ImGui::Text("Waiting for steps after %llu (set skip to 0 for startup)",
                  static_cast<unsigned long long>(skips[settings.skip_index]));
    float plot_height = std::max(90.f, ImGui::GetContentRegionAvail().y - 116);
    if (ImGui::BeginTable("charts", 2, ImGuiTableFlags_SizingStretchSame)) {
      ImGui::TableNextColumn();
      Plot(history, window, false, settings.demean, plot_height);
      ImGui::TableNextColumn();
      Plot(history, window, true, settings.demean, plot_height);
      ImGui::EndTable();
    }
  }
  ImGui::End();
  return visible ? height : 0;
}
}  // namespace wind3d::load_charts
