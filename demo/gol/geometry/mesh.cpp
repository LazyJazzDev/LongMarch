#include "mesh.h"

#include <map>
#include <queue>

#include "line.h"
#include "segment.h"
#include "util.h"
#define LAND_PI 3.14159265358979323846

namespace geometry {

using namespace glm;

Mesh::Mesh() = default;

Mesh::Mesh(std::vector<vec2> outline) {
  auto outline_backup = outline;

  //        for (int i = 0; i < outline.size(); i++) {
  //            int j = (i + 1) % outline.size();
  //            triangles.push_back({outline[i], outline[j], outline[i]});
  //            LOG_INFO("({}, {})", outline[i].x, outline[i].y);
  //        }
  //        return;

  auto signal = [](float x) {
    if (x < -1e-4)
      return -1;
    else if (x > 1e-4)
      return 1;
    return 0;
  };

  auto is_intersect = [signal](const Segment &seg0, const Segment &seg1) {
    Line line0{seg0};
    Line line1{seg1};
    return (std::abs(signal(line0(seg1.u)) - signal(line0(seg1.v))) == 2) &&
           (std::abs(signal(line1(seg0.u)) - signal(line1(seg0.v))) == 2);
  };

  auto cross = [](const vec2 &v0, const vec2 &v1) { return v0.x * v1.y - v0.y * v1.x; };

  float area = 0.0;
  for (size_t i = 1; i < outline.size() - 1; i++) {
    area += cross(outline[i] - outline[0], outline[i + 1] - outline[0]);
  }
  int direction = signal(area);

  if (!direction) {
    LOG_ERROR("Outline has no area.");
    return;
  }

  while (outline.size() > 2) {
    int sz = int(outline.size());
    int result_index = sz;
    float result_area = -1.0;
    int mid_index = 0;
    int prev_index, next_index;
    Triangle triangle{};
    while (mid_index < sz) {
      prev_index = (mid_index - 1 + sz) % sz;
      next_index = (mid_index + 1) % sz;
      triangle = Triangle{outline[prev_index], outline[mid_index], outline[next_index]};

      auto triangle_direction =
          signal(cross(outline[mid_index] - outline[prev_index], outline[next_index] - outline[prev_index]));
      if (triangle_direction == -direction) {
        mid_index++;
        continue;
      }
      bool pass = true;
      auto shortcut = Segment{outline[prev_index], outline[next_index]};
      for (int i = 0; i < sz - 2; i++) {
        int j = (next_index + i) % sz;
        int k = (next_index + i + 1) % sz;
        if (is_intersect(shortcut, Segment{outline[j], outline[k]})) {
          pass = false;
          break;
        }
        if (i && triangle_direction) {
          if (triangle.IsInside(outline[j])) {
            pass = false;
            break;
          }
        }
      }
      if (pass) {
        auto this_area = std::abs(cross(triangle.v1 - triangle.v0, triangle.v2 - triangle.v0));
        //                    result_index = mid_index;
        //                    result_area = this_area;
        //                    break;
        if (this_area > result_area) {
          result_area = this_area;
          result_index = mid_index;
        }
      }
      mid_index++;
    }

    mid_index = result_index;

    if (mid_index == outline.size()) {
      LOG_ERROR("Cannot find a mid_index");

      outline = outline_backup;
      for (auto v : outline) {
        LOG_INFO("({}, {})", v.x, v.y);
      }

      triangles.clear();
      for (int i = 0; i < outline.size(); i++) {
        int j = (i + 1) % outline.size();
        triangles.push_back({outline[i], outline[j], outline[i]});
      }
      return;
      return;
    }

    prev_index = (mid_index - 1 + sz) % sz;
    next_index = (mid_index + 1) % sz;
    triangle = Triangle{outline[prev_index], outline[mid_index], outline[next_index]};
    if (result_area > eps) {
      triangles.push_back(triangle);
    }
    for (int i = mid_index; i < sz - 1; i++) {
      outline[i] = outline[i + 1];
    }
    outline.pop_back();
  }
}

std::vector<Triangle> &Mesh::GetTriangles() {
  return triangles;
}

const std::vector<Triangle> &Mesh::GetTriangles() const {
  return triangles;
}

Mesh Mesh::DiscardIntersection(const Mesh &mesh) {
  std::vector<vec2> existed_pos;
  std::map<Segment, int> segments;
  std::queue<std::pair<Segment, int>> q;

  auto correct_pos = [&existed_pos](vec2 p) {
    for (auto &pos : existed_pos) {
      if (length(pos - p) < 1e-8)
        return pos;
    }
    existed_pos.push_back(p);
    return p;
  };

  auto emplace_task = [&q, correct_pos](vec2 u, vec2 v, int w) {
    u = correct_pos(u);
    v = correct_pos(v);
    if (u != v) {
      q.push({Segment{u, v}, w});
    }
  };

  auto signal = [](float x) {
    if (x < 0.0)
      return -1;
    if (x > 0.0)
      return 1;
    return 0;
  };

  for (auto &triangle : mesh.GetTriangles()) {
    emplace_task(triangle.v0, triangle.v1, 1);
    emplace_task(triangle.v1, triangle.v2, 1);
    emplace_task(triangle.v2, triangle.v0, 1);
  }

  auto is_intersect = [signal](const Segment &seg0, const Segment &seg1) {
    Line line0{seg0};
    Line line1{seg1};
    return (signal(line0(seg1.u)) != signal(line0(seg1.v)) &&
            std::abs(signal(line1(seg0.u)) - signal(line1(seg0.v))) == 2) ||
           (signal(line1(seg0.u)) != signal(line1(seg0.v)) &&
            std::abs(signal(line0(seg1.u)) - signal(line0(seg1.v))) == 2);
  };

  while (!q.empty()) {
    auto task = q.front();
    q.pop();

    if (!segments.empty()) {
      bool pass = true;
      for (auto segment : segments) {
        Line line{segment.first};
        Line task_line{task.first};
        if (task_line == line) {
          if ((segment.first.u == task.first.u && segment.first.v == task.first.v) ||
              (segment.first.u == task.first.v && segment.first.v == task.first.u)) {
            segments.at(segment.first) += task.second;
            pass = false;
            break;
          } else {
            vec2 dir = line.n;
            std::swap(dir.x, dir.y);
            dir.x = -dir.x;
            auto tu = task.first.u;
            auto tv = task.first.v;
            float t0 = dot(dir, tu);
            float t1 = dot(dir, tv);
            auto su = segment.first.u;
            auto sv = segment.first.v;
            float s0 = dot(dir, su);
            float s1 = dot(dir, sv);
            if (s0 > s1) {
              std::swap(s0, s1);
              std::swap(su, sv);
            }
            if (t0 > t1) {
              std::swap(t0, t1);
              std::swap(tu, tv);
            }

            if (s0 < t0 && t0 < s1 && s0 < t1 && t1 < s1) {
              segments.erase(segment.first);
              emplace_task(su, tu, segment.second);
              emplace_task(tu, tv, segment.second + task.second);
              emplace_task(tv, sv, segment.second);
              pass = false;
              break;
            }

            if (t0 < s0 && s0 < t1 && t0 < s1 && s1 < t1) {
              emplace_task(tu, su, task.second);
              emplace_task(su, sv, task.second);
              emplace_task(sv, tv, task.second);
              pass = false;
              break;
            }

            if ((t0 < s0 && s0 < t1) || (s0 < t1 && t1 < s1)) {
              segments.erase(segment.first);
              emplace_task(tu, su, task.second);
              emplace_task(su, tv, task.second + segment.second);
              emplace_task(tv, sv, segment.second);
              pass = false;
              break;
            }

            if ((t0 < s1 && s1 < t1) || (s0 < t0 && t0 < s1)) {
              segments.erase(segment.first);
              emplace_task(su, tu, segment.second);
              emplace_task(tu, sv, segment.second + task.second);
              emplace_task(sv, tv, task.second);
              pass = false;
              break;
            }
          }
        } else {
          if (is_intersect(task.first, segment.first)) {
            segments.erase(segment.first);
            auto mid = Intersection(Line(task.first), Line(segment.first));
            emplace_task(task.first.u, mid, task.second);
            emplace_task(task.first.v, mid, task.second);
            emplace_task(segment.first.u, mid, segment.second);
            emplace_task(segment.first.v, mid, segment.second);
            pass = false;
            break;
          }
        }
      }
      if (!pass)
        continue;
    }
  }

  LOG_INFO("Sketch created");
  return {};
}

Mesh &Mesh::operator+=(const Mesh &mesh) {
  triangles.reserve(triangles.size() + mesh.triangles.size());
  for (auto triangle : mesh.triangles)
    triangles.push_back(triangle);
  return *this;
}
}  // namespace geometry
