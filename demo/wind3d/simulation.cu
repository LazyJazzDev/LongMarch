#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>

#include "pressure.cuh"
#include "simulation.h"
#include "wall_model.h"

namespace wind3d {
namespace {
constexpr float density = 1.225f;
#if CUDART_VERSION >= 13000
cudaMemLocation Location(int device) {
  return {cudaMemLocationTypeDevice, device};
}
#else
int Location(int device) {
  return device;
}
#endif

struct Physics {
  float viscosity, wall_viscosity;
  bool wall_model, wall_momentum, cut_cells;
};

__host__ __device__ inline float WallViscosity(Physics physics, float speed, float distance) {
  return physics.wall_model ? WallEffectiveViscosity(physics.wall_viscosity, speed, distance) : physics.viscosity;
}

struct Grid {
  int nx, ny, nz;
  float h;
  int begin{};
  bool cubic{};
  // Face apertures of u, v and w on a cut-cell boundary; null otherwise.
  const unsigned char *aperture[3]{};

  __host__ __device__ int C(int x, int y, int z) const {
    return (z * ny + y) * nx + x;
  }

  __host__ __device__ int F(int x, int y, int z) const {
    return (z * (ny + 1) + y) * (nx + 1) + x;
  }

  __device__ bool Inside(int x, int y, int z) const {
    return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
  }

  __device__ Vec3 Position(int x, int y, int z) const {
    return {(x + 0.5f) * h - 6, (y + 0.5f) * h - 4, (z + 0.5f) * h - 6};
  }
};

__device__ float Read(const float *f, Grid g, int x, int y, int z) {
  return f[g.F(max(0, min(x, g.nx)), max(0, min(y, g.ny)), max(0, min(z, g.nz)))];
}

// Trilinear interpolation of a face field. For a velocity component on a
// cut-cell boundary, each weight is scaled by its face's open fraction and the
// result renormalized: the projection is non-expansive only in the
// aperture-weighted energy norm, so a large velocity on a nearly closed face
// must not spread into open faces, or the step amplifies it.
__device__ float Sample(const float *f, Grid g, Vec3 p, int component = -1) {
  p.x = fminf(fmaxf(p.x, 0), float(g.nx));
  p.y = fminf(fmaxf(p.y, 0), float(g.ny));
  p.z = fminf(fmaxf(p.z, 0), float(g.nz));
  int x = int(p.x), y = int(p.y), z = int(p.z);
  const unsigned char *aperture = component >= 0 ? g.aperture[component] : nullptr;
  float result = 0, weighted = 0, total = 0;
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        float weight = (i ? p.x - x : 1 - p.x + x) * (j ? p.y - y : 1 - p.y + y) * (k ? p.z - z : 1 - p.z + z);
        float value = Read(f, g, x + i, y + j, z + k);
        result += value * weight;
        if (aperture) {
          float open = aperture[g.F(min(x + i, g.nx), min(y + j, g.ny), min(z + k, g.nz))] * weight;
          weighted += value * open;
          total += open;
        }
      }
  return aperture && total > 1e-3f ? weighted / total : result;
}

// Monotone cubic Hermite interpolation between f1 and f2 (Fedkiw, Stam and
// Jensen 2001): slopes that disagree in sign with f2 - f1 are zeroed, so the
// result stays within [f1, f2] and creates no new extrema.
__device__ inline float MonotoneCubic(float f0, float f1, float f2, float f3, float t) {
  float delta = f2 - f1, d1 = 0.5f * (f2 - f0), d2 = 0.5f * (f3 - f1);
  if (delta == 0 || d1 * delta <= 0)
    d1 = 0;
  if (delta == 0 || d2 * delta <= 0)
    d2 = 0;
  float t2 = t * t, t3 = t2 * t;
  float value = (2 * t3 - 3 * t2 + 1) * f1 + (t3 - 2 * t2 + t) * d1 + (-2 * t3 + 3 * t2) * f2 + (t3 - t2) * d2;
  return fminf(fmaxf(value, fminf(f1, f2)), fmaxf(f1, f2));
}

// Transported value at p: monotone tricubic where the eight surrounding faces
// are fully open, otherwise the (aperture-weighted) trilinear Sample.
__device__ float SampleTransported(const float *f, Grid g, Vec3 p, int component) {
  if (!g.cubic)
    return Sample(f, g, p, component);
  p.x = fminf(fmaxf(p.x, 0), float(g.nx));
  p.y = fminf(fmaxf(p.y, 0), float(g.ny));
  p.z = fminf(fmaxf(p.z, 0), float(g.nz));
  int x = int(p.x), y = int(p.y), z = int(p.z);
  const unsigned char *aperture = g.aperture[component];
  if (aperture)
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i)
          if (aperture[g.F(min(x + i, g.nx), min(y + j, g.ny), min(z + k, g.nz))] != 255)
            return Sample(f, g, p, component);
  float tx = p.x - x, ty = p.y - y, tz = p.z - z, planes[4];
  for (int k = 0; k < 4; ++k) {
    float rows[4];
    for (int j = 0; j < 4; ++j)
      rows[j] = MonotoneCubic(Read(f, g, x - 1, y + j - 1, z + k - 1), Read(f, g, x, y + j - 1, z + k - 1),
                              Read(f, g, x + 1, y + j - 1, z + k - 1), Read(f, g, x + 2, y + j - 1, z + k - 1), tx);
    planes[k] = MonotoneCubic(rows[0], rows[1], rows[2], rows[3], ty);
  }
  return MonotoneCubic(planes[0], planes[1], planes[2], planes[3], tz);
}

__global__ void Mask(Grid g, unsigned char *solid, Controls controls) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= g.nx * g.ny * g.nz)
    return;
  int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
  Vec3 center = g.Position(x, y, z);
  solid[n] = AircraftDistance(center, controls) <= 0;
  if (solid[n])
    for (int axis = 0; axis < 3; ++axis) {
      Vec3 offset{axis == 0 ? .5f * g.h : 0, axis == 1 ? .5f * g.h : 0, axis == 2 ? .5f * g.h : 0};
      if (AircraftDistance(Add(center, offset), controls) > 0 &&
          AircraftDistance(Add(center, Scale(offset, -1)), controls) > 0) {
        // Bit 1 marks an underresolved solid sheet. Face-area sampling must
        // not reconnect its two fluid sides through a single pressure cell.
        solid[n] = 3;
        break;
      }
    }
}

__device__ bool Blocked(Grid g, const unsigned char *solid, int x, int y, int z, int axis, unsigned char mask = 255) {
  if (g.Inside(x, y, z) && (solid[g.C(x, y, z)] & mask))
    return true;
  if (axis == 0)
    --x;
  if (axis == 1)
    --y;
  if (axis == 2)
    --z;
  return g.Inside(x, y, z) && (solid[g.C(x, y, z)] & mask);
}

// Subcell sampling of the pressure-cell connection captures thin walls even
// when both adjacent cell centres are fluid. These faces also define A and loads.
// A blocked face stores 1..255: its share of the true wetted area. A staircase
// approximating a surface with unit normal n has |n_a| faces of axis a per unit
// area, so every face carries 1 / |n|_1 of its own area for wall shear.
//
// Every face also stores its aperture, 0 (closed) to 255 (open). Without cut
// cells it is 0 for a blocked face and 255 otherwise. With cut cells it is the
// open fraction of the face area (Ng, Min and Gibou 2009), except that a thin
// wall crossing the connection between two fluid cell centres closes the face.
__global__ void FaceMask(Grid g,
                         const unsigned char *solid,
                         unsigned char *bu,
                         unsigned char *bv,
                         unsigned char *bw,
                         unsigned char *au,
                         unsigned char *av,
                         unsigned char *aw,
                         Controls controls,
                         bool cut_cells) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  bu[n] = bv[n] = bw[n] = 0;
  au[n] = av[n] = aw[n] = 255;
  Vec3 body = BodyPoint(g.Position(x, y, z), controls);
  if (body.x < -3.1f - 2 * g.h || body.x > 3.4f + 2 * g.h || fabsf(body.z) > 3.4f + 2 * g.h ||
      body.y < -0.85f - 2 * g.h || body.y > 1.4f + 2 * g.h)
    return;
  unsigned char *faces[] = {bu, bv, bw}, *apertures[] = {au, av, aw};
  for (int axis = 0; axis < 3; ++axis) {
    bool centre_solid = Blocked(g, solid, x, y, z, axis), blocked = centre_solid;
    Vec3 normal{axis == 0 ? 1.0f : 0, axis == 1 ? 1.0f : 0, axis == 2 ? 1.0f : 0};
    Vec3 center = Add(g.Position(x, y, z), Scale(normal, -0.5f * g.h));
    for (int sample = -3; sample <= 3 && !blocked; ++sample)
      blocked = AircraftDistance(Add(center, Scale(normal, sample * g.h / 8)), controls) <= 0;
    unsigned char aperture = blocked ? 0 : 255;
    if (cut_cells && !Blocked(g, solid, x, y, z, axis, 2) && (centre_solid || !blocked)) {
      // Open fraction from 6 x 6 samples of the face square; faces far from
      // the surface are uniform. A thin wall between fluid centres stays closed.
      float d = AircraftDistance(center, controls);
      if (fabsf(d) > 2 * g.h) {
        aperture = d > 0 ? 255 : 0;
      } else {
        int open = 0;
        constexpr int samples = 6;
        for (int i = 0; i < samples; ++i)
          for (int j = 0; j < samples; ++j) {
            float a = ((i + 0.5f) / samples - 0.5f) * g.h, b = ((j + 0.5f) / samples - 0.5f) * g.h;
            Vec3 offset{axis == 0 ? 0 : a, axis == 1 ? 0 : (axis == 0 ? a : b), axis == 2 ? 0 : b};
            open += AircraftDistance(Add(center, offset), controls) > 0;
          }
        // Faces with a sliver of open area are closed: the projection barely
        // constrains their velocity, which can then grow without bound.
        aperture = 8 * open < samples * samples ? 0 : (unsigned char)rintf(255.0f * open / (samples * samples));
      }
      blocked = aperture == 0;
    }
    apertures[axis][n] = aperture;
    unsigned char area = 0;
    if (blocked) {
      float d = 0.5f * g.h, gradient[3];
      for (int k = 0; k < 3; ++k) {
        Vec3 e{k == 0 ? d : 0, k == 1 ? d : 0, k == 2 ? d : 0};
        gradient[k] =
            AircraftDistance(Add(center, e), controls) - AircraftDistance(Add(center, Scale(e, -1)), controls);
      }
      float l1 = fabsf(gradient[0]) + fabsf(gradient[1]) + fabsf(gradient[2]);
      float l2 = sqrtf(gradient[0] * gradient[0] + gradient[1] * gradient[1] + gradient[2] * gradient[2]);
      area = l1 > 0 ? (unsigned char)fmaxf(1, fminf(255, rintf(255 * l2 / l1))) : 255;
    }
    faces[axis][n] = area;
  }
}

__global__ void Boundary(Grid g,
                         const unsigned char *bu,
                         const unsigned char *bv,
                         const unsigned char *bw,
                         float *u,
                         float *v,
                         float *w,
                         float speed,
                         bool reset) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  if (reset || x == 0 || y == 0 || y == g.ny || z == 0 || z == g.nz) {
    u[n] = speed;
    v[n] = 0;
    w[n] = 0;
  }
  if (bu[n])
    u[n] = 0;
  if (bv[n])
    v[n] = 0;
  if (bw[n])
    w[n] = 0;
}

// Share of a blocked face's area that carries wall shear. The laminar no-slip
// discretization keeps whole faces, as the original demo does.
__device__ inline float WallArea(Physics physics, unsigned char face) {
  return face ? (physics.wall_model ? face / 255.0f : 1.0f) : 0.0f;
}

__device__ Vec3 FaceVelocity(Grid g, const float *u, const float *v, const float *w, Vec3 p) {
  return {Sample(u, g, Add(p, {0, -0.5f, -0.5f}), 0), Sample(v, g, Add(p, {-0.5f, 0, -0.5f}), 1),
          Sample(w, g, Add(p, {-0.5f, -0.5f, 0}), 2)};
}

// Whether any of the six same-component neighbours of a face is open.
__device__ bool OpenNeighbour(Grid g, const unsigned char *blocked, int x, int y, int z) {
  for (int k = 0; k < 6; ++k) {
    int a = x + (k == 0) - (k == 1), b = y + (k == 2) - (k == 3), c = z + (k == 4) - (k == 5);
    if (a >= 0 && b >= 0 && c >= 0 && a <= g.nx && b <= g.ny && c <= g.nz && !blocked[g.F(a, b, c)])
      return true;
  }
  return false;
}

// Extends fluid velocities into blocked faces before transport, one layer per
// pass, so interpolation near a wall sees the slip velocity at the first fluid
// sample. With the wall model, wall shear comes from the log law alone; zero
// values inside the solid would otherwise act as a numerical no-slip wall with
// a viscosity of order U h. Boundary later restores zero on blocked faces.
__global__ void Extrapolate(Grid g, const unsigned char *blocked, float *f, int layer) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1) || !blocked[n])
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  bool first = OpenNeighbour(g, blocked, x, y, z);
  if (layer == 1 ? !first : first)
    return;
  float sum = 0;
  int used = 0;
  for (int k = 0; k < 6; ++k) {
    int a = x + (k == 0) - (k == 1), b = y + (k == 2) - (k == 3), c = z + (k == 4) - (k == 5);
    if (a < 0 || b < 0 || c < 0 || a > g.nx || b > g.ny || c > g.nz)
      continue;
    int m = g.F(a, b, c);
    bool source = layer == 1 ? !blocked[m] : blocked[m] && OpenNeighbour(g, blocked, a, b, c);
    if (source) {
      sum += f[m];
      ++used;
    }
  }
  if (used)
    f[n] = sum / used;
}

__device__ Vec3
Departure(Grid g, const float *u, const float *v, const float *w, int x, int y, int z, int axis, float dt) {
  Vec3 offset{axis == 0 ? 0.0f : 0.5f, axis == 1 ? 0.0f : 0.5f, axis == 2 ? 0.0f : 0.5f};
  Vec3 p{x + offset.x, y + offset.y, z + offset.z};
  Vec3 velocity = FaceVelocity(g, u, v, w, p);
  Vec3 midpoint = Add(p, Scale(velocity, -0.5f * dt / g.h));
  velocity = FaceVelocity(g, u, v, w, midpoint);
  return Add(Add(p, Scale(velocity, -dt / g.h)), Scale(offset, -1));
}

__global__ void Advect(Grid g,
                       const float *u,
                       const float *v,
                       const float *w,
                       const float *source,
                       float *destination,
                       int axis,
                       float dt) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  destination[n] = SampleTransported(source, g, Departure(g, u, v, w, x, y, z, axis, dt), axis);
}

__global__ void CorrectTransport(Grid g,
                                 const float *u,
                                 const float *v,
                                 const float *w,
                                 const float *source,
                                 const float *reverse,
                                 float *forward,
                                 const unsigned char *bu,
                                 const unsigned char *bv,
                                 const unsigned char *bw,
                                 int axis,
                                 float dt,
                                 Physics physics) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  float transported = forward[n];
  if (reverse) {
    Vec3 back = Departure(g, u, v, w, x, y, z, axis, dt);
    int ix = int(floorf(back.x)), iy = int(floorf(back.y)), iz = int(floorf(back.z));
    float low = 1e30f, high = -1e30f;
    for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
          float value = Read(source, g, ix + i, iy + j, iz + k);
          low = fminf(low, value);
          high = fmaxf(high, value);
        }
    // Selle et al. (2008): revert to the first-order value when the corrected
    // value leaves the range of the departure stencil. Clamping to that range
    // instead keeps part of the correction, and with the projection it lets
    // perturbations grow when the Courant number is small.
    float corrected = transported + 0.5f * (source[n] - reverse[n]);
    if (corrected >= low && corrected <= high)
      transported = corrected;
  }
  float fluid_lap = 0, wall_flux = 0;
  bool sampled = false;
  Vec3 velocity{};
  const unsigned char *blocked[] = {bu, bv, bw};
  for (int direction = 0; direction < 3; ++direction)
    for (int sign = -1; sign <= 1; sign += 2) {
      int neighbour[] = {x, y, z};
      neighbour[direction] += sign;
      float value = Read(source, g, neighbour[0], neighbour[1], neighbour[2]);
      if (direction != axis) {
        // A tangential MAC velocity straddles two pressure cells. Each half
        // of its dual face uses that cell's wall flag. A stationary wall at
        // the midpoint has ghost velocity -u, rather than the fluid velocity
        // on the far side of a thin control surface.
        int face[] = {x, y, z};
        face[direction] += sign > 0;
        float fraction = 0, area = 0;
        for (int half = 0; half < 2; ++half) {
          if (face[0] >= 0 && face[0] <= g.nx && face[1] >= 0 && face[1] <= g.ny && face[2] >= 0 && face[2] <= g.nz) {
            unsigned char flag = blocked[direction][g.F(face[0], face[1], face[2])];
            fraction += flag ? 0.5f : 0;
            area += 0.5f * WallArea(physics, flag);
          }
          --face[axis];
        }
        if (fraction > 0 && physics.wall_momentum) {
          if (!sampled) {
            velocity = FaceVelocity(g, u, v, w,
                                    {x + (axis ? 0.5f : 0), y + (axis == 1 ? 0 : 0.5f), z + (axis == 2 ? 0 : 0.5f)});
            sampled = true;
          }
          float normal = direction == 0 ? velocity.x : direction == 1 ? velocity.y : velocity.z;
          float tangential = sqrtf(fmaxf(0, Dot(velocity, velocity) - normal * normal));
          wall_flux += WallViscosity(physics, tangential, 0.5f * g.h) * area * -2 * source[n];
        }
        fluid_lap += (1 - fraction) * (value - source[n]);
        continue;
      }
      fluid_lap += value - source[n];
    }
  forward[n] = transported + dt / (g.h * g.h) * (physics.viscosity * fluid_lap + wall_flux);
}

// With cut cells, a cell holds fluid (and a pressure unknown) when any of its
// faces is open, even if its centre lies inside the body.
__global__ void CellMask(Grid g,
                         unsigned char *solid,
                         const unsigned char *au,
                         const unsigned char *av,
                         const unsigned char *aw) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= g.nx * g.ny * g.nz)
    return;
  int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
  solid[n] = !(au[g.F(x, y, z)] | au[g.F(x + 1, y, z)] | av[g.F(x, y, z)] | av[g.F(x, y + 1, z)] | aw[g.F(x, y, z)] |
               aw[g.F(x, y, z + 1)]);
}

__device__ inline float Open(const unsigned char *aperture, int n) {
  return aperture[n] * (1.0f / 255);
}

// Divergence of the aperture-weighted face fluxes; the stationary solid
// supplies no flux through the closed part of a face.
__global__ void Divergence(Grid g,
                           const unsigned char *solid,
                           const unsigned char *au,
                           const unsigned char *av,
                           const unsigned char *aw,
                           const float *u,
                           const float *v,
                           const float *w,
                           float *div) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= g.nx * g.ny * g.nz)
    return;
  int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
  int a = g.F(x, y, z), bx = g.F(x + 1, y, z), by = g.F(x, y + 1, z), bz = g.F(x, y, z + 1);
  div[n] = solid[n] ? 0
                    : (Open(au, bx) * u[bx] - Open(au, a) * u[a] + Open(av, by) * v[by] - Open(av, a) * v[a] +
                       Open(aw, bz) * w[bz] - Open(aw, a) * w[a]) /
                          g.h;
}

__global__ void Project(Grid g,
                        const unsigned char *bu,
                        const unsigned char *bv,
                        const unsigned char *bw,
                        const float *p,
                        float *u,
                        float *v,
                        float *w,
                        float dt) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  float *fields[] = {u, v, w};
  const unsigned char *blocked[] = {bu, bv, bw};
  for (int axis = 0; axis < 3; ++axis) {
    int a = x - (axis == 0), b = y - (axis == 1), c = z - (axis == 2);
    if (blocked[axis][n]) {
      fields[axis][n] = 0;
      continue;
    }
    if (g.Inside(x, y, z) && g.Inside(a, b, c))
      fields[axis][n] -= dt / (density * g.h) * (p[g.C(x, y, z)] - p[g.C(a, b, c)]);
    else if (axis == 0 && x == g.nx && y < g.ny && z < g.nz)
      fields[axis][n] += 2 * dt / (density * g.h) * p[g.C(a, b, c)];
  }
}

struct Statistics {
  float before{}, after{}, max_speed{};
  int fastest{};
  Vec3 force{}, moment{}, friction{};
  // Drag (x) and lift (y) force of each aircraft component.
  float drag[kComponents]{}, lift[kComponents]{};
  int invalid{};
};

// Relaxes velocity toward the free stream in the last `length` metres before
// the outlet, with a rate that rises quadratically to 1 / (4 dt) at the outlet.
__global__ void Sponge(Grid g, float *u, float *v, float *w, float speed, float length) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= (g.nx + 1) * (g.ny + 1) * (g.nz + 1))
    return;
  int x = n % (g.nx + 1);
  // World x of this face column: the domain spans x = [-6, 10].
  float distance = 10 - (x * g.h - 6);
  if (distance >= length)
    return;
  float s = fminf(fmaxf(1 - distance / length, 0), 1), blend = 0.25f * s * s;
  if (x > 0)
    u[n] += blend * (speed - u[n]);
  v[n] -= blend * v[n];
  w[n] -= blend * w[n];
}

__device__ Vec3 CenterVelocity(Grid g, const float *u, const float *v, const float *w, int x, int y, int z) {
  return {0.5f * (u[g.F(x, y, z)] + u[g.F(x + 1, y, z)]), 0.5f * (v[g.F(x, y, z)] + v[g.F(x, y + 1, z)]),
          0.5f * (w[g.F(x, y, z)] + w[g.F(x, y, z + 1)])};
}

__global__ void Measure(Grid g,
                        const unsigned char *solid,
                        const unsigned char *bu,
                        const unsigned char *bv,
                        const unsigned char *bw,
                        const unsigned char *au,
                        const unsigned char *av,
                        const unsigned char *aw,
                        const float *u,
                        const float *v,
                        const float *w,
                        const float *old_u,
                        const float *old_v,
                        const float *old_w,
                        const float *p,
                        const float *before,
                        const float *after,
                        Physics physics,
                        Controls controls,
                        Statistics *blocks) {
  __shared__ Statistics values[256];
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  Statistics value;
  if (n < g.nx * g.ny * g.nz) {
    int x = n % g.nx, y = n / g.nx % g.ny, z = n / (g.nx * g.ny);
    Vec3 velocity = CenterVelocity(g, u, v, w, x, y, z);
    value.before = before[n] * before[n];
    value.after = after[n] * after[n];
    value.max_speed = Length(velocity);
    value.fastest = n;
    value.invalid = !isfinite(velocity.x + velocity.y + velocity.z + p[n] + before[n] + after[n]);
    if (!solid[n]) {
      const unsigned char *blocked[] = {bu, bv, bw}, *apertures[] = {au, av, aw};
      // Wetted-area vector of a cut cell, from the fluid side into the body.
      Vec3 wetted{}, total{};
      for (int axis = 0; axis < 3; ++axis)
        for (int sign = -1; sign <= 1; sign += 2) {
          int a = x + (axis == 0 && sign > 0), b = y + (axis == 1 && sign > 0), c = z + (axis == 2 && sign > 0);
          int face = g.F(a, b, c);
          float closed = 1 - Open(apertures[axis], face);
          if (closed <= 0)
            continue;
          Vec3 outward{axis == 0 ? float(sign) : 0, axis == 1 ? float(sign) : 0, axis == 2 ? float(sign) : 0};
          if (physics.cut_cells) {
            // Pressure acts on the closed part of each face of the cell.
            Vec3 force = Scale(outward, p[n] * g.h * g.h * closed);
            Vec3 point = Add(g.Position(x, y, z), Scale(outward, g.h * 0.5f));
            wetted = Add(wetted, Scale(outward, g.h * g.h * closed));
            total = Add(total, force);
            value.force = Add(value.force, force);
            value.moment = Add(value.moment, Cross(point, force));
            continue;
          }
          unsigned char flag = blocked[axis][face];
          // The explicit viscous update uses the pre-step velocities.
          Vec3 tangent = CenterVelocity(g, old_u, old_v, old_w, x, y, z);
          if (axis == 0)
            tangent.x = 0;
          if (axis == 1)
            tangent.y = 0;
          if (axis == 2)
            tangent.z = 0;
          float wall = WallViscosity(physics, Length(tangent), 0.5f * g.h) * WallArea(physics, flag);
          Vec3 shear = Scale(tangent, 2 * density * wall * g.h);
          Vec3 force = Add(Scale(outward, p[n] * g.h * g.h), shear);
          Vec3 point = Add(g.Position(x, y, z), Scale(outward, g.h * 0.5f));
          value.friction = Add(value.friction, shear);
          total = Add(total, force);
          value.force = Add(value.force, force);
          value.moment = Add(value.moment, Cross(point, force));
        }
      float area = Length(wetted);
      if (physics.cut_cells && area > 0) {
        // Log-law shear on the true wetted area, along the tangential velocity.
        Vec3 normal = Scale(wetted, 1 / area);
        Vec3 velocity = CenterVelocity(g, old_u, old_v, old_w, x, y, z);
        Vec3 tangent = Add(velocity, Scale(normal, -Dot(velocity, normal)));
        float wall = WallViscosity(physics, Length(tangent), 0.5f * g.h);
        Vec3 shear = Scale(tangent, 2 * density * wall * area / g.h);
        value.friction = Add(value.friction, shear);
        total = Add(total, shear);
        value.force = Add(value.force, shear);
        value.moment = Add(value.moment, Cross(g.Position(x, y, z), shear));
      }
      if (total.x != 0 || total.y != 0) {
        int component;
        AircraftDistance(g.Position(x, y, z), controls, component);
        value.drag[component] = total.x;
        value.lift[component] = total.y;
      }
    }
  }

  values[threadIdx.x] = value;
  __syncthreads();
  for (int offset = 128; offset; offset >>= 1) {
    if (threadIdx.x < offset) {
      auto &a = values[threadIdx.x];
      auto b = values[threadIdx.x + offset];
      a.before += b.before;
      a.after += b.after;
      if (b.max_speed > a.max_speed) {
        a.max_speed = b.max_speed;
        a.fastest = b.fastest;
      }
      a.force = Add(a.force, b.force);
      a.moment = Add(a.moment, b.moment);
      a.friction = Add(a.friction, b.friction);
      for (int k = 0; k < kComponents; ++k) {
        a.drag[k] += b.drag[k];
        a.lift[k] += b.lift[k];
      }
      a.invalid |= b.invalid;
    }
    __syncthreads();
  }
  if (!threadIdx.x)
    blocks[g.begin / 256 + blockIdx.x] = values[0];
}

struct Totals {
  double before, after, force[3], moment[3], friction[3], drag[kComponents], lift[kComponents];
  float max_speed;
  int fastest, invalid;
};

constexpr int total_blocks = 256;

// Second reduction stage in double precision, without atomics: each thread
// sums a strided subset of the single-precision block results.
__global__ void ReduceStatistics(const Statistics *blocks, int count, Totals *totals) {
  __shared__ Totals values[256];
  Totals value{};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    Statistics b = blocks[i];
    value.before += b.before;
    value.after += b.after;
    value.force[0] += b.force.x;
    value.force[1] += b.force.y;
    value.force[2] += b.force.z;
    value.moment[0] += b.moment.x;
    value.moment[1] += b.moment.y;
    value.moment[2] += b.moment.z;
    value.friction[0] += b.friction.x;
    value.friction[1] += b.friction.y;
    value.friction[2] += b.friction.z;
    for (int k = 0; k < kComponents; ++k) {
      value.drag[k] += b.drag[k];
      value.lift[k] += b.lift[k];
    }
    if (b.max_speed > value.max_speed) {
      value.max_speed = b.max_speed;
      value.fastest = b.fastest;
    }
    value.invalid |= b.invalid;
  }
  values[threadIdx.x] = value;
  __syncthreads();
  for (int offset = 128; offset; offset >>= 1) {
    if (threadIdx.x < offset) {
      auto &a = values[threadIdx.x];
      const auto &b = values[threadIdx.x + offset];
      a.before += b.before;
      a.after += b.after;
      for (int k = 0; k < 3; ++k) {
        a.force[k] += b.force[k];
        a.moment[k] += b.moment[k];
        a.friction[k] += b.friction[k];
      }
      for (int k = 0; k < kComponents; ++k) {
        a.drag[k] += b.drag[k];
        a.lift[k] += b.lift[k];
      }
      if (b.max_speed > a.max_speed) {
        a.max_speed = b.max_speed;
        a.fastest = b.fastest;
      }
      a.invalid |= b.invalid;
    }
    __syncthreads();
  }
  if (!threadIdx.x)
    totals[blockIdx.x] = values[0];
}

__global__ void
Export(Grid g, Grid output, const float *u, const float *v, const float *w, const float *p, Cell *cells) {
  int n = g.begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= output.nx * output.ny * output.nz)
    return;
  int x = n % output.nx, y = n / output.nx % output.ny, z = n / (output.nx * output.ny);
  float ratio = float(g.nx) / output.nx;
  Vec3 position{(x + 0.5f) * ratio, (y + 0.5f) * ratio, (z + 0.5f) * ratio};
  Cell cell{Sample(u, g, Add(position, {0, -0.5f, -0.5f})), Sample(v, g, Add(position, {-0.5f, 0, -0.5f})),
            Sample(w, g, Add(position, {-0.5f, -0.5f, 0})), 0};
  position = Add(position, {-0.5f, -0.5f, -0.5f});
  int ix = int(floorf(position.x)), iy = int(floorf(position.y)), iz = int(floorf(position.z));
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        int a = max(0, min(g.nx - 1, ix + i)), b = max(0, min(g.ny - 1, iy + j)), c = max(0, min(g.nz - 1, iz + k));
        cell.pressure += p[g.C(a, b, c)] * (i ? position.x - ix : 1 - position.x + ix) *
                         (j ? position.y - iy : 1 - position.y + iy) * (k ? position.z - iz : 1 - position.z + iz);
      }
  cells[n] = cell;
}
}  // namespace

size_t SolverBytes(int nx) {
  size_t ny = nx / 2, nz = nx * 3 / 4;
  size_t nf = (nx + 1) * (ny + 1) * (nz + 1), nc = nx * ny * nz;
  // Seven face velocity fields, three face masks, three cell fields, the cell
  // mask, block statistics, the pressure fine level and CG vectors, and the
  // multigrid hierarchy (a geometric series bounded by 1/7 of the fine level).
  size_t fluid = 7 * 4 * nf + 6 * nf + 3 * 4 * nc + nc + (nc + 255) / 256 * sizeof(Statistics);
  size_t pressure = (sizeof(float4) + 3 * 4) * nc * 8 / 7 + 3 * 4 * nc;
  size_t display = size_t(192) * 96 * 144 * sizeof(Cell);
  return fluid + pressure + display + (size_t(256) << 20);
}

bool ValidateClosedSurface(int cuda_device, bool cut_cells) {
  // A uniform pressure on the closed aircraft surface, with no flow, must give
  // no net force. This checks that the integrated faces (or closed face
  // fractions) form a watertight surface, including thin walls and deflected
  // controls.
  const Grid g{128, 64, 96, 16.0f / 128};
  const int nc = g.nx * g.ny * g.nz, nf = (g.nx + 1) * (g.ny + 1) * (g.nz + 1), blocks = (nc + 255) / 256;
  const Controls controls{30, 4, 25, 15, -10, 10};
  const Physics physics{1.5e-5f, 1.5e-5f, true, !cut_cells, cut_cells};
  float *fields = nullptr;
  unsigned char *masks = nullptr;
  Statistics *statistics = nullptr;
  bool passed = cudaSetDevice(cuda_device) == cudaSuccess &&
                cudaMallocManaged(&fields, (size_t(3) * nf + nc) * sizeof(float)) == cudaSuccess &&
                cudaMallocManaged(&masks, size_t(6) * nf + nc) == cudaSuccess &&
                cudaMallocManaged(&statistics, blocks * sizeof(Statistics)) == cudaSuccess;
  double force[3]{}, scale = 0;
  if (passed) {
    std::fill(fields, fields + 3 * nf, 0.0f);
    std::fill(fields + 3 * nf, fields + 3 * nf + nc, 1000.0f);
    unsigned char *solid = masks + 6 * nf, *apertures = masks + 3 * nf;
    Mask<<<blocks, 256>>>(g, solid, controls);
    FaceMask<<<(nf + 255) / 256, 256>>>(g, solid, masks, masks + nf, masks + 2 * nf, apertures, apertures + nf,
                                        apertures + 2 * nf, controls, cut_cells);
    // Check the underresolved-sheet flags before CellMask replaces them with
    // active-cell occupancy. All six connections must remain closed.
    passed = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    int thin_cells = 0;
    for (int i = 0; passed && i < nc; ++i)
      if (solid[i] & 2) {
        ++thin_cells;
        int x = i % g.nx, y = i / g.nx % g.ny, z = i / (g.nx * g.ny);
        for (int axis = 0; axis < 3; ++axis)
          if (apertures[axis * nf + g.F(x, y, z)] ||
              apertures[axis * nf + g.F(x + (axis == 0), y + (axis == 1), z + (axis == 2))])
            passed = false;
      }
    passed = passed && thin_cells > 0;
    std::cout << "thin_cell_connections=" << (passed ? "PASS" : "FAIL") << " cells=" << thin_cells << '\n';
    if (cut_cells)
      CellMask<<<blocks, 256>>>(g, solid, apertures, apertures + nf, apertures + 2 * nf);
    float *p = fields + 3 * nf;
    Measure<<<blocks, 256>>>(g, solid, masks, masks + nf, masks + 2 * nf, apertures, apertures + nf, apertures + 2 * nf,
                             fields, fields + nf, fields + 2 * nf, fields, fields + nf, fields + 2 * nf, p, p, p,
                             physics, controls, statistics);
    passed = passed && cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    for (int i = 0; passed && i < blocks; ++i) {
      force[0] += statistics[i].force.x;
      force[1] += statistics[i].force.y;
      force[2] += statistics[i].force.z;
    }
    // Reference: the pressure load on one side of the projected body.
    for (int i = 0; passed && i < nf; ++i)
      scale += masks[i] ? 1000.0 * g.h * g.h : 0;
  }
  double residual = std::sqrt(force[0] * force[0] + force[1] * force[1] + force[2] * force[2]);
  passed = passed && scale > 0 && residual < 1e-4 * scale;
  cudaFree(statistics);
  cudaFree(masks);
  cudaFree(fields);
  std::cout << "closed_surface_validation" << (cut_cells ? "_cut" : "") << '=' << (passed ? "PASS" : "FAIL")
            << " net=" << residual << " N reference=" << scale << " N\n";
  return passed;
}

bool ValidateWallDiffusion(int cuda_device) {
  // A zero-thickness stationary plate separates fluid cells on both sides.
  // Check all six normal/tangent combinations against the analytical shear
  // and against the actual fluid momentum loss from CorrectTransport, for
  // laminar no-slip diffusion and for the log-law wall model.
  const Grid g{8, 8, 8, 0.25f};
  constexpr int nf = 9 * 9 * 9, nc = 8 * 8 * 8;
  float *fields = nullptr;
  unsigned char *masks = nullptr;
  Statistics *statistics = nullptr;
  bool passed = cudaSetDevice(cuda_device) == cudaSuccess &&
                cudaMallocManaged(&fields, (4 * nf + nc) * sizeof(float)) == cudaSuccess &&
                cudaMallocManaged(&masks, 6 * nf + nc) == cudaSuccess &&
                cudaMallocManaged(&statistics, 2 * sizeof(Statistics)) == cudaSuccess;
  const Physics models[] = {{0.005f, 1.5e-5f, false, true, false}, {1.5e-5f, 1.5e-5f, true, true, false}};
  for (int model = 0; model < 2; ++model)
    for (int normal = 0; passed && normal < 3; ++normal)
      for (int tangent = 0; passed && tangent < 3; ++tangent) {
        if (normal == tangent)
          continue;
        const Physics physics = models[model];
        const float wall = WallViscosity(physics, 3, 0.5f * g.h);
        const float dt = 0.1f * g.h * g.h / std::max(physics.viscosity, wall);
        std::fill(fields, fields + 4 * nf + nc, 0.0f);
        std::fill(masks, masks + 3 * nf + nc, 0);
        float *source = fields + tangent * nf, *output = fields + 3 * nf, *zero = fields + 4 * nf;
        std::fill(source, source + nf, 3.0f);
        std::fill(output, output + nf, 3.0f);
        for (int a = 2; a < 6; ++a)
          for (int b = 2; b < 6; ++b) {
            int face[3];
            face[normal] = 4;
            face[(normal + 1) % 3] = a;
            face[(normal + 2) % 3] = b;
            masks[normal * nf + g.F(face[0], face[1], face[2])] = 255;
          }
        unsigned char *apertures = masks + 3 * nf + nc;
        for (int i = 0; i < 3 * nf; ++i)
          apertures[i] = masks[i] ? 0 : 255;
        CorrectTransport<<<(nf + 255) / 256, 256>>>(g, fields, fields + nf, fields + 2 * nf, source, nullptr, output,
                                                    masks, masks + nf, masks + 2 * nf, tangent, dt, physics);
        Measure<<<2, 256>>>(g, masks + 3 * nf, masks, masks + nf, masks + 2 * nf, apertures, apertures + nf,
                            apertures + 2 * nf, fields, fields + nf, fields + 2 * nf, fields, fields + nf,
                            fields + 2 * nf, zero, zero, zero, physics, Controls{}, statistics);
        passed = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
        if (!passed)
          break;
        double momentum_loss = 0;
        for (int i = 0; i < nf; ++i)
          momentum_loss += source[i] - output[i];
        momentum_loss *= density * g.h * g.h * g.h / dt;
        Vec3 force = Add(statistics[0].force, statistics[1].force);
        float measured = tangent == 0 ? force.x : tangent == 1 ? force.y : force.z;
        float expected = 2 * 16 * 2 * density * wall * g.h * 3;
        float tolerance = 1e-4f * expected + 1e-6f;
        passed = std::abs(measured - expected) < tolerance && std::abs(momentum_loss - expected) < tolerance;
        if (!passed)
          std::cerr << "Wall diffusion mismatch: model=" << model << " normal=" << normal << " tangent=" << tangent
                    << " fluid=" << momentum_loss << " wall=" << measured << " expected=" << expected << '\n';
      }
  cudaFree(statistics);
  cudaFree(masks);
  cudaFree(fields);
  std::cout << "wall_diffusion_validation=" << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}

struct Simulation::Impl {
  mutable std::mutex mutex;
  std::condition_variable wake;
  std::thread worker;
  bool stop{}, paused{}, working{};
  uint64_t generation{};
  Controls requested;
  std::shared_ptr<const Snapshot> latest;
  std::deque<std::shared_ptr<const Snapshot>> pending_snapshots;
  std::deque<LoadSample> load_history;
  bool load_capacity_limited{};
  std::string error;
  Grid grid{};
  pressure::Solver pressure_solver;
  SolverOptions options;
  int device{};
  cudaStream_t stream{};
  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> ready;
  cudaEvent_t timing[4]{};
  float *u{}, *v{}, *w{}, *un{}, *vn{}, *wn{}, *reverse{}, *pressure{}, *div{}, *after{};
  unsigned char *solid{}, *bu{}, *bv{}, *bw{}, *au{}, *av{}, *aw{};
  Cell *cells{};
  Statistics *statistics{};
  Totals *totals{};
  Grid display_grid{};
  std::vector<void *> allocations;

  bool Check(cudaError_t result) {
    if (result == cudaSuccess)
      return true;
    std::lock_guard<std::mutex> guard(mutex);
    error = cudaGetErrorString(result);
    return false;
  }

  template <typename T>
  bool Allocate(T *&p, size_t count, bool distributed = true) {
    if (!Check(options.devices == 1 ? cudaMalloc(reinterpret_cast<void **>(&p), count * sizeof(T))
                                    : cudaMallocManaged(reinterpret_cast<void **>(&p), count * sizeof(T))))
      return false;
    allocations.push_back(p);
    if (!Check(cudaMemsetAsync(p, 0, count * sizeof(T), stream)))
      return false;
    if (options.devices > 1) {
      if (!Check(cudaStreamSynchronize(stream)))
        return false;
      // Keep pages on their owning GPU; neighbours access them directly over NVLink.
      const size_t bytes = count * sizeof(T), page = 65536;
      const size_t pages = (bytes + page - 1) / page;
      int owners = distributed ? options.devices : 1;
      for (int d = 0; d < owners; ++d) {
        size_t begin = pages * d / owners * page;
        size_t end = std::min(bytes, pages * (d + 1) / owners * page);
        if (begin >= end)
          continue;
        auto address = reinterpret_cast<char *>(p) + begin;
        if (!Check(cudaMemAdvise(address, end - begin, cudaMemAdviseSetPreferredLocation, Location(device + d))))
          return false;
        for (int peer = 0; peer < options.devices; ++peer)
          if (!Check(cudaMemAdvise(address, end - begin, cudaMemAdviseSetAccessedBy, Location(device + peer))))
            return false;
#if CUDART_VERSION >= 13000
        if (!Check(cudaMemPrefetchAsync(address, end - begin, Location(device + d), 0, stream)))
          return false;
#else
        if (!Check(cudaMemPrefetchAsync(address, end - begin, device + d, stream)))
          return false;
#endif
      }
    }
    return true;
  }

  bool Initialize() {
    if (!Check(cudaSetDevice(device)))
      return false;
    if (options.devices > 1) {
      for (int d = 0; d < options.devices; ++d) {
        int concurrent = 0;
        if (!Check(cudaDeviceGetAttribute(&concurrent, cudaDevAttrConcurrentManagedAccess, device + d)))
          return false;
        if (!concurrent)
          return Check(cudaErrorNotSupported);
        for (int peer = 0; peer < options.devices; ++peer) {
          if (peer == d)
            continue;
          int accessible = 0;
          if (!Check(cudaDeviceCanAccessPeer(&accessible, device + d, device + peer)))
            return false;
          if (!accessible)
            return Check(cudaErrorPeerAccessUnsupported);
        }
      }
    }
    int low = 0, high = 0;
    if (!Check(cudaDeviceGetStreamPriorityRange(&low, &high)) ||
        !Check(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, low)))
      return false;
    streams.push_back(stream);
    for (auto &event : timing)
      if (!Check(cudaEventCreate(&event)))
        return false;
    for (int d = 0; d < options.devices; ++d) {
      if (!Check(cudaSetDevice(device + d)))
        return false;
      if (d) {
        cudaStream_t extra{};
        if (!Check(cudaStreamCreateWithFlags(&extra, cudaStreamNonBlocking)))
          return false;
        streams.push_back(extra);
      }
      cudaEvent_t event{};
      if (!Check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming)))
        return false;
      ready.push_back(event);
    }
    if (!Check(cudaSetDevice(device)))
      return false;
    size_t nf = size_t(grid.nx + 1) * (grid.ny + 1) * (grid.nz + 1);
    size_t nc = size_t(grid.nx) * grid.ny * grid.nz;
    int display_nx = std::min(grid.nx, 192);
    display_grid = {display_nx, display_nx / 2, display_nx * 3 / 4, 16.0f / display_nx};
    size_t display_cells = size_t(display_grid.nx) * display_grid.ny * display_grid.nz;
    return Allocate(u, nf) && Allocate(v, nf) && Allocate(w, nf) && Allocate(un, nf) && Allocate(vn, nf) &&
           Allocate(wn, nf) && Allocate(reverse, nf) && Allocate(pressure, nc, false) && Allocate(div, nc) &&
           Allocate(after, nc) && Allocate(solid, nc) && Allocate(bu, nf) && Allocate(bv, nf) && Allocate(bw, nf) &&
           Allocate(au, nf) && Allocate(av, nf) && Allocate(aw, nf) &&
           (!(options.display && (options.export_field || options.export_final)) ||
            Allocate(cells, display_cells, false)) &&
           Allocate(statistics, (nc + 255) / 256) && Allocate(totals, total_blocks, false) &&
           Check(pressure_solver.Initialize(grid.nx, grid.ny, grid.nz, stream)) && Check(cudaStreamSynchronize(stream));
  }

  template <class Launch>
  bool Parallel(int blocks, Launch launch) {
    if (options.devices == 1) {
      if (blocks) {
        Grid local = grid;
        local.cubic = options.cubic;
        if (options.cut_cells) {
          local.aperture[0] = au;
          local.aperture[1] = av;
          local.aperture[2] = aw;
        }
        launch(local, blocks, stream);
      }
      return Check(cudaGetLastError());
    }
    for (int d = 0; d < options.devices; ++d) {
      if (!Check(cudaSetDevice(device + d)))
        return false;
      Grid local = grid;
      local.cubic = options.cubic;
      if (options.cut_cells) {
        local.aperture[0] = au;
        local.aperture[1] = av;
        local.aperture[2] = aw;
      }
      int begin = blocks * d / options.devices;
      int end = blocks * (d + 1) / options.devices;
      local.begin = begin * 256;
      if (end > begin)
        launch(local, end - begin, streams[d]);
      if (!Check(cudaGetLastError()) || !Check(cudaEventRecord(ready[d], streams[d])))
        return false;
    }
    if (!Check(cudaSetDevice(device)))
      return false;
    for (int d = 1; d < options.devices; ++d)
      if (!Check(cudaStreamWaitEvent(stream, ready[d])))
        return false;
    if (!Check(cudaEventRecord(ready[0], stream)))
      return false;
    for (int d = 1; d < options.devices; ++d) {
      if (!Check(cudaSetDevice(device + d)) || !Check(cudaStreamWaitEvent(streams[d], ready[0])))
        return false;
    }
    return Check(cudaSetDevice(device));
  }

  void Run() {
    if (!Check(cudaSetDevice(device)))
      return;
    auto g = grid;
    if (options.cut_cells) {
      g.aperture[0] = au;
      g.aperture[1] = av;
      g.aperture[2] = aw;
    }
    int nc = g.nx * g.ny * g.nz, nf = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
    int cb = (nc + 255) / 256, fb = (nf + 255) / 256;
    std::vector<Totals> measurements(total_blocks);
    size_t display_count = options.display ? size_t(display_grid.nx) * display_grid.ny * display_grid.nz : 0;
    const Physics physics{options.viscosity, options.wall_viscosity, options.wall_model,
                          !options.wall_model || options.wall_momentum, options.cut_cells};
    double impulse[3]{}, angular_impulse[3]{}, friction_impulse[3]{}, drag_impulse[kComponents]{},
        lift_impulse[kComponents]{};
    uint64_t applied_generation = ~uint64_t(0), step = 0;
    float time = 0, max_speed = 12;
    Controls mesh_controls;
    std::shared_ptr<const std::vector<Vertex>> mesh;
    while (true) {
      Controls controls;
      uint64_t current_generation;
      {
        std::unique_lock<std::mutex> lock(mutex);
        wake.wait(lock, [&] {
          return stop || generation != applied_generation ||
                 (!paused && (!options.retain_snapshots || pending_snapshots.size() < 2));
        });
        if (stop)
          break;
        controls = requested;
        current_generation = generation;
        working = true;
      }
      auto start = std::chrono::steady_clock::now();
      bool reset = current_generation != applied_generation;
      if (reset) {
        step = 0;
        time = 0;
        max_speed = controls.speed;
        std::fill(impulse, impulse + 3, 0.0);
        std::fill(angular_impulse, angular_impulse + 3, 0.0);
        std::fill(friction_impulse, friction_impulse + 3, 0.0);
        std::fill(drag_impulse, drag_impulse + kComponents, 0.0);
        std::fill(lift_impulse, lift_impulse + kComponents, 0.0);
        if (!Check(cudaMemsetAsync(pressure, 0, size_t(nc) * sizeof(float), stream)))
          break;
      }
      applied_generation = current_generation;
      // No-slip ghost values can increase the diffusion diagonal from six
      // to ten at corners; retain a conservative explicit stability bound.
      float speed_bound = std::max(max_speed, controls.speed);
      float diffusion = std::max(physics.viscosity, WallViscosity(physics, 2 * speed_bound, 0.5f * g.h));
      float dt = std::min(options.cfl * g.h / speed_bound, 0.08f * g.h * g.h / diffusion);
      if (options.max_dt > 0)
        dt = std::min(dt, options.max_dt);
      bool geometry_changed = !mesh || controls.angle != mesh_controls.angle || controls.flap != mesh_controls.flap ||
                              controls.aileron != mesh_controls.aileron ||
                              controls.elevator != mesh_controls.elevator || controls.rudder != mesh_controls.rudder;
      if (geometry_changed) {
        if (!Parallel(cb, [&](Grid g, int blocks, cudaStream_t queue) {
              Mask<<<blocks, 256, 0, queue>>>(g, solid, controls);
            }))
          break;
        if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
              FaceMask<<<blocks, 256, 0, queue>>>(g, solid, bu, bv, bw, au, av, aw, controls, physics.cut_cells);
            }))
          break;
        if (physics.cut_cells)
          if (!Parallel(cb, [&](Grid g, int blocks, cudaStream_t queue) {
                CellMask<<<blocks, 256, 0, queue>>>(g, solid, au, av, aw);
              }))
            break;
        if (!Check(pressure_solver.Build(solid, au, av, aw)))
          break;
      }
      if (!Check(cudaEventRecord(timing[0], stream)))
        break;
      if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
            Boundary<<<blocks, 256, 0, queue>>>(g, bu, bv, bw, u, v, w, controls.speed, reset);
          }))
        break;
      auto extend = [&](float *x, float *y, float *z) {
        if (!physics.wall_model)
          return true;
        const unsigned char *masks[] = {bu, bv, bw};
        float *fields[] = {x, y, z};
        for (int layer = 1; layer <= 2; ++layer)
          if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
                for (int axis = 0; axis < 3; ++axis)
                  Extrapolate<<<blocks, 256, 0, queue>>>(g, masks[axis], fields[axis], layer);
              }))
            return false;
        return true;
      };
      if (!extend(u, v, w))
        break;
      if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
            Advect<<<blocks, 256, 0, queue>>>(g, u, v, w, u, un, 0, dt);
          }))
        break;
      if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
            Advect<<<blocks, 256, 0, queue>>>(g, u, v, w, v, vn, 1, dt);
          }))
        break;
      if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
            Advect<<<blocks, 256, 0, queue>>>(g, u, v, w, w, wn, 2, dt);
          }))
        break;
      if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
            Boundary<<<blocks, 256, 0, queue>>>(g, bu, bv, bw, un, vn, wn, controls.speed, false);
          }))
        break;
      if (options.corrected_transport && !extend(un, vn, wn))
        break;
      float *sources[] = {u, v, w}, *destinations[] = {un, vn, wn};
      for (int axis = 0; axis < 3; ++axis) {
        if (options.corrected_transport)
          if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
                Advect<<<blocks, 256, 0, queue>>>(g, u, v, w, destinations[axis], reverse, axis, -dt);
              }))
            break;
        if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
              CorrectTransport<<<blocks, 256, 0, queue>>>(g, u, v, w, sources[axis],
                                                          options.corrected_transport ? reverse : nullptr,
                                                          destinations[axis], bu, bv, bw, axis, dt, physics);
            }))
          break;
      }
      if (!error.empty())
        break;
      std::swap(u, un);
      std::swap(v, vn);
      std::swap(w, wn);
      if (options.sponge > 0)
        if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
              Sponge<<<blocks, 256, 0, queue>>>(g, u, v, w, controls.speed, options.sponge);
            }))
          break;
      if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
            Boundary<<<blocks, 256, 0, queue>>>(g, bu, bv, bw, u, v, w, controls.speed, false);
          }))
        break;
      if (!Parallel(cb, [&](Grid g, int blocks, cudaStream_t queue) {
            Divergence<<<blocks, 256, 0, queue>>>(g, solid, au, av, aw, u, v, w, div);
          }))
        break;
      if (!Check(cudaEventRecord(timing[1], stream)))
        break;
      int pressure_cycles = 0;
      float pressure_residual = 0;
      if (!Check(pressure_solver.Solve(div, pressure, density * g.h * g.h / dt, options.pressure_tolerance,
                                       options.pressure_cycles, pressure_cycles, pressure_residual)))
        break;
      if (!Check(cudaEventRecord(timing[2], stream)))
        break;
      if (!Parallel(0, [&](Grid, int, cudaStream_t) {}))
        break;
      if (!Parallel(fb, [&](Grid g, int blocks, cudaStream_t queue) {
            Project<<<blocks, 256, 0, queue>>>(g, bu, bv, bw, pressure, u, v, w, dt);
          }))
        break;
      if (!Parallel(cb, [&](Grid g, int blocks, cudaStream_t queue) {
            Divergence<<<blocks, 256, 0, queue>>>(g, solid, au, av, aw, u, v, w, after);
          }))
        break;
      if (!Parallel(cb, [&](Grid g, int blocks, cudaStream_t queue) {
            Measure<<<blocks, 256, 0, queue>>>(g, solid, bu, bv, bw, au, av, aw, u, v, w, un, vn, wn, pressure, div,
                                               after, physics, controls, statistics);
          }))
        break;
      ReduceStatistics<<<total_blocks, 256, 0, stream>>>(statistics, cb, totals);
      auto snapshot = std::make_shared<Snapshot>();
      bool export_field =
          options.export_field || (options.export_final && options.stop_after && step + 1 == options.stop_after);
      if (display_count && export_field) {
        Export<<<(display_count + 255) / 256, 256, 0, stream>>>(g, display_grid, u, v, w, pressure, cells);
        snapshot->cells.resize(display_count);
        if (!Check(cudaMemcpyAsync(snapshot->cells.data(), cells, display_count * sizeof(Cell), cudaMemcpyDeviceToHost,
                                   stream)))
          break;
      }
      if (!Check(cudaGetLastError()) ||
          !Check(cudaMemcpyAsync(measurements.data(), totals, measurements.size() * sizeof(Totals),
                                 cudaMemcpyDeviceToHost, stream)) ||
          !Check(cudaEventRecord(timing[3], stream)) || !Check(cudaStreamSynchronize(stream)))
        break;
      float milliseconds[3]{};
      for (int i = 0; i < 3; ++i)
        if (!Check(cudaEventElapsedTime(&milliseconds[i], timing[i], timing[i + 1])))
          return;
      snapshot->transport_seconds = milliseconds[0] * 0.001f;
      snapshot->pressure_seconds = milliseconds[1] * 0.001f;
      snapshot->diagnostics_seconds = milliseconds[2] * 0.001f;
      double before_sum = 0, after_sum = 0, force[3]{}, moment[3]{}, friction[3]{}, drag[kComponents]{},
             lift[kComponents]{};
      max_speed = controls.speed;
      for (auto value : measurements) {
        before_sum += value.before;
        after_sum += value.after;
        if (value.max_speed > max_speed) {
          max_speed = value.max_speed;
          int n = value.fastest;
          snapshot->fastest = {(n % g.nx + 0.5f) * g.h - 6, (n / g.nx % g.ny + 0.5f) * g.h - 4,
                               (n / (g.nx * g.ny) + 0.5f) * g.h - 6};
        }
        if (value.invalid) {
          std::lock_guard<std::mutex> guard(mutex);
          error = "Non-finite fluid state";
          return;
        }
        for (int k = 0; k < 3; ++k) {
          force[k] += value.force[k];
          friction[k] += value.friction[k];
          moment[k] += value.moment[k];
        }
        for (int k = 0; k < kComponents; ++k) {
          drag[k] += value.drag[k];
          lift[k] += value.lift[k];
        }
      }
      for (int k = 0; k < 3; ++k) {
        impulse[k] += force[k] * dt;
        friction_impulse[k] += friction[k] * dt;
        snapshot->friction_impulse[k] = friction_impulse[k];
        angular_impulse[k] += moment[k] * dt;
        snapshot->impulse[k] = impulse[k];
        snapshot->angular_impulse[k] = angular_impulse[k];
      }
      for (int k = 0; k < kComponents; ++k) {
        drag_impulse[k] += drag[k] * dt;
        lift_impulse[k] += lift[k] * dt;
        snapshot->drag_impulse[k] = drag_impulse[k];
        snapshot->lift_impulse[k] = lift_impulse[k];
      }
      snapshot->max_speed = max_speed;
      snapshot->force = {float(force[0]), float(force[1]), float(force[2])};
      snapshot->moment = {float(moment[0]), float(moment[1]), float(moment[2])};
      snapshot->solver_nx = g.nx;
      snapshot->pressure_converged = pressure_residual <= options.pressure_tolerance;
      snapshot->pressure_cycles = pressure_cycles;
      snapshot->pressure_residual = pressure_residual;
      if (display_count) {
        snapshot->nx = display_grid.nx;
        snapshot->ny = display_grid.ny;
        snapshot->nz = display_grid.nz;
      }
      snapshot->step = ++step;
      snapshot->generation = current_generation;
      snapshot->time = time += dt;
      snapshot->dt = dt;
      snapshot->controls = controls;
      if (!mesh || controls.angle != mesh_controls.angle || controls.flap != mesh_controls.flap ||
          controls.aileron != mesh_controls.aileron || controls.elevator != mesh_controls.elevator ||
          controls.rudder != mesh_controls.rudder) {
        mesh = std::make_shared<const std::vector<Vertex>>(AircraftMesh(controls));
        mesh_controls = controls;
      }
      snapshot->aircraft = mesh;
      snapshot->divergence_before = float(std::sqrt(before_sum / nc));
      snapshot->divergence_after = float(std::sqrt(after_sum / nc));
      snapshot->step_seconds = std::chrono::duration<float>(std::chrono::steady_clock::now() - start).count();
      {
        std::lock_guard<std::mutex> guard(mutex);
        if (generation == current_generation) {
          if (options.record_loads) {
            load_history.push_back(
                {snapshot->step, snapshot->time, snapshot->dt, snapshot->force, snapshot->moment, snapshot->controls});
            while (load_history.size() > 1 && load_history[1].time <= snapshot->time - 2.0)
              load_history.pop_front();
            load_capacity_limited = load_history.size() > 65536;
            if (load_capacity_limited)
              load_history.pop_front();
          }
          if (options.retain_snapshots)
            pending_snapshots.push_back(snapshot);
          latest = std::move(snapshot);
        }
        working = false;
        if (options.stop_after && step >= options.stop_after)
          paused = true;
      }
    }
  }

  ~Impl() {
    {
      std::lock_guard<std::mutex> guard(mutex);
      stop = true;
    }
    wake.notify_one();
    if (worker.joinable())
      worker.join();
    cudaSetDevice(device);
    for (int d = 0; d < int(streams.size()); ++d) {
      cudaSetDevice(device + d);
      cudaStreamSynchronize(streams[d]);
    }
    cudaSetDevice(device);
    for (auto event : timing)
      if (event)
        cudaEventDestroy(event);
    pressure_solver.Release();
    for (auto allocation : allocations)
      cudaFree(allocation);
    for (int d = 0; d < int(streams.size()); ++d) {
      cudaSetDevice(device + d);
      if (d < int(ready.size()))
        cudaEventDestroy(ready[d]);
      cudaStreamDestroy(streams[d]);
    }
    cudaSetDevice(device);
  }
};

Simulation::Simulation() : impl_(std::make_unique<Impl>()) {
}

Simulation::~Simulation() = default;

std::unique_ptr<Simulation> Simulation::Create(int cuda_device, int nx, const SolverOptions &options) {
  int available = 0;
  if (cudaGetDeviceCount(&available) != cudaSuccess || options.devices < 1 || options.devices > 8 || cuda_device < 0 ||
      cuda_device >= available || options.devices > available - cuda_device)
    return {};
  if (cuda_device < 0 || nx < 64 || nx > max_grid || nx % 16 || options.pressure_tolerance < 1e-6f ||
      options.pressure_tolerance > 0.1f || options.pressure_cycles < 1 || options.pressure_cycles > 256 ||
      !(options.viscosity >= 1e-6f && options.viscosity <= 0.1f) ||
      !(options.wall_viscosity >= 1e-8f && options.wall_viscosity <= 0.1f) ||
      !(options.cfl >= 0.05f && options.cfl <= 8) || !(options.sponge >= 0 && options.sponge <= 6) ||
      (options.cut_cells && (!options.wall_model || options.wall_momentum)))
    return {};
  if (options.max_dt < 0 || !std::isfinite(options.max_dt) || !(options.initial_controls.speed > 0) ||
      !std::isfinite(options.initial_controls.speed) || !std::isfinite(options.initial_controls.angle) ||
      !std::isfinite(options.initial_controls.flap) || !std::isfinite(options.initial_controls.aileron) ||
      !std::isfinite(options.initial_controls.elevator) || !std::isfinite(options.initial_controls.rudder))
    return {};
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaSetDevice(cuda_device) != cudaSuccess || cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess ||
      SolverBytes(nx) > free_bytes) {
    std::cerr << "Grid " << nx << " needs about " << SolverBytes(nx) / 1e9 << " GB; " << free_bytes / 1e9
              << " GB is free\n";
    return {};
  }
  auto result = std::unique_ptr<Simulation>(new Simulation);
  result->impl_->device = cuda_device;
  result->impl_->options = options;
  result->impl_->requested = options.initial_controls;
  result->impl_->grid = {nx, nx / 2, nx * 3 / 4, 16.0f / nx};
  if (!result->impl_->Initialize())
    return {};
  result->impl_->worker = std::thread([p = result->impl_.get()] { p->Run(); });
  return result;
}

void Simulation::Configure(const Controls &controls, bool paused) {
  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->requested = controls;
    impl_->paused = paused;
  }
  impl_->wake.notify_one();
}

void Simulation::Reset() {
  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    ++impl_->generation;
    impl_->load_history.clear();
    impl_->pending_snapshots.clear();
    impl_->load_capacity_limited = false;
  }
  impl_->wake.notify_one();
}

bool Simulation::Paused() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->paused && !impl_->working;
}

std::shared_ptr<const Snapshot> Simulation::Latest() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->latest;
}

std::shared_ptr<const Snapshot> Simulation::TakeNext() {
  std::shared_ptr<const Snapshot> result;
  {
    std::lock_guard<std::mutex> guard(impl_->mutex);
    if (!impl_->pending_snapshots.empty()) {
      result = std::move(impl_->pending_snapshots.front());
      impl_->pending_snapshots.pop_front();
    }
  }
  if (result)
    impl_->wake.notify_one();
  return result;
}

LoadHistory Simulation::RecentLoads() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return {impl_->generation, impl_->load_capacity_limited,
          std::vector<LoadSample>(impl_->load_history.begin(), impl_->load_history.end())};
}

std::string Simulation::Error() const {
  std::lock_guard<std::mutex> guard(impl_->mutex);
  return impl_->error;
}
}  // namespace wind3d
