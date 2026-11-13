// MAC-grid compute kernels. Dispatches are ordered by RHI resource dependencies.
// Buffer views contain float-element offsets into one device-local allocation.
RHI_RW_BYTE_ADDRESS_BUFFER(0, 0, heap);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 1, display);

struct View {
  uint offset;

  __subscript(int index)->float {
    get {
      return heap.Load<float>((offset + index) * 4);
    }
    set {
      heap.Store<float>((offset + index) * 4, newValue);
    }
  }
};

struct View4 {
  __subscript(int index)->float4 {
    get {
      return display.Load<float4>(index * 16);
    }
    set {
      display.Store<float4>(index * 16, newValue);
    }
  }
};

View V(uint offset) {
  View v;
  v.offset = offset;
  return v;
}

float Hypot(float x, float y) {
  return length(float2(x, y));
}

struct DispatchSettings {
  uint4 level0, level1, coarse0, coarse1;
  uint4 fields0, fields1;
  float4 values0, values1;
  int4 controls;
  uint4 scratch;
};

RHI_UNIFORM(0, 2, DispatchSettings, settings);

struct Level {
  int w, h;
  View p, b, r, east, north, boundary;
};

bool Fluid(View mask, int x, int y, int w, int h) {
  return x >= 0 && x < w && y >= 0 && y < h && mask[y * w + x];
}

// An implicit function with the selected profile zero contour (not a distance field).
float WingImplicit(float x, float y, float angle, int profile) {
  float a = angle * 0.01745329252f;
  x -= 2;
  y -= 1.5f;
  float q = (cos(a) * x - sin(a) * y) / 1.25f + 0.5f;
  float t = (sin(a) * x + cos(a) * y) / 1.25f;
  return max(abs(t - WingCamber(q, profile)) - WingThickness(q, profile), max(-q, q - 1)) * 1.25f;
}

float OpenFraction(float x, float y, float sx, float sy, float angle, int profile) {
  float fraction = 0;
  float a = WingImplicit(x, y, angle, profile);
  // Subsegments also detect the thin leading/trailing edge when both face endpoints are outside.
  for (int k = 1; k <= 8; ++k) {
    float b = WingImplicit(x + sx * (k / 8.0f), y + sy * (k / 8.0f), angle, profile);
    fraction += a >= 0 && b >= 0 ? 1 : a <= 0 && b <= 0 ? 0 : a > 0 ? a / (a - b) : b / (b - a);
    a = b;
  }
  fraction *= 0.125f;
  // Remove vanishing apertures to keep the pressure system well conditioned.
  return fraction < 0.01f ? 0 : fraction > 0.99f ? 1 : fraction;
}

void SmoothGeometry(int tid, Level l, float angle, int profile) {
  int i = tid;
  if (i >= l.w * l.h)
    return;
  float h = 6.0f / l.w;
  int x = i % l.w, y = i / l.w;
  l.east[i] = x + 1 < l.w ? OpenFraction((x + 1) * h, y * h, 0, h, angle, profile) : 0;
  l.north[i] = y + 1 < l.h ? OpenFraction(x * h, (y + 1) * h, h, 0, angle, profile) : 0;
  l.boundary[i] = x == l.w - 1 ? 2 : 0;
}

void ActiveCells(int tid, Level l, View mask) {
  int i = tid;
  if (i < l.w * l.h)
    mask[i] =
        l.east[i] + l.north[i] + l.boundary[i] + (i % l.w ? l.east[i - 1] : 1) + (i >= l.w ? l.north[i - l.w] : 0) > 0;
}

float UWeight(Level l, int x, int y) {
  return x == 0 || x == l.w ? 1 : l.east[y * l.w + x - 1];
}

float VWeight(Level l, int x, int y) {
  return y == 0 || y == l.h ? 0 : l.north[(y - 1) * l.w + x];
}

float Flux(Level l, View u, View v, int x, int y) {
  int j = y * (l.w + 1) + x, i = y * l.w + x;
  return UWeight(l, x + 1, y) * u[j + 1] - UWeight(l, x, y) * u[j] + VWeight(l, x, y + 1) * v[i + l.w] -
         VWeight(l, x, y) * v[i];
}

void Geometry(int tid, View mask, int w, int h, float angle, int profile) {
  int i = tid;
  if (i >= w * h)
    return;
  float dx = 6.0f / w;
  mask[i] = WingImplicit((i % w + 0.5f) * dx, (i / w + 0.5f) * dx, angle, profile) >= 0;
}

void FineOperator(int tid, Level l, View mask) {
  int i = tid;
  if (i >= l.w * l.h)
    return;
  int x = i % l.w, y = i / l.w;
  bool f = mask[i];
  l.east[i] = f && Fluid(mask, x + 1, y, l.w, l.h) ? 1 : 0;
  l.north[i] = f && Fluid(mask, x, y + 1, l.w, l.h) ? 1 : 0;
  // Outflow pressure is zero at the face, half a cell from the last pressure node.
  l.boundary[i] = f && x == l.w - 1 ? 2 : 0;
}

void CoarsenOperator(int tid, Level fine, Level coarse) {
  int i = tid;
  if (i >= coarse.w * coarse.h)
    return;
  int x = (i % coarse.w) * 2, y = (i / coarse.w) * 2, j = y * fine.w + x;
  coarse.east[i] = 0.5f * (fine.east[j + 1] + fine.east[j + fine.w + 1]);
  coarse.north[i] = 0.5f * (fine.north[j + fine.w] + fine.north[j + fine.w + 1]);
  coarse.boundary[i] =
      0.5f * (fine.boundary[j] + fine.boundary[j + 1] + fine.boundary[j + fine.w] + fine.boundary[j + fine.w + 1]);
}

float Diagonal(Level l, int i) {
  return l.east[i] + l.north[i] + l.boundary[i] + (i % l.w ? l.east[i - 1] : 0) + (i >= l.w ? l.north[i - l.w] : 0);
}

float Neighbors(Level l, View p, int i) {
  float sum = 0;
  if (i % l.w + 1 < l.w)
    sum += l.east[i] * p[i + 1];
  if (i % l.w)
    sum += l.east[i - 1] * p[i - 1];
  if (i + l.w < l.w * l.h)
    sum += l.north[i] * p[i + l.w];
  if (i >= l.w)
    sum += l.north[i - l.w] * p[i - l.w];
  return sum;
}

void Smooth(int tid, Level l, int color) {
  int i = tid;
  if (i >= l.w * l.h || ((i % l.w + i / l.w) & 1) != color)
    return;
  float d = Diagonal(l, i);
  l.p[i] = d > 0 ? (l.b[i] + Neighbors(l, l.p, i)) / d : 0;
}

void Residual(int tid, Level l) {
  int i = tid;
  if (i < l.w * l.h)
    l.r[i] = l.b[i] - (Diagonal(l, i) * l.p[i] - Neighbors(l, l.p, i));
}

// The last grid is 4 x 2. Solve it directly instead of launching many tiny smoothing kernels.
void CoarseSolve(int tid, Level l) {
  float matrix[8][9];
  for (int a = 0; a < 8; ++a)
    for (int b = 0; b < 9; ++b)
      matrix[a][b] = 0;
  for (int i = 0; i < 8; ++i) {
    float d = Diagonal(l, i);
    matrix[i][i] = d > 0 ? d : 1;
    matrix[i][8] = d > 0 ? l.b[i] : 0;
    if (i % l.w + 1 < l.w)
      matrix[i][i + 1] = -l.east[i];
    if (i % l.w)
      matrix[i][i - 1] = -l.east[i - 1];
    if (i + l.w < 8)
      matrix[i][i + l.w] = -l.north[i];
    if (i >= l.w)
      matrix[i][i - l.w] = -l.north[i - l.w];
  }
  for (int k = 0; k < 8; ++k) {
    for (int i = k + 1; i < 8; ++i) {
      float scale = matrix[i][k] / matrix[k][k];
      for (int j = k + 1; j <= 8; ++j)
        matrix[i][j] -= scale * matrix[k][j];
    }
  }
  for (int i = 7; i >= 0; --i) {
    float value = matrix[i][8];
    for (int j = i + 1; j < 8; ++j)
      value -= matrix[i][j] * l.p[j];
    l.p[i] = value / matrix[i][i];
  }
}

void Restrict(int tid, Level f, Level c) {
  int i = tid;
  if (i >= c.w * c.h)
    return;
  int j = (i / c.w * 2) * f.w + (i % c.w * 2);
  c.b[i] = f.r[j] + f.r[j + 1] + f.r[j + f.w] + f.r[j + f.w + 1];
  c.p[i] = 0;
}

float CoarsePressure(Level c, int x, int y) {
  float sign = x >= c.w ? -1.0f : 1.0f;
  return sign * c.p[min(max(y, 0), c.h - 1) * c.w + min(max(x, 0), c.w - 1)];
}

void Prolong(int tid, Level c, Level f) {
  int i = tid;
  if (i >= f.w * f.h || Diagonal(f, i) == 0)
    return;
  float x = (i % f.w + 0.5f) * 0.5f - 0.5f, y = (i / f.w + 0.5f) * 0.5f - 0.5f;
  int ix = int(floor(x)), iy = int(floor(y));
  float a = x - ix, b = y - iy;
  f.p[i] += (1 - b) * ((1 - a) * CoarsePressure(c, ix, iy) + a * CoarsePressure(c, ix + 1, iy)) +
            b * ((1 - a) * CoarsePressure(c, ix, iy + 1) + a * CoarsePressure(c, ix + 1, iy + 1));
}

float Sample(View f, int w, int h, float x, float y) {
  x = min(max(x, 0), float(w - 1));
  y = min(max(y, 0), float(h - 1));
  int ix = int(x), iy = int(y), jx = min(ix + 1, w - 1), jy = min(iy + 1, h - 1);
  float a = x - ix, b = y - iy;
  return (1 - b) * ((1 - a) * f[iy * w + ix] + a * f[iy * w + jx]) +
         b * ((1 - a) * f[jy * w + ix] + a * f[jy * w + jx]);
}

float2 Velocity(View u, View v, int w, int h, float x, float y) {
  return float2(Sample(u, w + 1, h, x, y - 0.5f), Sample(v, w, h + 1, x - 0.5f, y));
}

void InitializeVelocity(int tid, View u, View v, int w, int h, float speed) {
  int i = tid;
  if (i < (w + 1) * h)
    u[i] = speed;
  if (i < w * (h + 1))
    v[i] = 0;
}

void InitializeDye(int tid, View dye, View mask, int w, int h) {
  int i = tid;
  if (i >= w * h)
    return;
  float stripe = sin((i / w + 0.5f) * (6.0f / w) * 3.14159265f / 0.12f);
  dye[i] = mask[i] ? 0.75f * exp(-50 * stripe * stripe) : 0;
}

void Boundaries(int tid, View u, View v, Level l, float speed) {
  int w = l.w, h = l.h;
  int i = tid;
  if (i < (w + 1) * h) {
    int x = i % (w + 1), y = i / (w + 1);
    if (x == 0)
      u[i] = speed;
    else if (UWeight(l, x, y) == 0)
      u[i] = 0;
  }
  if (i < w * (h + 1)) {
    int x = i % w, y = i / w;
    if (VWeight(l, x, y) == 0)
      v[i] = 0;
  }
}

void Advect(int tid, View u, View v, View next_u, View next_v, int w, int h, float dt) {
  int i = tid;
  float dx = 6.0f / w, tau = dt / dx;
  for (int component = 0; component < 2; ++component) {
    int fw = component ? w : w + 1, fh = component ? h + 1 : h;
    if (i >= fw * fh)
      continue;
    int x = i % fw, y = i / fw;
    float px = x + (component ? 0.5f : 0), py = y + (component ? 0 : 0.5f);
    float2 vel = Velocity(u, v, w, h, px, py);
    float2 mid = Velocity(u, v, w, h, px - 0.5f * tau * vel.x, py - 0.5f * tau * vel.y);
    View f = component ? v : u;
    float value = Sample(f, fw, fh, x - tau * mid.x, y - tau * mid.y);
    View destination = component ? next_v : next_u;
    destination[i] = value;
  }
}

void Diffuse(int tid, View u, View v, View next_u, View next_v, Level l, float factor) {
  int w = l.w, h = l.h;
  int i = tid;
  for (int component = 0; component < 2; ++component) {
    int fw = component ? w : w + 1, fh = component ? h + 1 : h;
    if (i >= fw * fh)
      continue;
    int x = i % fw, y = i / fw;
    View f = component ? v : u;
    float lap = f[y * fw + max(x - 1, 0)] + f[y * fw + min(x + 1, fw - 1)] + f[max(y - 1, 0) * fw + x] +
                f[min(y + 1, fh - 1) * fw + x] - 4 * f[i];
    float aperture = component ? VWeight(l, x, y) : UWeight(l, x, y);
    // Approximate no-slip drag on the blocked portion, implicit to avoid a tiny-aperture timestep.
    float drag = 2 * factor * (1 - aperture) / max(aperture, 0.1f);
    View destination = component ? next_v : next_u;
    destination[i] = (f[i] + factor * lap) / (1 + drag);
  }
}

void RHS(int tid, Level l, View u, View v, View mask, float dt) {
  int i = tid;
  if (i >= l.w * l.h)
    return;
  int x = i % l.w, y = i / l.w;
  float flux = Flux(l, u, v, x, y);
  l.b[i] = mask[i] ? -flux * (6.0f / l.w) / dt : 0;
  l.p[i] = 0;
}

void Project(int tid, Level l, View u, View v, float dt) {
  int i = tid;
  float scale = dt / (6.0f / l.w);
  if (i < (l.w + 1) * l.h) {
    int x = i % (l.w + 1), y = i / (l.w + 1), j = y * l.w + x;
    if (x == l.w)
      u[i] += scale * 2 * l.p[j - 1];
    else if (x > 0 && l.east[j - 1] > 0)
      u[i] -= scale * (l.p[j] - l.p[j - 1]);
  }
  if (i < l.w * (l.h + 1)) {
    int y = i / l.w;
    if (y > 0 && y < l.h && l.north[i - l.w] > 0)
      v[i] -= scale * (l.p[i] - l.p[i - l.w]);
  }
}

void AdvectDye(int tid, View u, View v, View dye, View next, int w, int h, float dt) {
  int i = tid;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w;
  float dx = 6.0f / w;
  float2 vel = Velocity(u, v, w, h, x + 0.5f, y + 0.5f);
  next[i] = Sample(dye, w, h, x - dt / dx * vel.x, y - dt / dx * vel.y);
}

void CorrectDye(int tid,
                View u,
                View v,
                View dye,
                View forward,
                View next,
                View mask,
                int w,
                int h,
                float dt,
                float time) {
  int i = tid;
  if (i >= w * h)
    return;
  int x = i % w, y = i / w;
  float dx = 6.0f / w;
  float2 vel = Velocity(u, v, w, h, x + 0.5f, y + 0.5f);
  float value = forward[i] + 0.5f * (dye[i] - Sample(forward, w, h, x + dt / dx * vel.x, y + dt / dx * vel.y));
  int sx = max(0, min(w - 2, int(floor(x - dt / dx * vel.x))));
  int sy = max(0, min(h - 2, int(floor(y - dt / dx * vel.y))));
  float lo = 1, hi = 0;
  for (int yy = 0; yy < 2; ++yy)
    for (int xx = 0; xx < 2; ++xx) {
      float d = dye[(sy + yy) * w + sx + xx];
      lo = min(lo, d);
      hi = max(hi, d);
    }
  value = min(hi, max(lo, value)) * exp(-dt * 0.015f);
  if (x < 3) {
    float stripe = sin((y + 0.5f) * dx * 3.14159265f / 0.12f);
    value = exp(-50 * stripe * stripe) * (0.75f + 0.25f * sin(time * 5));
  }
  next[i] = mask[i] ? value : 0;
}

void Display(int tid, View4 output, View u, View v, Level l, View dye, View mask) {
  int i = tid;
  if (i >= l.w * l.h)
    return;
  float2 vel = Velocity(u, v, l.w, l.h, i % l.w + 0.5f, i / l.w + 0.5f);
  output[i] = float4(vel.x, vel.y, l.p[i], mask[i] ? dye[i] : -1);
}

void MeasureFields(int tid, Level l, View u, View v, View mask, float dt, View stats) {
  int i = tid;
  if (i >= l.w * l.h)
    return;
  for (int k = 0; k < 7; ++k)
    stats[i * 7 + k] = 0;
  int x = i % l.w, y = i / l.w;
  float dx = 6.0f / l.w;
  float div = Flux(l, u, v, x, y) / dx;
  float2 vel = Velocity(u, v, l.w, l.h, x + 0.5f, y + 0.5f);
  if (!isfinite(div) || !isfinite(l.p[i]) || !isfinite(vel.x) || !isfinite(vel.y))
    stats[i * 7 + 6] = 1;
  if (mask[i]) {
    float before = l.b[i] * dt / (dx * dx);
    stats[i * 7 + 0] = before * before;
    stats[i * 7 + 1] = div * div;
    stats[i * 7 + 2] = l.r[i] * l.r[i];
    stats[i * 7 + 3] = 1.0f;
    stats[i * 7 + 4] = Hypot(vel.x, vel.y);
  } else {
    stats[i * 7 + 5] = Hypot(vel.x, vel.y);
  }
}

void Manufactured(int tid, Level l, View exact, bool rhs) {
  int i = tid;
  if (i >= l.w * l.h)
    return;
  if (!rhs)
    exact[i] = cos(1.7f * (i % l.w + 0.5f) / l.w) * sin(3.1f * (i / l.w + 0.5f) / l.h);
  else {
    l.b[i] = Diagonal(l, i) * exact[i] - Neighbors(l, exact, i);
    l.p[i] = 0;
  }
}

static const int surface_segments = 512;

// Counterclockwise closed polygon, cosine spacing resolves the rounded nose.
float2 SurfacePoint(int index, float angle, int profile) {
  int k = index <= surface_segments / 2 ? index : surface_segments - index;
  float q = 0.5f * (1 - cos(k * (6.28318530718f / surface_segments)));
  float x = (q - 0.5f) * 1.25f;
  float y = (WingCamber(q, profile) + WingThickness(q, profile) * (index <= surface_segments / 2 ? -1 : 1)) * 1.25f;
  float a = angle * 0.01745329252f;
  return float2(x * cos(a) + y * sin(a), -x * sin(a) + y * cos(a));
}

float SurfacePressure(Level l, View mask, float x, float y) {
  int ix = int(floor(x)), iy = int(floor(y));
  float a = x - ix, b = y - iy, sum = 0, weight = 0;
  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      if (!Fluid(mask, ix + dx, iy + dy, l.w, l.h))
        continue;
      float w = (dx ? a : 1 - a) * (dy ? b : 1 - b);
      sum += w * l.p[(iy + dy) * l.w + ix + dx];
      weight += w;
    }
  return weight > 0 ? sum / weight : 0;
}

void IntegrateSmoothLoads(int tid,
                          Level l,
                          View u,
                          View v,
                          View mask,
                          float angle,
                          int profile,
                          float viscosity,
                          View sums,
                          int analytic) {
  int i = tid;
  if (i >= surface_segments)
    return;
  sums[i * 7 + 6] = 0;
  float2 a = SurfacePoint(i, angle, profile), b = SurfacePoint(i + 1, angle, profile);
  float rx = 0.5 * (a.x + b.x), ry = 0.5 * (a.y + b.y);
  float tx = b.x - a.x, ty = b.y - a.y, length = Hypot(tx, ty);
  float nx = ty / length, ny = -tx / length;
  float h = 6.0f / l.w;
  float px = float((rx + 2 + nx * h) / h), py = float((ry + 1.5 + ny * h) / h);
  float p = analytic == 1 ? 7 : analytic == 2 ? 0.7 * rx - 0.4 * ry : SurfacePressure(l, mask, px - 0.5f, py - 0.5f);
  float fx = -p * ty, fy = p * tx;
  float2 vel = Velocity(u, v, l.w, l.h, px, py);
  float normal = vel.x * nx + vel.y * ny;
  float sx = analytic ? 0 : viscosity * (vel.x - normal * nx) * length / h;
  float sy = analytic ? 0 : viscosity * (vel.y - normal * ny) * length / h;
  sums[i * 7 + 0] = fx;
  sums[i * 7 + 1] = fy;
  sums[i * 7 + 2] = rx * fy - ry * fx;
  sums[i * 7 + 3] = sx;
  sums[i * 7 + 4] = sy;
  sums[i * 7 + 5] = rx * sy - ry * sx;
}

void IntegrateLoads(int tid, Level l, View u, View v, View mask, float viscosity, View sums, int analytic = 0) {
  int i = tid;
  if (i >= l.w * l.h)
    return;
  for (int k = 0; k < 7; ++k)
    sums[i * 7 + k] = 0;
  if (!mask[i])
    return;
  int x = i % l.w, y = i / l.w;
  const int ox[4] = {1, -1, 0, 0}, oy[4] = {0, 0, 1, -1};
  const float h = float(6.0f) / l.w;
  float2 velocity = Velocity(u, v, l.w, l.h, x + 0.5f, y + 0.5f);
  float local[6] = {0, 0, 0, 0, 0, 0};
  bool boundary = false;
  for (int k = 0; k < 4; ++k) {
    int nx = x + ox[k], ny = y + oy[k];
    if (nx < 0 || nx >= l.w || ny < 0 || ny >= l.h || mask[ny * l.w + nx])
      continue;
    boundary = true;
    // d points from fluid to solid, opposite to the body's outward normal.
    float rx = (x + 0.5 + 0.5 * ox[k]) * h - 2.0;
    float ry = (y + 0.5 + 0.5 * oy[k]) * h - 1.5;
    float p = analytic == 1 ? 7.0 : analytic == 2 ? 0.7 * rx - 0.4 * ry : l.p[i];
    float fx = p * ox[k] * h, fy = p * oy[k] * h;
    local[0] += fx;
    local[1] += fy;
    local[2] += rx * fy - ry * fx;
    // Stationary no-slip wall: tangential shear from the adjacent cell center,
    // half a cell from the wall. Normal viscous traction is zero for a smooth
    // incompressible no-slip wall; stair-step corners remain an approximation.
    float normal = velocity.x * ox[k] + velocity.y * oy[k];
    float sx = analytic ? 0 : 2 * viscosity * (velocity.x - normal * ox[k]);
    float sy = analytic ? 0 : 2 * viscosity * (velocity.y - normal * oy[k]);
    local[3] += sx;
    local[4] += sy;
    local[5] += rx * sy - ry * sx;
  }
  if (boundary)
    for (int k = 0; k < 6; ++k)
      sums[i * 7 + k] = local[k];
}

Level L(uint4 a, uint4 b) {
  Level l;
  l.w = a.x;
  l.h = a.y;
  l.p = V(a.z);
  l.b = V(a.w);
  l.r = V(b.x);
  l.east = V(b.y);
  l.north = V(b.z);
  l.boundary = V(b.w);
  return l;
}

groupshared float reduction[1792];

[numthreads(256, 1, 1)] void CSMain(uint3 id
                                    : SV_DispatchThreadID, uint3 group
                                    : SV_GroupID, uint lane
                                    : SV_GroupIndex) {
  int i = id.x;
  Level l = L(settings.level0, settings.level1), c = L(settings.coarse0, settings.coarse1);
  View u = V(settings.fields0.x), v = V(settings.fields0.y), next_u = V(settings.fields0.z),
       next_v = V(settings.fields0.w);
  View mask = V(settings.fields1.x), dye = V(settings.fields1.y), next_dye = V(settings.fields1.z),
       corrected_dye = V(settings.fields1.w);
  float speed = settings.values0.x, angle = settings.values0.y, viscosity = settings.values0.z, dt = settings.values0.w,
        time = settings.values1.x;
  int profile = int(settings.values1.y), color = settings.controls.y, analytic = settings.controls.y;
  View stats = V(settings.scratch.x);
  int count = settings.controls.z;
  switch (settings.controls.x) {
    case 0:
      SmoothGeometry(i, l, angle, profile);
      break;
    case 1:
      ActiveCells(i, l, mask);
      break;
    case 2:
      Geometry(i, mask, l.w, l.h, angle, profile);
      break;
    case 3:
      FineOperator(i, l, mask);
      break;
    case 4:
      CoarsenOperator(i, l, c);
      break;
    case 6:
      Residual(i, l);
      break;
    case 8:
      Restrict(i, l, c);
      break;
    case 9:
      Prolong(i, c, l);
      break;
    case 10:
      InitializeVelocity(i, u, v, l.w, l.h, speed);
      break;
    case 11:
      InitializeDye(i, dye, mask, l.w, l.h);
      break;
    case 12:
      Boundaries(i, u, v, l, speed);
      break;
    case 13:
      Advect(i, u, v, next_u, next_v, l.w, l.h, dt);
      break;
    case 14:
      Diffuse(i, u, v, next_u, next_v, l, viscosity * dt / (6.0 / l.w) / (6.0 / l.w));
      break;
    case 15:
      RHS(i, l, u, v, mask, dt);
      break;
    case 16:
      Project(i, l, u, v, dt);
      break;
    case 17:
      AdvectDye(i, u, v, dye, next_dye, l.w, l.h, dt);
      break;
    case 18:
      CorrectDye(i, u, v, dye, next_dye, corrected_dye, mask, l.w, l.h, dt, time);
      break;
    case 19: {
      View4 output;
      Display(i, output, u, v, l, dye, mask);
      break;
    }
    case 20:
      MeasureFields(i, l, u, v, mask, dt, stats);
      break;
    case 21:
      Manufactured(i, l, next_dye, color != 0);
      break;
    case 22:
      IntegrateSmoothLoads(i, l, u, v, mask, angle, profile, viscosity, stats, analytic);
      break;
    case 23:
      IntegrateLoads(i, l, u, v, mask, viscosity, stats, analytic);
      break;
    case 24:
      for (int k = 0; k < 7; ++k)
        reduction[k * 256 + lane] = i < count ? stats[i * 7 + k] : 0;
      GroupMemoryBarrierWithGroupSync();
      for (int stride = 128; stride > 0; stride /= 2) {
        if (lane < stride)
          for (int k = 0; k < 7; ++k) {
            float other = reduction[k * 256 + lane + stride];
            reduction[k * 256 + lane] = settings.controls.w != 0 && k >= 4 ? max(reduction[k * 256 + lane], other)
                                                                           : reduction[k * 256 + lane] + other;
          }
        GroupMemoryBarrierWithGroupSync();
      }
      if (lane == 0)
        for (int k = 0; k < 7; ++k)
          heap.Store<float>((settings.scratch.y + group.x * 7 + k) * 4, reduction[k * 256]);
      break;
  }
}

    // One lane per active red/black cell. This entry excludes the direct solver's
    // private matrix and the diagnostic reduction's shared storage from the hot kernel.
    [numthreads(256, 1, 1)] void SmoothMain(uint3 id
                                            : SV_DispatchThreadID) {
  Level l = L(settings.level0, settings.level1);
  int row = int(id.x) / (l.w / 2);
  int i = int(id.x) * 2 + ((row + settings.controls.y) & 1);
  Smooth(i, l, settings.controls.y);
}

// The final three grids (16x8, 8x4, 4x2) fit entirely in one thread group.
// Keep the same V-cycle arithmetic and ordering, replacing dispatch boundaries
// with device-memory group barriers. No other group accesses these grids here.
void GroupSmooth(uint lane, Level l) {
  for (int sweep = 0; sweep < 3; ++sweep) {
    Smooth(lane, l, 0);
    DeviceMemoryBarrierWithGroupSync();
    Smooth(lane, l, 1);
    DeviceMemoryBarrierWithGroupSync();
  }
}

[numthreads(128, 1, 1)] void CoarseCycleMain(uint lane
                                             : SV_GroupIndex) {
  Level fine = L(settings.level0, settings.level1);
  Level middle = L(settings.coarse0, settings.coarse1);
  // Each level allocates p,b,r,east,north,boundary consecutively.
  uint base = middle.boundary.offset + 32;
  Level last = L(uint4(4, 2, base, base + 8), uint4(base + 16, base + 24, base + 32, base + 40));
  GroupSmooth(lane, fine);
  Residual(lane, fine);
  DeviceMemoryBarrierWithGroupSync();
  Restrict(lane, fine, middle);
  DeviceMemoryBarrierWithGroupSync();
  GroupSmooth(lane, middle);
  Residual(lane, middle);
  DeviceMemoryBarrierWithGroupSync();
  Restrict(lane, middle, last);
  DeviceMemoryBarrierWithGroupSync();
  if (lane == 0)
    CoarseSolve(0, last);
  DeviceMemoryBarrierWithGroupSync();
  Prolong(lane, last, middle);
  DeviceMemoryBarrierWithGroupSync();
  GroupSmooth(lane, middle);
  Prolong(lane, middle, fine);
  DeviceMemoryBarrierWithGroupSync();
  GroupSmooth(lane, fine);
}
