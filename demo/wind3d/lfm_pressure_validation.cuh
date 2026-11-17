#pragma once

// Included in namespace wind3d after the private MAC kernels.
// Potential flow over a sphere has Cp = 1 - 9/4 sin(theta)^2 and zero drag:
// https://farside.ph.utexas.edu/teaching/336L/Fluidhtml/node102.html
// Validate the actual pressure predictor, aperture divergence, solver and
// pressure load integration against this curved-wall analytic solution.
namespace pressure_validation {
constexpr float radius = 0.75f, incoming = 12;

__host__ __device__ Vec3 Potential(Vec3 p) {
  float r = sqrtf(Dot(p, p));
  if (r < radius) {
    p = Scale(p, radius / fmaxf(r, 1e-6f));
    r = radius;
  }
  float q = radius * radius * radius / (r * r * r);
  return {incoming * (1 + 0.5f * q - 1.5f * q * p.x * p.x / (r * r)), -incoming * 1.5f * q * p.x * p.y / (r * r),
          -incoming * 1.5f * q * p.x * p.z / (r * r)};
}

__host__ __device__ float ExactPressure(Vec3 p) {
  Vec3 v = Potential(p);
  return 0.5f * density * (incoming * incoming - Dot(v, v));
}

__global__ void SphereSetup(Grid g,
                            unsigned char *solid,
                            unsigned char *a0,
                            unsigned char *a1,
                            unsigned char *a2,
                            unsigned char *b0,
                            unsigned char *b1,
                            unsigned char *b2,
                            float *u,
                            float *v,
                            float *w) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int nf = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
  if (n >= nf)
    return;
  int x = n % (g.nx + 1), y = n / (g.nx + 1) % (g.ny + 1), z = n / ((g.nx + 1) * (g.ny + 1));
  unsigned char *a[] = {a0, a1, a2}, *b[] = {b0, b1, b2};
  float *f[] = {u, v, w};
  for (int axis = 0; axis < 3; ++axis) {
    Vec3 p{(x + (axis == 0 ? 0 : .5f)) * g.h - 6, (y + (axis == 1 ? 0 : .5f)) * g.h - 4,
           (z + (axis == 2 ? 0 : .5f)) * g.h - 6};
    int count = 0;
    if (Length(p) > radius + 2 * g.h)
      count = 36;
    else
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
          float q = ((i + .5f) / 6 - .5f) * g.h, r = ((j + .5f) / 6 - .5f) * g.h;
          Vec3 d{axis == 0 ? 0 : q, axis == 1 ? 0 : (axis == 0 ? q : r), axis == 2 ? 0 : r};
          count += Length(Add(p, d)) > radius;
        }
    a[axis][n] = count < 5 ? 0 : (unsigned char)rintf(255.f * count / 36);
    b[axis][n] = a[axis][n] ? 0 : 255;
    Vec3 vel = Potential(p);
    f[axis][n] = axis == 0 ? vel.x : axis == 1 ? vel.y : vel.z;
  }
}

int SphereCheck(int nx) {
  Grid g{nx, nx / 2, nx * 3 / 4, 16.f / nx};
  g.cubic = true;
  int nc = g.nx * g.ny * g.nz, nf = (g.nx + 1) * (g.ny + 1) * (g.nz + 1), fb = (nf + 255) / 256, cb = (nc + 255) / 256;
  float *f = nullptr;
  unsigned char *m = nullptr;
  if (cudaMalloc(&f, (6 * nf + 3 * nc) * sizeof(float)) != cudaSuccess)
    return 2;
  std::unique_ptr<float, decltype(&cudaFree)> fields_owner(f, &cudaFree);
  if (cudaMalloc(&m, 6 * nf + nc) != cudaSuccess)
    return 2;
  std::unique_ptr<unsigned char, decltype(&cudaFree)> masks_owner(m, &cudaFree);
  float *u = f, *v = f + nf, *w = f + 2 * nf, *pu = f + 3 * nf, *pv = f + 4 * nf, *pw = f + 5 * nf, *div = f + 6 * nf,
        *initial = div + nc, *pressure = initial + nc;
  unsigned char *au = m, *av = m + nf, *aw = m + 2 * nf, *bu = m + 3 * nf, *bv = m + 4 * nf, *bw = m + 5 * nf,
                *solid = m + 6 * nf;
  g.aperture[0] = au;
  g.aperture[1] = av;
  g.aperture[2] = aw;
  SphereSetup<<<fb, 256>>>(g, solid, au, av, aw, bu, bv, bw, u, v, w);
  CellMask<<<cb, 256>>>(g, solid, au, av, aw);
  if (cudaDeviceSynchronize() != cudaSuccess)
    return 3;
  cudaStream_t stream{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    return 3;
  auto destroy_stream = [](void *p) { cudaStreamDestroy(static_cast<cudaStream_t>(p)); };
  std::unique_ptr<void, decltype(destroy_stream)> stream_owner(stream, destroy_stream);
  pressure::Solver solver;
  if (solver.Initialize(g.nx, g.ny, g.nz, stream) != cudaSuccess || solver.Build(solid, au, av, aw) != cudaSuccess)
    return 3;
  // Apply the same far-field boundary and two-layer wall extension as LFM.
  // Subtract the initial discrete divergence to isolate pressure recovery.
  Boundary<<<fb, 256>>>(g, bu, bv, bw, u, v, w, incoming, false);
  for (int layer = 1; layer <= 2; ++layer) {
    Extrapolate<<<fb, 256>>>(g, bu, u, layer);
    Extrapolate<<<fb, 256>>>(g, bv, v, layer);
    Extrapolate<<<fb, 256>>>(g, bw, w, layer);
  }
  Divergence<<<cb, 256>>>(g, solid, au, av, aw, u, v, w, initial);
  std::vector<float> p(nc);
  std::vector<unsigned char> apertures(3 * nf);
  if (cudaMemcpy(apertures.data(), m, 3 * nf, cudaMemcpyDeviceToHost) != cudaSuccess)
    return 3;
  float dt = .1f * g.h / incoming;
  bool passed = true;
  {
    float *dst[] = {pu, pv, pw};
    for (int axis = 0; axis < 3; ++axis)
      LfmMomentumPredictor<<<fb, 256>>>(g, LfmVelocity{{u, v, w}}, dst[axis], axis, dt);
    // Boundary is applied identically to the production predictor.
    Boundary<<<fb, 256>>>(g, bu, bv, bw, pu, pv, pw, incoming, false);
    Divergence<<<cb, 256>>>(g, solid, au, av, aw, pu, pv, pw, div);
    LfmAccelerationDivergence<<<cb, 256>>>(g, div, initial);
    if (cudaMemset(pressure, 0, nc * sizeof(float)) != cudaSuccess)
      return 3;
    int cycles;
    float residual;
    if (cudaDeviceSynchronize() != cudaSuccess)
      return 3;
    auto status = solver.Solve(div, pressure, density * g.h * g.h / dt, 1e-6f, 256, cycles, residual);
    if (status != cudaSuccess) {
      std::cerr << solver.Diagnostic();
      return 4;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaMemcpy(p.data(), pressure, nc * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
      return 3;
    double error = 0, area = 0, drag = 0, lift = 0, exact_drag = 0;
    for (int z = 0; z < g.nz; ++z)
      for (int y = 0; y < g.ny; ++y)
        for (int x = 0; x < g.nx; ++x) {
          Vec3 a{};
          a.x = (int(apertures[g.F(x, y, z)]) - int(apertures[g.F(x + 1, y, z)])) * g.h * g.h / 255;
          a.y = (int(apertures[nf + g.F(x, y, z)]) - int(apertures[nf + g.F(x, y + 1, z)])) * g.h * g.h / 255;
          a.z = (int(apertures[2 * nf + g.F(x, y, z)]) - int(apertures[2 * nf + g.F(x, y, z + 1)])) * g.h * g.h / 255;
          float weight = Length(a);
          if (weight == 0)
            continue;
          Vec3 pos{(x + .5f) * g.h - 6, (y + .5f) * g.h - 4, (z + .5f) * g.h - 6};
          float exact = ExactPressure(pos), value = p[g.C(x, y, z)];
          error += weight * (value - exact) * (value - exact);
          area += weight;
          drag += a.x * value;
          lift += a.y * value;
          exact_drag += a.x * exact;
        }
    double cp_error = sqrt(error / area) / (0.5 * density * incoming * incoming);
    double limit = nx == 64 ? .17 : nx == 128 ? .10 : .065;
    passed &= std::isfinite(cp_error) && cp_error < limit && residual <= 1e-6f && std::abs(drag) < .05 &&
              std::abs(lift) < .05 && std::abs(exact_drag) < .001;
    std::cout << "sphere_pressure grid=" << nx << " pressure=" << residual << " cycles=" << cycles
              << " Cp_rmse=" << sqrt(error / area) / (0.5 * density * incoming * incoming) << " drag=" << drag
              << " lift=" << lift << " exact_drag=" << exact_drag << "\n";
  }
  return passed ? 0 : 5;
}
}  // namespace pressure_validation

bool ValidatePressureRecovery(int cuda_device) {
  if (cudaSetDevice(cuda_device) != cudaSuccess)
    return false;
  for (int nx : {64, 128, 256})
    if (pressure_validation::SphereCheck(nx) != 0)
      return false;
  std::cout << "lfm_pressure_recovery_validation=PASS\n";
  return true;
}
