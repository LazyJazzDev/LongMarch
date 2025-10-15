#include "level_set_grid.h"

struct Sphere {
  float radius{1.0f};
  float cx{0.0f};
  float cy{0.0f};
  float cz{0.0f};
  __device__ __host__ float operator()(float x, float y, float z) const {
    x -= cx;
    y -= cy;
    z -= cz;
    return sqrtf(x * x + y * y + z * z) - radius;
  }
};

template <class SDFOp>
struct Reverse {
  SDFOp SDF;
  Reverse(const SDFOp &sdf) : SDF(sdf) {
  }
  __device__ __host__ float operator()(float x, float y, float z) const {
    return -SDF(x, y, z);
  }
};

struct Cube {
  float side_x{1.0f};
  float side_y{1.0f};
  float side_z{1.0f};
  float cx{0.0f};
  float cy{0.0f};
  float cz{0.0f};
  __device__ __host__ float operator()(float x, float y, float z) const {
    x -= cx;
    y -= cy;
    z -= cz;
    float dx = fabsf(x) - side_x * 0.5f;
    float dy = fabsf(y) - side_y * 0.5f;
    float dz = fabsf(z) - side_z * 0.5f;
    if (dx < 0 && dy < 0 && dz < 0) {
      return fmaxf(fmaxf(dx, dy), dz);
    }
    dx = fmaxf(dx, 0.0f);
    dy = fmaxf(dy, 0.0f);
    dz = fmaxf(dz, 0.0f);
    return sqrtf(dx * dx + dy * dy + dz * dz);
  }
};

template <class SDFOp>
struct Extend {
  SDFOp SDF;
  float radius;
  Extend(const SDFOp &sdf, float radius = 0.0f) : SDF(sdf), radius(radius) {
  }
  __device__ __host__ float operator()(float x, float y, float z) const {
    return SDF(x, y, z) - radius;
  }
};

template <class Op1, class Op2>
struct Union {
  Op1 op1;
  Op2 op2;
  Union(const Op1 &o1 = Op1(), const Op2 &o2 = Op2()) : op1(o1), op2(o2) {
  }
  __device__ __host__ float operator()(float x, float y, float z) const {
    float d1 = op1(x, y, z);
    float d2 = op2(x, y, z);
    return fminf(d1, d2);
  }
};

template <class Op1, class Op2>
struct Intersection {
  Op1 op1;
  Op2 op2;
  Intersection(const Op1 &o1 = Op1(), const Op2 &o2 = Op2()) : op1(o1), op2(o2) {
  }
  __device__ __host__ float operator()(float x, float y, float z) const {
    float d1 = op1(x, y, z);
    float d2 = op2(x, y, z);
    return fmaxf(d1, d2);
  }
};

void GenerateSocket() {
  LevelSetGrid grid(400, 400, 400, 0.00025f, -0.05f, -0.05f, -0.05f);

  printf("Grid size: %dx%dx%d\n", grid.SizeX(), grid.SizeY(), grid.SizeZ());

  grid.SetLevelSet(
      Intersection{Intersection{Extend{Cube{0.0395f, 0.05f, 0.018f}, 0.005f}, Cube{0.0495f, 0.05f, 0.028f}},
                   Reverse{Cube{0.012f, 0.012f, 0.005f, 0.0f, 0.019f, 0.0f}}});

  auto mesh = grid.MarchingCubes();
  mesh.SaveToFile("socket.obj");
}

void GeneratePlug() {
  LevelSetGrid grid(400, 400, 400, 0.00025f, -0.05f, -0.05f, -0.05f);

  printf("Grid size: %dx%dx%d\n", grid.SizeX(), grid.SizeY(), grid.SizeZ());

  grid.SetLevelSet(Union{Cube{0.021f, 0.038f, 0.009f}, Cube{0.0118f, 0.0118f, 0.0048f, 0.0f, -0.0239f, 0.0f}});

  auto mesh = grid.MarchingCubes();
  mesh.SaveToFile("plug.obj");
}

int main() {
  GenerateSocket();
  GeneratePlug();
}
