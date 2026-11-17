// Included inside Simulation::Impl in native builds.
bool ModelPoses(const std::vector<double> &state,
                const aero::Transform &body,
                std::vector<aero::PartPose> &world,
                std::string &message) const {
  using namespace aero::mesh_math;
  for (double value : body.rotation)
    if (!std::isfinite(value)) {
      message = "Nonfinite body rotation";
      return false;
    }
  if (!std::isfinite(body.translation.x) || !std::isfinite(body.translation.y) || !std::isfinite(body.translation.z)) {
    message = "Nonfinite body translation";
    return false;
  }
  aero::Vector columns[3];
  for (int i = 0; i < 3; ++i)
    columns[i] = {body.rotation[i], body.rotation[i + 3], body.rotation[i + 6]};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      if (std::abs(Dot(columns[i], columns[j]) - (i == j ? 1.0 : 0.0)) > 1e-8) {
        message = "Body pose must be rigid";
        return false;
      }
  if (Dot(Cross(columns[0], columns[1]), columns[2]) < 0) {
    message = "Body pose must preserve handedness";
    return false;
  }
  if (!options.model->Evaluate(state, world, message))
    return false;
  for (size_t i = 0; i < world.size(); ++i) {
    auto local = world[i].local_to_body;
    auto &t = world[i].local_to_body;
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 3; ++col) {
        t.rotation[3 * row + col] = 0;
        for (int k = 0; k < 3; ++k)
          t.rotation[3 * row + col] += body.rotation[3 * row + k] * local.rotation[3 * k + col];
      }
    t.translation = body.Point(local.translation);
    auto bounds = model_geometry->Meshes()[i]->Nodes()[0];
    for (int mask = 0; mask < 8; ++mask) {
      auto p = t.Point({mask & 1 ? bounds.upper.x : bounds.lower.x, mask & 2 ? bounds.upper.y : bounds.lower.y,
                        mask & 4 ? bounds.upper.z : bounds.lower.z});
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.x <= -6 + 2 * grid.h ||
          p.x >= 10 - 2 * grid.h || p.y <= -4 + 2 * grid.h || p.y >= 4 - 2 * grid.h || p.z <= -6 + 2 * grid.h ||
          p.z >= 6 - 2 * grid.h) {
        message = "Model bounds must stay at least two cells inside the flow domain";
        return false;
      }
    }
  }
  return true;
}

bool InitializeModel() {
  model_geometry = aero::ModelGeometry::Load(*options.model, error);
  if (!model_geometry) {
    std::cerr << error << '\n';
    return false;
  }
  requested_model_state =
      options.initial_model_state.empty() ? options.model->DefaultState() : options.initial_model_state;
  requested_body_pose = options.initial_body_pose;
  if (!ModelPoses(requested_model_state, requested_body_pose, requested_poses, error)) {
    std::cerr << error << '\n';
    return false;
  }
  std::unordered_map<const aero::MeshGeometry *, aero::MeshView> cache;
  for (const auto &mesh : model_geometry->Meshes()) {
    auto found = cache.find(mesh.get());
    if (found == cache.end()) {
      aero::MeshTriangle *triangles{};
      aero::MeshNode *nodes{};
      if (!Allocate(triangles, mesh->Triangles().size(), false) || !Allocate(nodes, mesh->Nodes().size(), false))
        return false;
      if (!Check(cudaMemcpyAsync(triangles, mesh->Triangles().data(), mesh->Triangles().size() * sizeof(*triangles),
                                 cudaMemcpyHostToDevice, stream)) ||
          !Check(cudaMemcpyAsync(nodes, mesh->Nodes().data(), mesh->Nodes().size() * sizeof(*nodes),
                                 cudaMemcpyHostToDevice, stream)))
        return false;
      found = cache.emplace(mesh.get(), aero::MeshView{triangles, nodes, int(mesh->Nodes().size())}).first;
    }
    device_meshes.push_back(found->second);
  }
  return Allocate(device_instances, device_meshes.size(), false);
}

bool UploadModel(const std::vector<aero::PartPose> &poses, std::shared_ptr<const std::vector<Vertex>> &display) {
  std::vector<aero::MeshInstanceView> instances(poses.size());
  std::vector<Vertex> vertices;
  aero::Vector lower{1e300, 1e300, 1e300}, upper{-1e300, -1e300, -1e300};
  for (size_t i = 0; i < poses.size(); ++i) {
    const auto &t = poses[i].local_to_body;
    instances[i].mesh = device_meshes[i];
    std::copy(t.rotation.begin(), t.rotation.end(), instances[i].rotation);
    instances[i].translation = t.translation;
    const auto &mesh = *model_geometry->Meshes()[i];
    const auto bounds = mesh.Nodes()[0];
    for (int mask = 0; mask < 8; ++mask) {
      auto p = t.Point({mask & 1 ? bounds.upper.x : bounds.lower.x, mask & 2 ? bounds.upper.y : bounds.lower.y,
                        mask & 4 ? bounds.upper.z : bounds.lower.z});
      lower = {std::min(lower.x, p.x), std::min(lower.y, p.y), std::min(lower.z, p.z)};
      upper = {std::max(upper.x, p.x), std::max(upper.y, p.y), std::max(upper.z, p.z)};
    }
    if (options.display)
      for (const auto &triangle : mesh.Triangles()) {
        auto n = t.Rotate(triangle.normal);
        for (auto local : triangle.vertex) {
          auto p = t.Point(local);
          vertices.push_back(
              {{float(p.x), float(p.y), float(p.z)}, {float(n.x), float(n.y), float(n.z)}, {.65f, .7f, .75f}});
        }
      }
  }
  // Synchronize before local host storage is released and before rebuilding masks.
  if (!Check(cudaMemcpyAsync(device_instances, instances.data(), instances.size() * sizeof(instances[0]),
                             cudaMemcpyHostToDevice, stream)) ||
      !Check(cudaStreamSynchronize(stream)))
    return false;
  model_view = {device_instances, int(instances.size()), lower, upper};
  display = std::make_shared<const std::vector<Vertex>>(std::move(vertices));
  return true;
}
