# Submit with: meshy-envelope submit demo/wind3d/task.star
# One complete node gives all eight GPUs. Both compilation and simulation run there.
job(
    name = "wind3d-8gpu",
    resource = Resource(nodes = 1, cpus = 16, memory_gb = 128),
    environment = Environment(no_environment = NoEnvironment()),
    entry = Entry(unverified_binary_contact_infra_team_for_support = UnverifiedBinary(
        path = "/usr/bin/bash",
        args = [
            "${MESHY_ENVELOPE_GIT_CHECKOUT_DIR}/demo/wind3d/task.sh",
            "--no-export", "--gpus", "8", "--grid", "384", "--steps", "80", "--warmup", "10",
        ],
    )),
    schedule = Schedule(time_limit = Duration(minutes = 30)),
)
