job(
    name = "wind3d-web",
    resource = Resource(nodes = 1, cpus = 16, memory_gb = 128),
    environment = Environment(no_environment = NoEnvironment()),
    entry = Entry(unverified_binary_contact_infra_team_for_support = UnverifiedBinary(
        path = "/usr/bin/bash",
        args = ["${MESHY_ENVELOPE_GIT_CHECKOUT_DIR}/demo/wind3d/web.sh",
                "${MESHY_ENVELOPE_GIT_CHECKOUT_DIR}/out/wind3d-web/live", "384", "8"],
    )),
    schedule = Schedule(time_limit = Duration(hours = 8)),
)
