import sys

import graphics_hello as gh
import graphics_hello_rt_multi_shader_group as rt_multi_shader_group

# Same RT scene as rt_multi_shader_group; shaders come from the assets submodule at run time.
SHADER_DIRECTORY = gh.REPO_ROOT / "assets" / "shaders" / "raytracing"


class Module(rt_multi_shader_group.Module):
    title = " Graphics External Shader"

    def create_shaders(self):
        if not SHADER_DIRECTORY.is_dir():
            raise RuntimeError(f"Shader directory not found: {SHADER_DIRECTORY}; initialize the assets submodule")
        directory = str(SHADER_DIRECTORY)
        for path in sorted(SHADER_DIRECTORY.iterdir()):
            print(path.name)
        create = self.core.create_shader_from_directory
        self.raygen_shader = create(directory, "raygen.slang", "Main", "lib_6_3")
        self.miss_shader = create(directory, "miss.slang", "Main", "lib_6_3")
        self.closest_hit_shader = create(directory, "closest_hit.slang", "Main", "lib_6_3")
        self.sphere_closest_hit_shader = create(directory, "sphere_chit.slang", "Main", "lib_6_3")
        self.sphere_intersection_shader = create(directory, "sphere_int.slang", "Main", "lib_6_3")
        self.callable_shader = create(directory, "callable.slang", "Main", "lib_6_3")


if __name__ == "__main__":
    sys.exit(gh.main(module="external_shader"))
