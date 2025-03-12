import long_march
import long_march.grassland
import long_march.grassland.graphics as graphics

core_settings = graphics.CoreSettings()
core_settings.frames_in_flight = 1
core = graphics.create_core(settings = core_settings)
print(core)

# print current working directory
