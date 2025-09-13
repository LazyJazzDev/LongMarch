from long_march import graphics

core_settings = graphics.CoreSettings()
print(core_settings)

core = graphics.Core()
print(core)
core.init_auto()
print(core)
window=core.create_window(800, 600, "Hello Triangle", resizable=True)
print(window)
window.register_drop_event(lambda paths: print(f"Files dropped: {paths}"))
while not window.should_close():
    window.poll_events()
