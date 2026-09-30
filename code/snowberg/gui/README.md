# Snowberg GUI

Snowberg GUI is a small retained renderer with a declarative frame API on top of
`grassland::graphics`. It draws into an existing scene image, so demos can keep
their own game or 3D renderer and compose controls at the end of the frame.

```cpp
snowberg::gui::Context ui(core, window, snowberg::gui::DefaultFont());

// Once per frame, after polling window events:
ui.BeginFrame();
ui.BeginPanel("controls", 24, 24, 280, "Controls");
ui.Checkbox("Animate", &animate);
ui.Slider("Intensity", &intensity, 0.0f, 1.0f);
if (ui.Button("Reset")) ResetScene();
ui.EndPanel();

// After the scene has been rendered into scene_image:
auto *display_image = ui.EndFrame(commands, scene_image);
commands->CmdPresent(window, display_image);
```

Coordinates use window logical pixels. Panels lay out their contents vertically;
`BeginRow` and `EndRow` distribute a fixed number of controls across a row.
Control labels are stable IDs within each panel, so labels should be unique in
that panel. `Custom` provides a `Canvas` for code drawn shapes, labels and
particle fields while sharing the same layout and pointer hit testing.

`Theme` exposes the common panel, control, accent and text colors, a dark mode
flag and backdrop blur. A theme can be set through `ui.Style()` before drawing.
The default is a light frosted appearance; dark scenes can use a translucent
dark palette. The blur and glass composite are GPU compute passes. Rendering
allocates images at the scene resolution and recreates them after a resize.

For 3D controls, create a `WorldPanel`, set its local to world transform and
ray test the pointer against it. Feed the nearest hit to `BeginFrame`, build
controls through `Controls()`, then call `Render` with the scene color and depth
images. The panel is rendered into an offscreen image and drawn as a depth
tested world quad. See `demo/graphics_hello/modules/cube` for a mixed 2D and
3D example.

Current scope: mouse driven panels, rows, text, buttons, checkboxes, choices,
sliders, plots and custom drawing. Keyboard focus, text fields, clipping and
scrollable layout are future extensions. The API deliberately keeps scene
state, simulation and file dialogs in the application.
