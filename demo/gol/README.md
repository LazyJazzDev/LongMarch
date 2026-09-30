# Game of Life

Conway's Game of Life, ported from the `gol` GUI of
[Yao-class-cpp-studio/Assignment-2](https://github.com/Yao-class-cpp-studio/Assignment-2)
to `grassland::graphics`. The cell renderer and simulation remain on graphics;
the controls now use the shared Snowberg GUI.

```sh
cmake --build cmake-build-release --target demo_gol
cmake-build-release/demo/gol/demo_gol
cmake-build-release/demo/gol/demo_gol 60 40 --random 0.3
cmake-build-release/demo/gol/demo_gol 200 200 --pattern demo/gol/patterns/gosper-glider-gun.cells --play
```

Click cells to toggle them. The dark glass panels provide play/pause, clear,
randomize, speed, boundary mode, file operations and grid size. Speed cycles
through normal, 2×, 5× and frame rate. Frame rate mode advances one generation
per rendered frame while playing. **Wrap edges** selects periodic boundaries;
turn it off for fixed/dead edges. Width and height range from 2 to 256.

**Open** and **Save** use native file dialogs for `.cells` files. Ctrl+O / Ctrl+S
(Cmd+O / Cmd+S on macOS) trigger the same actions. Files contain `O` for live
cells and `.` for dead cells, with one row per line; lines starting with `!` are
comments. Saving retains exact grid dimensions and empty borders. Opening
centers the pattern and grows either dimension when needed, then pauses playback.
Canceling or failing to open leaves the grid and playback state unchanged.
Files over 1 MiB, unequal row lengths and dimensions above 256 are rejected.

Scroll over the grid to pan, or hold the right mouse button and drag. Pinch to
zoom on a trackpad. Ctrl+scroll also zooms; Ctrl/Cmd + `+`, `-` or `0` zooms in,
out or resets the view.

Options:

- `WIDTH HEIGHT`: grid size (default 64 × 64).
- `--backend auto|vulkan|d3d12|metal`: graphics backend.
- `--random DENSITY`: random starting grid.
- `--pattern FILE`: start from a `.cells` pattern.
- `--play`: start the simulation immediately.
- `--frames N`: exit after `N` rendered frames.
- `--screenshot FILE`: save the last composited frame as PNG.
