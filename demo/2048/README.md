# 2048

The 2048 puzzle game, ported from the GUI of
[Yao-class-cpp-studio/Assignment-3](https://github.com/Yao-class-cpp-studio/Assignment-3)
to `grassland::graphics`. The original targeted the legacy grassland Vulkan
wrapper; this version runs on every LongMarch backend (Vulkan, D3D12, Metal).

```sh
cmake --build build --target demo_2048
build/demo/2048/demo_2048
```

Use the arrow keys to move the blocks. **MENU** opens the menu with
**KEEP GOING** and **NEW GAME**; a game-over screen offers **TRY AGAIN**.

Clicking the **SCORE** board five times in a row hands the game to the autoplay:
the score-board label becomes **AI**, the score board fades to red, and the arrow keys
stop answering until the board is clicked once more. The clicks have to be
consecutive; a pause longer than about a second starts the count over, so the
gesture stays out of the way of ordinary play.

Options:

- `--backend auto|vulkan|d3d12|metal`: select the graphics backend.
- `--frames N`: exit after `N` rendered frames.
- `--screenshot FILE`: save the last frame as PNG on exit.
- `--ai`: start with the autoplay already running.
- `--ai-stop-at N`: with `--ai`, close once a block of `N` has been built, which
  is how a screenshot run stops on a board worth capturing.
- `--ai-benchmark N`, `--ai-budget MS`, `--ai-seed SEED`: play `N` games with
  the autoplay without opening a window and report how far each one got.

## Autoplay

The strategy is an expectimax search over the game's own rules, added on top of
the assignment port rather than taken from it. The parts it reasons about are
the real ones: moves are `update_step` from `2048_lib/`, and spawns come from
`PickSpawnCell` in `game_rules.h`, the same function the interactive game calls.
The search only ever answers with a `Direction`, which the application pushes
through the same input buffer as the arrow keys, so it cannot touch the random
block generator or the score.

`ai_player.h` holds the search. A board is one nibble per cell holding the tile
exponent, and whole rows are resolved through lookup tables, so a move is a
table lookup per row. The player picks the move with the highest expected value,
the game averages over every empty cell a block could spawn in, and leaves are
judged by a score that rewards free cells and merge potential and penalizes
lines that are not monotonic, following the classic `nneonneo/2048-ai`
heuristic and its parameters.

A line carries a log2 probability budget instead of a raw probability cutoff:
spawning a 2 costs `ceil(-log2(0.9 / empty))` units and a 4 costs
`ceil(-log2(0.1 / empty))`, and the line falls back to the heuristic when its
budget is used up, so every unit of the budget is a factor of two. Player nodes
are cached in a direct mapped table keyed on the whole position, the remaining
depth and the remaining budget; a collision only discards cached work, because
the stored key is compared in full. That table is what makes the deeper search
affordable: positions reached by different orders of moves are searched once.

The search runs on a worker thread with 150 ms per move, and it deepens the
spawn depth up to `kSearchDepth` while the budget lasts, so an unfinished layer
never changes the answer: the deepest completed layer decided the move. The
worker keeps the frame rate independent of how long the strategy thinks.
`AiPlayer::ValidateModel` replays random positions through both the search's
move model and `update_step` and reports any disagreement;
`demo_2048 --ai-benchmark` runs it before the games.

`2048_lib/` holds `update_step`, the part students implement in the assignment;
the demo ships a reference implementation that passes the assignment's tests.

Text is vector geometry: `font/` triangulates FreeType glyph outlines, as the
old `grassland::font::Factory` did, without its on-disk cache. The font is
`fonts/ClearSans-Bold-webfont.woff` from the assets submodule (Clear Sans,
Apache License 2.0). Menu and game-over screens fade in by rendering the overlay
into a second frame that the resolve pass blends over the board.

The `application/` layer is shared in copy with `demo/gol`: models are drawn as
instances in framebuffer pixel coordinates with depth testing, rendered into a
supersampled target, and resolved to the window for anti-aliasing.
