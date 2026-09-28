# Cycles shadow terminator geometry offset

`code/sparkium/shaders/shadow_terminator.slang` adapts the geometry-offset
functions from Blender Cycles, copyright 2011–2022 Blender Foundation,
under Apache-2.0 (see LICENSE in this directory).

Source: https://github.com/blender/blender/blob/cc93b7f5a480e2198edd8ed26e70f25b758a8d74/intern/cycles/kernel/light/sample.h

Changes: Slang syntax; geometry loading and object transforms performed by the
mesh hit-record code; separate reflection/transmission offsets; angular weight
exposed as a helper. The original height and weight formulas are retained.
