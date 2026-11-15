// The world stays bit-packed in fixed 256x256 storage pages. Compute tiles
// may be smaller, without changing rendering or the sparse save format.
#ifndef TILES_AXIS
#define TILES_AXIS 64
#endif
#ifndef STORAGE_TILES_AXIS
#define STORAGE_TILES_AXIS 64
#endif
#ifndef TILE_SIZE
#define TILE_SIZE 256
#endif
#define WORDS_PER_ROW (TILE_SIZE / 32)
#define WORDS_PER_TILE (TILE_SIZE * TILE_SIZE / 32)
#define HALO_WORDS (4 * WORDS_PER_ROW + 1)
#ifndef GROUP_THREADS
#define GROUP_THREADS 256
#endif
struct Batch {
  uint first_tile;
  uint x;
  uint y;
  uint value;
};

RHI_UNIFORM(0, 2, Batch, batch);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 0, cells);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 1, halos);

uint Tile(uint x, uint y) {
  return (y & (TILES_AXIS - 1)) * TILES_AXIS + (x & (TILES_AXIS - 1));
}

uint Address(uint tile, uint x, uint y) {
  uint wx = (tile % TILES_AXIS) * TILE_SIZE + x * 32;
  uint wy = (tile / TILES_AXIS) * TILE_SIZE + y;
  uint page = (wy / 256) * STORAGE_TILES_AXIS + wx / 256;
  return (page * 2048 + (wy & 255) * 8 + (wx & 255) / 32) * 4;
}

uint Load(uint tile, uint x, uint y) {
  return cells.Load(Address(tile, x, y));
}

// Pack multiple tiles' border words into each workgroup. Small compute tiles
// no longer launch mostly idle groups for their few border words.
[numthreads(128, 1, 1)] void ExportHalo(uint3 thread
                                        : SV_DispatchThreadID) {
  uint tile = thread.x / (4 * WORDS_PER_ROW);
  uint lane = thread.x % (4 * WORDS_PER_ROW);
  if (tile >= TILES_AXIS * TILES_AXIS)
    return;
  uint value = 0;
  if (lane < WORDS_PER_ROW)
    value = Load(tile, lane, 0);
  else if (lane < 2 * WORDS_PER_ROW)
    value = Load(tile, lane - WORDS_PER_ROW, TILE_SIZE - 1);
  else {
    uint side = lane >= 3 * WORDS_PER_ROW ? WORDS_PER_ROW - 1 : 0;
    uint shift = lane >= 3 * WORDS_PER_ROW ? 31 : 0;
    for (uint i = 0; i < 32; ++i)
      value |= ((Load(tile, side, (lane % WORDS_PER_ROW) * 32 + i) >> shift) & 1) << i;
  }
  halos.Store((tile * HALO_WORDS + lane) * 4, value);
  if (lane == 0) {
    uint corners = (Load(tile, 0, 0) & 1) | ((Load(tile, WORDS_PER_ROW - 1, 0) >> 31) << 1) |
                   ((Load(tile, 0, TILE_SIZE - 1) & 1) << 2) |
                   ((Load(tile, WORDS_PER_ROW - 1, TILE_SIZE - 1) >> 31) << 3);
    halos.Store((tile * HALO_WORDS + HALO_WORDS - 1) * 4, corners);
  }
}

uint Halo(uint tile, uint index) {
  return halos.Load((tile * HALO_WORDS + index) * 4);
}

uint EdgeCell(uint2 tile, int x, int y) {
  int dx = x < 0 ? -1 : (x >= TILE_SIZE ? 1 : 0);
  int dy = y < 0 ? -1 : (y >= TILE_SIZE ? 1 : 0);
  if (batch.value == 0 && (int(tile.x) + dx < 0 || int(tile.y) + dy < 0 || int(tile.x) + dx >= TILES_AXIS ||
                           int(tile.y) + dy >= TILES_AXIS))
    return 0;
  uint neighbor = Tile(tile.x + dx, tile.y + dy);
  if (dx == 0)
    return (Halo(neighbor, (dy < 0 ? WORDS_PER_ROW : 0) + (uint(x) >> 5)) >> (uint(x) & 31)) & 1;
  if (dy == 0)
    return (Halo(neighbor, (dx < 0 ? 3 * WORDS_PER_ROW : 2 * WORDS_PER_ROW) + (uint(y) >> 5)) >> (uint(y) & 31)) & 1;
  uint corner = (dy < 0 ? 2 : 0) + (dx < 0 ? 1 : 0);
  return (Halo(neighbor, HALO_WORDS - 1) >> corner) & 1;
}

// Bit planes decode each tile into old states and two horizontal-count bits.
// A uint holds the same bit for 32 cells, so the vertical scan updates all
// 32 cells together without expanding and repacking eight-cell nibbles.
// Three bits per cell use 24 KiB for 256x256 or 1.5 KiB for 64x64.
groupshared uint old_states[WORDS_PER_TILE];
groupshared uint horizontal_low[WORDS_PER_TILE];
groupshared uint horizontal_high[WORDS_PER_TILE];

uint2 HorizontalBits(uint center, uint left, uint right) {
  return uint2(center ^ left ^ right, (center & left) | (right & (center | left)));
}

uint2 HorizontalAt(uint index) {
  return uint2(horizontal_low[index], horizontal_high[index]);
}

uint NextWord(uint2 a, uint2 b, uint2 c, uint old) {
  uint ones = a.x ^ b.x ^ c.x;
  uint carry = (a.x & b.x) | (c.x & (a.x | b.x));
  uint twos = a.y ^ b.y ^ c.y;
  uint fours = ((a.y & b.y) | (c.y & (a.y | b.y))) ^ (twos & carry);
  twos ^= carry;
  // Inclusive sums 8 and 9 alias 0 and 1 in these three planes. All four
  // totals imply death, so the unused eight's bit cannot affect the rule.
  return (ones & twos & ~fours) | (~(ones | twos) & fours & old);
}

// Read a whole old top/bottom border word once, before expanding its cells.
uint HaloRowWord(uint2 tile, uint word, int y) {
  if (batch.value == 0 && ((y < 0 && tile.y == 0) || (y >= TILE_SIZE && tile.y + 1 == TILES_AXIS)))
    return 0;
  uint neighbor = Tile(tile.x, tile.y + (y < 0 ? -1 : 1));
  return Halo(neighbor, (y < 0 ? WORDS_PER_ROW : 0) + word);
}

[numthreads(GROUP_THREADS, 1, 1)] void Step(uint3 group
                                            : SV_GroupID, uint lane
                                            : SV_GroupIndex) {
  uint tile_index = group.y * TILES_AXIS + group.x;
  uint2 tile = uint2(group.x, group.y);
  // Fuse decoding with the horizontal scan. All main reads complete at the
  // barrier before any thread writes a new generation into this tile.
  for (uint i = lane; i < WORDS_PER_TILE; i += GROUP_THREADS) {
    uint xword = i % WORDS_PER_ROW, y = i / WORDS_PER_ROW;
    uint bits = Load(tile_index, xword, y);
    uint left = xword > 0 ? Load(tile_index, xword - 1, y) >> 31 : EdgeCell(tile, -1, int(y));
    uint right = xword + 1 < WORDS_PER_ROW ? Load(tile_index, xword + 1, y) & 1 : EdgeCell(tile, TILE_SIZE, int(y));
    uint shiftedLeft = (bits << 1) | left;
    uint shiftedRight = (bits >> 1) | (right << 31);
    uint2 count = HorizontalBits(bits, shiftedLeft, shiftedRight);
    old_states[i] = bits;
    horizontal_low[i] = count.x;
    horizontal_high[i] = count.y;
  }
  GroupMemoryBarrierWithGroupSync();
  // Interior rows have no halo branches. Reuse the center read for state
  // and count instead of routing every packet through the generic accessor.
  for (uint index = lane + WORDS_PER_ROW; index < WORDS_PER_TILE - WORDS_PER_ROW; index += GROUP_THREADS) {
    uint result = NextWord(HorizontalAt(index - WORDS_PER_ROW), HorizontalAt(index),
                           HorizontalAt(index + WORDS_PER_ROW), old_states[index]);
    cells.Store(Address(tile_index, index % WORDS_PER_ROW, index / WORDS_PER_ROW), result);
  }
  // Only the two boundary rows consult old neighboring halos.
  if (lane < 2 * WORDS_PER_ROW) {
    uint word = lane % WORDS_PER_ROW;
    int y = lane < WORDS_PER_ROW ? 0 : TILE_SIZE - 1;
    int outerY = lane < WORDS_PER_ROW ? -1 : TILE_SIZE;
    uint innerY = lane < WORDS_PER_ROW ? 1 : TILE_SIZE - 2;
    uint index = uint(y) * WORDS_PER_ROW + word;
    uint bits = HaloRowWord(tile, word, outerY);
    uint left = word > 0 ? HaloRowWord(tile, word - 1, outerY) >> 31 : EdgeCell(tile, -1, outerY);
    uint right = word + 1 < WORDS_PER_ROW ? HaloRowWord(tile, word + 1, outerY) & 1 : EdgeCell(tile, TILE_SIZE, outerY);
    uint shiftedLeft = (bits << 1) | left;
    uint shiftedRight = (bits >> 1) | (right << 31);
    uint result = NextWord(HorizontalBits(bits, shiftedLeft, shiftedRight), HorizontalAt(index),
                           HorizontalAt(innerY * WORDS_PER_ROW + word), old_states[index]);
    cells.Store(Address(tile_index, word, uint(y)), result);
  }
}

    [numthreads(1, 1, 1)] void Toggle() {
  uint tile = Tile(batch.x / TILE_SIZE, batch.y / TILE_SIZE);
  uint address = Address(tile, (batch.x % TILE_SIZE) / 32, batch.y % TILE_SIZE);
  cells.Store(address, cells.Load(address) ^ (1u << (batch.x & 31)));
}

uint Hash(uint v) {
  v ^= v >> 16;
  v *= 0x7feb352d;
  v ^= v >> 15;
  v *= 0x846ca68b;
  return v ^ (v >> 16);
}
[numthreads(GROUP_THREADS, 1, 1)] void Fill(uint3 group
                                            : SV_GroupID, uint lane
                                            : SV_GroupIndex) {
  for (uint index = lane; index < WORDS_PER_TILE; index += GROUP_THREADS) {
    uint word = (group.y * TILES_AXIS + group.x) * WORDS_PER_TILE + index;
    cells.Store(word * 4, batch.value == 0 ? 0 : Hash(word ^ batch.x));
  }
}
