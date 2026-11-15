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
#define SHARED_WORDS (TILE_SIZE * TILE_SIZE / 8)
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

[numthreads(32, 1, 1)] void ExportHalo(uint3 group
                                       : SV_GroupID, uint lane
                                       : SV_GroupIndex) {
  uint tile = group.y * TILES_AXIS + group.x;
  if (lane >= 4 * WORDS_PER_ROW)
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

// Eight four-bit cells per uint; a 256x256 tile uses 32 KiB, a 128x128 tile 8 KiB.
// During the stencil pass only the old state bit is shared; counters stay in
// registers so no thread overwrites a neighbor's old state.
groupshared uint local_cells[SHARED_WORDS];

uint Expand(uint bits) {
  bits &= 255u;
  bits = (bits | (bits << 12)) & 0x000f000fu;
  bits = (bits | (bits << 6)) & 0x03030303u;
  return (bits | (bits << 3)) & 0x11111111u;
}

uint Pack(uint cells8) {
  cells8 = (cells8 | (cells8 >> 3)) & 0x03030303u;
  cells8 = (cells8 | (cells8 >> 6)) & 0x000f000fu;
  return (cells8 | (cells8 >> 12)) & 255u;
}

uint EqualNibble(uint a, uint b) {
  uint different = a ^ b;
  return ~(different | (different >> 1) | (different >> 2) | (different >> 3)) & 0x11111111u;
}

uint CachedRow(uint2 tile, int x, int y) {
  if (y >= 0 && y < TILE_SIZE && x >= 0 && x < TILE_SIZE)
    return local_cells[uint(y) * (TILE_SIZE / 8) + uint(x) / 8];
  if (x < 0)
    return EdgeCell(tile, -1, y) << 28;
  if (x >= TILE_SIZE)
    return EdgeCell(tile, TILE_SIZE, y);
  if (batch.value == 0 && ((y < 0 && tile.y == 0) || (y >= TILE_SIZE && tile.y + 1 == TILES_AXIS)))
    return 0;
  uint neighbor = Tile(tile.x, tile.y + (y < 0 ? -1 : 1));
  return Expand(Halo(neighbor, (y < 0 ? WORDS_PER_ROW : 0) + uint(x) / 32) >> (uint(x) & 31));
}

[numthreads(GROUP_THREADS, 1, 1)] void Step(uint3 group
                                            : SV_GroupID, uint lane
                                            : SV_GroupIndex) {
  uint tile_index = group.y * TILES_AXIS + group.x;
  uint2 tile = uint2(tile_index % TILES_AXIS, tile_index / TILES_AXIS);
  // Decode the entire old main tile before writing any main bit in place.
  for (uint i = lane; i < WORDS_PER_TILE; i += GROUP_THREADS) {
    uint bits = Load(tile_index, i % WORDS_PER_ROW, i / WORDS_PER_ROW);
    for (uint j = 0; j < 4; ++j)
      local_cells[i * 4 + j] = Expand(bits >> (j * 8));
  }
  GroupMemoryBarrierWithGroupSync();
  for (uint index = lane; index < WORDS_PER_TILE; index += GROUP_THREADS) {
    int startX = int(index % WORDS_PER_ROW) * 32, y = int(index / WORDS_PER_ROW);
    uint result = 0;
    for (uint i = 0; i < 4; ++i) {
      int x = startX + int(i) * 8;
      uint a = CachedRow(tile, x, y - 1);
      uint b = CachedRow(tile, x, y);
      uint c = CachedRow(tile, x, y + 1);
      // Sum <= 8 fits in four bits, so packed additions cannot carry into
      // the next cell. Only afterwards truncate to the three-bit counter.
      uint count = a + c;
      count += (a << 4) | (CachedRow(tile, x - 8, y - 1) >> 28);
      count += (a >> 4) | (CachedRow(tile, x + 8, y - 1) << 28);
      count += (b << 4) | (CachedRow(tile, x - 8, y) >> 28);
      count += (b >> 4) | (CachedRow(tile, x + 8, y) << 28);
      count += (c << 4) | (CachedRow(tile, x - 8, y + 1) >> 28);
      count += (c >> 4) | (CachedRow(tile, x + 8, y + 1) << 28);
      // Per cell: bit 0 = old state, bits 1..3 = count modulo 8.
      // Count 8 becomes 0; both imply death, so this preserves Life's rule.
      uint encoded = ((count & 0x77777777u) << 1) | b;
      uint counters = encoded & 0xeeeeeeeeu;
      uint alive = EqualNibble(counters, 0x66666666u) | (EqualNibble(counters, 0x44444444u) & encoded);
      result |= Pack(alive) << (i * 8);
    }
    cells.Store(Address(tile_index, index % WORDS_PER_ROW, index / WORDS_PER_ROW), result);
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
