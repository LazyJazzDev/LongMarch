// 64x64 tiles, each 256x256 cells packed into 2048 uints (8 KiB).
// Halo per tile: top[8], bottom[8], left[8], right[8], four corner bits.
#ifndef TILES_AXIS
#define TILES_AXIS 64
#endif
#ifndef CACHE_TILES
#define CACHE_TILES 256
#endif
struct Batch {
  uint first_tile;
};

RHI_UNIFORM(0, 3, Batch, batch);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 0, cells);
// Four independent byte cells per uint; no 8-bit storage feature required.
RHI_RW_BYTE_ADDRESS_BUFFER(0, 1, decoded);
RHI_RW_BYTE_ADDRESS_BUFFER(0, 2, halos);

uint Tile(uint x, uint y) {
  return (y & (TILES_AXIS - 1)) * TILES_AXIS + (x & (TILES_AXIS - 1));
}

uint Load(uint tile, uint x, uint y) {
  return cells.Load((tile * 2048 + y * 8 + x) * 4);
}

[numthreads(32, 1, 1)] void ExportHalo(uint3 group
                                       : SV_GroupID, uint lane
                                       : SV_GroupIndex) {
  uint tile = group.x;
  uint value = 0;
  if (lane < 8)
    value = Load(tile, lane, 0);
  else if (lane < 16)
    value = Load(tile, lane - 8, 255);
  else {
    uint side = lane >= 24 ? 7 : 0;
    uint shift = side == 7 ? 31 : 0;
    for (uint i = 0; i < 32; ++i)
      value |= ((Load(tile, side, (lane & 7) * 32 + i) >> shift) & 1) << i;
  }
  halos.Store((tile * 33 + lane) * 4, value);
  if (lane == 0) {
    uint corners = (Load(tile, 0, 0) & 1) | ((Load(tile, 7, 0) >> 31) << 1) | ((Load(tile, 0, 255) & 1) << 2) |
                   ((Load(tile, 7, 255) >> 31) << 3);
    halos.Store((tile * 33 + 32) * 4, corners);
  }
}

    // Decode only the 256x256 main tile. Borders remain in the separate halo buffer.
    [numthreads(256, 1, 1)] void Decode(uint3 group
                                        : SV_GroupID, uint lane
                                        : SV_GroupIndex) {
  for (uint index = lane; index < 2048; index += 256) {
    uint bits = cells.Load(((group.x + batch.first_tile) * 2048 + index) * 4);
    for (uint i = 0; i < 8; ++i) {
      uint nibble = (bits >> (i * 4)) & 15;
      uint bytes = (nibble & 1) | ((nibble & 2) << 7) | ((nibble & 4) << 14) | ((nibble & 8) << 21);
      decoded.Store((group.x * 16384 + index * 8 + i) * 4, bytes);
    }
  }
}

uint Halo(uint tile, uint index) {
  return halos.Load((tile * 33 + index) * 4);
}

uint EdgeCell(uint2 tile, int x, int y) {
  int dx = x < 0 ? -1 : (x >= 256 ? 1 : 0);
  int dy = y < 0 ? -1 : (y >= 256 ? 1 : 0);
  uint neighbor = Tile(tile.x + dx, tile.y + dy);
  if (dx == 0)
    return (Halo(neighbor, (dy < 0 ? 8 : 0) + (uint(x) >> 5)) >> (uint(x) & 31)) & 1;
  if (dy == 0)
    return (Halo(neighbor, (dx < 0 ? 24 : 16) + (uint(y) >> 5)) >> (uint(y) & 31)) & 1;
  uint corner = (dy < 0 ? 2 : 0) + (dx < 0 ? 1 : 0);
  return (Halo(neighbor, 32) >> corner) & 1;
}

// Four old-generation cells, one in each byte. Never read another tile's main data.
uint CachedRow(uint2 tile, int x, int y) {
  if (y >= 0 && y < 256 && x >= 0 && x < 256)
    return decoded.Load(((Tile(tile.x, tile.y) % CACHE_TILES) * 16384 + uint(y) * 64 + uint(x) / 4) * 4);
  if (x < 0)
    return EdgeCell(tile, -1, y) << 24;
  if (x >= 256)
    return EdgeCell(tile, 256, y);
  return EdgeCell(tile, x, y) | (EdgeCell(tile, x + 1, y) << 8) | (EdgeCell(tile, x + 2, y) << 16) |
         (EdgeCell(tile, x + 3, y) << 24);
}

[numthreads(256, 1, 1)] void Step(uint3 group
                                  : SV_GroupID, uint lane
                                  : SV_GroupIndex) {
  uint tile_index = group.x + batch.first_tile;
  uint2 tile = uint2(tile_index % TILES_AXIS, tile_index / TILES_AXIS);
  for (uint index = lane; index < 2048; index += 256) {
    int startX = int(index & 7) * 32, y = int(index >> 3);
    uint result = 0;
    for (uint i = 0; i < 8; ++i) {
      int x = startX + int(i) * 4;
      uint a = CachedRow(tile, x, y - 1);
      uint b = CachedRow(tile, x, y);
      uint c = CachedRow(tile, x, y + 1);
      // Each byte's sum is at most eight: additions cannot carry across cells.
      uint count = a + c;
      count += (a << 8) | (CachedRow(tile, x - 4, y - 1) >> 24);
      count += (a >> 8) | (CachedRow(tile, x + 4, y - 1) << 24);
      count += (b << 8) | (CachedRow(tile, x - 4, y) >> 24);
      count += (b >> 8) | (CachedRow(tile, x + 4, y) << 24);
      count += (c << 8) | (CachedRow(tile, x - 4, y + 1) >> 24);
      count += (c >> 8) | (CachedRow(tile, x + 4, y + 1) << 24);
      for (uint j = 0; j < 4; ++j) {
        uint n = (count >> (j * 8)) & 255;
        uint alive = (b >> (j * 8)) & 1;
        result |= uint(n == 3 || (n == 2 && alive != 0)) << (i * 4 + j);
      }
    }
    // One writer per packed word. Old interior values are in decoded, not cells.
    cells.Store((tile_index * 2048 + index) * 4, result);
  }
}
