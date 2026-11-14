#include "gpu/tiled_life_storage.h"

#include <array>
#include <cstring>

#include "gpu/tiled_life.h"

namespace life_demo {
namespace {
constexpr uint32_t kTiles = TiledLife::kTilesPerAxis * TiledLife::kTilesPerAxis;
constexpr uint32_t kWords = TiledLife::kWordsPerTile;
constexpr uint32_t kRaw = 0xffffffffu;

void Write(std::ostream &out, uint32_t value, int bytes = 4) {
  for (int i = 0; i < bytes; ++i)
    out.put(char(value >> (i * 8)));
}

bool Read(std::istream &in, uint32_t &value, int bytes = 4) {
  value = 0;
  for (int i = 0; i < bytes; ++i) {
    int c = in.get();
    if (c == std::char_traits<char>::eof())
      return false;
    value |= uint32_t(uint8_t(c)) << (i * 8);
  }
  return true;
}
}  // namespace

bool WriteTiledState(std::ostream &output, const std::vector<uint32_t> &words, std::string &error) {
  error.clear();
  if (words.size() != kTiles * kWords) {
    error = "Invalid grid dimensions";
    return false;
  }
  std::array<uint32_t, kTiles> counts{};
  uint32_t count = 0;
  for (uint32_t tile = 0; tile < kTiles; ++tile) {
    for (uint32_t i = 0; i < kWords; ++i)
      counts[tile] += words[tile * kWords + i] != 0;
    count += counts[tile] != 0;
  }
  output.write(kTiledStateMagic, 8);
  Write(output, TiledLife::kWorldSize);
  Write(output, TiledLife::kTileSize);
  Write(output, count);
  for (uint32_t tile = 0; tile < kTiles; ++tile) {
    if (!counts[tile])
      continue;
    bool raw = counts[tile] * 6 >= kWords * 4;
    Write(output, tile);
    Write(output, raw ? kRaw : counts[tile]);
    for (uint32_t i = 0; i < kWords; ++i) {
      uint32_t value = words[tile * kWords + i];
      if (raw)
        Write(output, value);
      else if (value) {
        Write(output, i, 2);
        Write(output, value);
      }
    }
  }
  if (!output) {
    error = "Could not write the grid";
    return false;
  }
  return true;
}

bool ReadTiledState(std::istream &input, std::vector<uint32_t> &words, std::string &error) {
  error = "Invalid sparse grid save file";
  char magic[8];
  if (!input.read(magic, 8) || std::memcmp(magic, kTiledStateMagic, 8))
    return false;
  uint32_t world, tile_size, count;
  if (!Read(input, world) || !Read(input, tile_size) || !Read(input, count) || world != TiledLife::kWorldSize ||
      tile_size != TiledLife::kTileSize || count > kTiles)
    return false;
  std::vector<uint32_t> result(kTiles * kWords, 0);
  uint32_t previous_tile = 0;
  for (uint32_t t = 0; t < count; ++t) {
    uint32_t tile, entries;
    if (!Read(input, tile) || !Read(input, entries) || tile >= kTiles || (t && tile <= previous_tile))
      return false;
    previous_tile = tile;
    if (entries == kRaw) {
      for (uint32_t i = 0; i < kWords; ++i)
        if (!Read(input, result[tile * kWords + i]))
          return false;
    } else {
      if (!entries || entries * uint64_t(6) >= kWords * 4)
        return false;
      uint32_t previous_word = 0;
      for (uint32_t i = 0; i < entries; ++i) {
        uint32_t index, value;
        if (!Read(input, index, 2) || !Read(input, value) || index >= kWords || !value || (i && index <= previous_word))
          return false;
        previous_word = index;
        result[tile * kWords + index] = value;
      }
    }
  }
  if (input.peek() != std::char_traits<char>::eof() || input.bad())
    return false;
  words = std::move(result);
  error.clear();
  return true;
}
}  // namespace life_demo
