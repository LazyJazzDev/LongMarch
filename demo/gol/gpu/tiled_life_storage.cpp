#include "gpu/tiled_life_storage.h"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <sstream>

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

static bool ReadLegacyState(std::istream &input, std::vector<uint32_t> &words, std::string &error) {
  error = "Invalid sparse grid save file";
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

namespace {
// V2: 24-byte header, followed by count 24-byte index entries, then payloads.
// Each index entry: tile, payload-relative offset, stored size, decoded size,
// flags (bit 0: indexed words, bit 1: zlib), CRC32 of decoded payload.
struct Entry {
  uint32_t tile, offset, stored, decoded, flags, checksum;
};

constexpr uint32_t kIndexedWords = 1, kCompressed = 2;

uint32_t Checksum(const std::string &bytes) {
  return uint32_t(crc32(0, reinterpret_cast<const Bytef *>(bytes.data()), uInt(bytes.size())));
}

std::string Compress(const std::string &bytes) {
  std::string compressed(compressBound(uLong(bytes.size())), '\0');
  uLongf size = compressed.size();
  if (compress2(reinterpret_cast<Bytef *>(compressed.data()), &size, reinterpret_cast<const Bytef *>(bytes.data()),
                uLong(bytes.size()), Z_BEST_SPEED) != Z_OK)
    return {};
  compressed.resize(size);
  return compressed;
}

bool InflateExact(const std::string &encoded, std::string &decoded) {
  z_stream stream{};
  stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(encoded.data()));
  stream.avail_in = uInt(encoded.size());
  stream.next_out = reinterpret_cast<Bytef *>(decoded.data());
  stream.avail_out = uInt(decoded.size());
  if (inflateInit(&stream) != Z_OK)
    return false;
  const int status = inflate(&stream, Z_FINISH);
  const bool ok = status == Z_STREAM_END && stream.total_in == encoded.size() && stream.total_out == decoded.size();
  inflateEnd(&stream);
  return ok;
}
}  // namespace

bool WriteTiledState(std::ostream &output, const std::vector<uint32_t> &words, std::string &error) {
  error = "Could not write the grid";
  if (words.size() != kTiles * kWords) {
    error = "Invalid grid dimensions";
    return false;
  }
  std::vector<Entry> entries;
  std::string payload;
  for (uint32_t tile = 0; tile < kTiles; ++tile) {
    const auto begin = words.begin() + tile * kWords;
    const auto nonzero = uint32_t(std::count_if(begin, begin + kWords, [](uint32_t value) { return value != 0; }));
    if (!nonzero)
      continue;
    const bool use_sparse = nonzero * 6 < kWords * 4;
    std::ostringstream raw(std::ios::binary), sparse(std::ios::binary);
    for (uint32_t i = 0; i < kWords; ++i) {
      const uint32_t value = words[tile * kWords + i];
      Write(raw, value);
      if (value && use_sparse) {
        Write(sparse, i, 2);
        Write(sparse, value);
      }
    }
    const auto dense = raw.str();
    auto best = dense;
    uint32_t flags = 0, decoded = uint32_t(dense.size()), checksum = Checksum(dense);
    auto consider = [&](const std::string &source, uint32_t representation) {
      if (source.size() < best.size()) {
        best = source;
        flags = representation;
        decoded = uint32_t(source.size());
        checksum = Checksum(source);
      }
      auto compressed = Compress(source);
      if (!compressed.empty() && compressed.size() < best.size()) {
        best = std::move(compressed);
        flags = representation | kCompressed;
        decoded = uint32_t(source.size());
        checksum = Checksum(source);
      }
    };
    consider(dense, 0);
    // Sparse word indices are worthwhile only below the raw tile's size.
    if (use_sparse)
      consider(sparse.str(), kIndexedWords);
    entries.push_back({tile, uint32_t(payload.size()), uint32_t(best.size()), decoded, flags, checksum});
    payload += best;
  }
  std::ostringstream index(std::ios::binary);
  for (const auto &entry : entries)
    for (uint32_t value : {entry.tile, entry.offset, entry.stored, entry.decoded, entry.flags, entry.checksum})
      Write(index, value);
  const auto table = index.str();
  output.write(kTiledStateMagic, 8);
  Write(output, TiledLife::kWorldSize);
  Write(output, TiledLife::kTileSize);
  Write(output, uint32_t(entries.size()));
  Write(output, Checksum(table));
  output.write(table.data(), std::streamsize(table.size()));
  output.write(payload.data(), std::streamsize(payload.size()));
  if (!output)
    return false;
  error.clear();
  return true;
}

bool ReadTiledState(std::istream &input, std::vector<uint32_t> &words, std::string &error) {
  error = "Invalid sparse grid save file";
  char magic[8];
  if (!input.read(magic, 8))
    return false;
  if (!std::memcmp(magic, kLegacyTiledStateMagic, 8)) {
    return ReadLegacyState(input, words, error);
  }
  if (std::memcmp(magic, kTiledStateMagic, 8))
    return false;
  uint32_t world, tile_size, count, index_crc;
  if (!Read(input, world) || !Read(input, tile_size) || !Read(input, count) || !Read(input, index_crc) ||
      world != TiledLife::kWorldSize || tile_size != TiledLife::kTileSize || count > kTiles)
    return false;
  std::string table(count * 24, '\0');
  if (!input.read(table.data(), std::streamsize(table.size())) || Checksum(table) != index_crc)
    return false;
  std::istringstream index(table, std::ios::binary);
  std::vector<Entry> entries(count);
  uint32_t offset = 0, previous_tile = 0;
  for (uint32_t i = 0; i < count; ++i) {
    auto &entry = entries[i];
    if (!Read(index, entry.tile) || !Read(index, entry.offset) || !Read(index, entry.stored) ||
        !Read(index, entry.decoded) || !Read(index, entry.flags) || !Read(index, entry.checksum) ||
        entry.tile >= kTiles || (i && entry.tile <= previous_tile) || entry.offset != offset || entry.flags > 3 ||
        !entry.stored || entry.stored > kWords * 4 || !entry.decoded || entry.decoded > kWords * 4)
      return false;
    if (entry.flags & kIndexedWords) {
      if (entry.decoded % 6 || entry.decoded >= kWords * 4)
        return false;
    } else if (entry.decoded != kWords * 4)
      return false;
    if (entry.flags & kCompressed) {
      if (entry.stored >= entry.decoded)
        return false;
    } else if (entry.stored != entry.decoded)
      return false;
    offset += entry.stored;
    previous_tile = entry.tile;
  }
  std::vector<uint32_t> result(kTiles * kWords, 0);
  for (const auto &entry : entries) {
    std::string encoded(entry.stored, '\0'), decoded(entry.decoded, '\0');
    if (!input.read(encoded.data(), std::streamsize(encoded.size())))
      return false;
    if (entry.flags & kCompressed) {
      if (!InflateExact(encoded, decoded))
        return false;
    } else
      decoded = std::move(encoded);
    if (Checksum(decoded) != entry.checksum)
      return false;
    std::istringstream data(decoded, std::ios::binary);
    if (entry.flags & kIndexedWords) {
      uint32_t previous = 0;
      for (uint32_t i = 0; i < entry.decoded / 6; ++i) {
        uint32_t word, value;
        if (!Read(data, word, 2) || !Read(data, value) || word >= kWords || !value || (i && word <= previous))
          return false;
        result[entry.tile * kWords + word] = value;
        previous = word;
      }
    } else {
      for (uint32_t i = 0; i < kWords; ++i)
        if (!Read(data, result[entry.tile * kWords + i]))
          return false;
    }
  }
  if (input.peek() != std::char_traits<char>::eof() || input.bad())
    return false;
  words = std::move(result);
  error.clear();
  return true;
}
}  // namespace life_demo
