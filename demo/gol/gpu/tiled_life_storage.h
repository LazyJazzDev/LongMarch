#pragma once

#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

namespace life_demo {
// V2 writes a sparse tile index and independently zlib-compressed payloads.
// Both V1 and V2 are accepted; failed reads leave the destination unchanged.
// Integers are little-endian. Header: magic[8], world:u32, tile_side:u32,
// tile_count:u32, index_crc32:u32. The index has tile_count entries containing
// six u32 fields: tile_id, payload_offset, stored_bytes, decoded_bytes, flags,
// decoded_crc32. Offsets are relative to the end of the index. Tiles and payloads
// are ordered, contiguous, and nonoverlapping. Empty tiles are omitted.
// Flag bit 0 selects sorted (word_index:u16, value:u32) pairs instead of 2048
// raw u32 words; bit 1 wraps that representation in a zlib (RFC 1950) stream.
// CRC32 covers the exact index bytes and each decompressed tile representation.
// Decoded payloads never exceed 8192 bytes; no trailing data is accepted.
bool WriteTiledState(std::ostream &output, const std::vector<uint32_t> &words, std::string &error);
bool ReadTiledState(std::istream &input, std::vector<uint32_t> &words, std::string &error);
inline constexpr char kLegacyTiledStateMagic[] = "GOLSPRS1";
inline constexpr char kTiledStateMagic[] = "GOLSPRS2";
}  // namespace life_demo
