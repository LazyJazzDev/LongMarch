#include "grassland/graphics/buffer.h"

namespace grassland::graphics {

void Buffer::UploadData(const void *data, size_t size) {
  UploadData(data, size, 0);
}

void Buffer::DownloadData(void *data, size_t size) {
  DownloadData(data, size, 0);
}

}  // namespace grassland::graphics
