#include "grassland/graphics/buffer.h"

namespace grassland::graphics {

BufferRange::BufferRange(Buffer *buffer, size_t offset, size_t size) : buffer(buffer), offset(offset), size(size) {
  if (buffer) {
    this->size = std::min(size, buffer->Size() - offset);
  }
}

void Buffer::PybindClassRegistration(py::classh<Buffer> &c) {
  c.def("type", &Buffer::Type);
  c.def("size", &Buffer::Size);
  c.def("resize", &Buffer::Resize, py::arg("new_size"));
  c.def(
      "upload_data",
      [](Buffer *buffer, py::bytes data, size_t size, size_t offset) {
        if (size == ~0ull) {
          size = PyBytes_Size(data.ptr());
        }
        if (size + offset > buffer->Size()) {
          throw std::runtime_error("Data size exceeds buffer size");
        }
        buffer->UploadData(PyBytes_AsString(data.ptr()), size, offset);
      },
      py::arg("data"), py::arg("size") = ~0ull, py::arg("offset") = 0);
  c.def(
      "download_data",
      [](Buffer *buffer, size_t size, size_t offset) {
        if (size == ~0ull) {
          size = buffer->Size() - offset;
        }
        if (size + offset > buffer->Size()) {
          throw std::runtime_error("Data size exceeds buffer size");
        }
        std::vector<char> data(size);
        buffer->DownloadData(data.data(), size, offset);
        return py::bytes(data.data(), size);
      },
      py::arg("size") = ~0ull, py::arg("offset") = 0);
}

}  // namespace grassland::graphics
