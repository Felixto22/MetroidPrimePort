#include "cluster.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace aurora::gfx::cluster {
namespace {
int cell_of(float v, int axis, int count) {
  return static_cast<int>(std::floor((v - GridMin[axis]) / CellSize));
}
} // namespace

void build(const Light* lights, uint32_t count, std::vector<uint32_t>& out) {
  count = std::min(count, MaxLights);
  out.assign(CellCount + count * 12u, 0u);
  std::memcpy(out.data() + CellCount, lights, count * sizeof(Light));
  // Each light's range of cells, clipped to the grid.
  struct Range {
    int lo[3], hi[3];
  };
  static thread_local std::vector<Range> ranges;
  static thread_local std::vector<uint32_t> cellCount;
  ranges.resize(count);
  cellCount.assign(CellCount, 0u);
  const int dims[3] = {GridX, GridY, GridZ};
  for (uint32_t l = 0; l < count; ++l) {
    Range& r = ranges[l];
    for (int a = 0; a < 3; ++a) {
      r.lo[a] = std::max(0, cell_of(lights[l].pos[a] - lights[l].radius, a, dims[a]));
      r.hi[a] = std::min(dims[a] - 1, cell_of(lights[l].pos[a] + lights[l].radius, a, dims[a]));
    }
  }
  // Count, then lay the indices out cell by cell.
  const auto each = [&](uint32_t l, auto&& fn) {
    const Range& r = ranges[l];
    const float* c = lights[l].pos;
    const float rad2 = lights[l].radius * lights[l].radius;
    for (int z = r.lo[2]; z <= r.hi[2]; ++z) {
      for (int y = r.lo[1]; y <= r.hi[1]; ++y) {
        for (int x = r.lo[0]; x <= r.hi[0]; ++x) {
          // Sphere against the cell's box.
          const float mn[3] = {GridMin[0] + x * CellSize, GridMin[1] + y * CellSize, GridMin[2] + z * CellSize};
          float d2 = 0.f;
          for (int a = 0; a < 3; ++a) {
            const float v = std::clamp(c[a], mn[a], mn[a] + CellSize);
            d2 += (v - c[a]) * (v - c[a]);
          }
          if (d2 <= rad2) {
            fn(static_cast<uint32_t>(x + GridX * (y + GridY * z)));
          }
        }
      }
    }
  };
  for (uint32_t l = 0; l < count; ++l) {
    each(l, [&](uint32_t cell) {
      if (cellCount[cell] < MaxPerCell) {
        ++cellCount[cell];
      }
    });
  }
  uint32_t total = 0;
  for (uint32_t c = 0; c < CellCount; ++c) {
    if (total + cellCount[c] > 0xffffu) {
      cellCount[c] = 0;
    }
    out[c] = total; // count filled in below
    total += cellCount[c];
  }
  const size_t indexBase = out.size();
  out.resize(indexBase + total, 0u);
  for (uint32_t l = 0; l < count; ++l) {
    each(l, [&](uint32_t cell) {
      const uint32_t first = out[cell] & 0xffffu;
      const uint32_t used = out[cell] >> 16u;
      if (used < cellCount[cell]) {
        out[indexBase + first + used] = l;
        out[cell] = first | ((used + 1u) << 16u);
      }
    });
  }
}
} // namespace aurora::gfx::cluster
