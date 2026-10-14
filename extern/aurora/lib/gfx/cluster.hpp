#pragma once

#include <cstdint>
#include <vector>

// Port extension: clustered lights (GX_AURORA_PORT_CLUSTER_LIGHTS). The scene's dynamic lights,
// binned into a view-space grid around the camera, for the per-pixel GX lighting to sum
// (ShaderConfig::clustered) on top of the few GX lights each draw is given.
//
// Layout in the frame's storage buffer, in u32 words from the base:
//   [0, CellCount)               per cell: first index (low 16 bits) and light count (high 16)
//   [CellCount, + 12 * lights)   per light: view position xyz, radius; colour rgb, 0;
//                                GX distance attenuation k0, k1, k2, 0 (all f32 bits)
//   [..., + indices)             the cells' light indices
namespace aurora::gfx::cluster {
constexpr float GridMin[3] = {-30.f, -15.f, -70.f};
constexpr float CellSize = 2.5f;
constexpr int GridX = 24;
constexpr int GridY = 12;
constexpr int GridZ = 28;
constexpr uint32_t CellCount = GridX * GridY * GridZ;
constexpr uint32_t MaxLights = 256;
constexpr uint32_t MaxPerCell = 24;

struct Light {
  float pos[3];
  float radius;
  float color[3];
  float pad0;
  float atten[3];
  float pad1;
};
static_assert(sizeof(Light) == 48);

// The words above for `lights` (at most MaxLights are kept).
void build(const Light* lights, uint32_t count, std::vector<uint32_t>& out);
} // namespace aurora::gfx::cluster
