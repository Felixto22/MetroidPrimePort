#pragma once

#include <cstdint>

#include <webgpu/webgpu_cpp.h>

// Port extension: a copy of the opaque frame for the draws after it to read (see
// GXPortSceneCapture). One RGBA16F texel per pixel: the colour as the EFB holds it, and in alpha
// the world's linear view depth (1e4 for the sky and the viewmodel). Kept as a W x H x 1 3D texture
// so it can stand in the volumetric fog's froxel slot of group 2 (GX glass draws read it there).
namespace aurora::gfx::scenecap {
struct Params {
  float depth[4]; // near, far, and the GX z range the world draws in (min, max)
};
static_assert(sizeof(Params) == 16);

// Registers the task (game thread); false if it could not be.
bool ensure_task();
// Records the copy from the FIFO processor (GX_AURORA_PORT_SCENE_CAPTURE).
bool record(const Params& params);
// The copy the last recorded capture fills (a 1x1x1 placeholder before the first).
const wgpu::TextureView& view();
void shutdown();
} // namespace aurora::gfx::scenecap
