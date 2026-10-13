#pragma once

#include <cstdint>

// Port extension: screen-space ambient occlusion over the opaque world as drawn so far (see
// GXPortSSAO). Not in Remastered: the retail and Remastered worlds both get it.
//  - pass 1, at the frame's size: per pixel, the view-space position from the depth buffer, a
//    normal from its neighbours, and a cosine-weighted hemisphere of samples around that normal
//    (rotated per pixel by interleaved gradient noise); a sample the depth buffer has something
//    in front of occludes, faded by how far that surface is from the pixel;
//  - pass 2: a depth-aware 5x5 blur of the occlusion, multiplied into the frame's linear colour.
// The viewmodel (nearer than the world's depth range) and the sky (at the far plane) are left
// alone, and the effect fades out with distance so fogged far walls keep their colour.
namespace aurora::gfx::ssao {
// The task's uniform, word for word (GXPortSsaoParams).
struct Params {
  float frustum[4]; // left, right, bottom, top at a view depth of 1
  float depth[4];   // near, far, and the GX z range the world draws in (min, max)
  float ao[4];      // radius (world units), strength (0..1), bias (world units), fade distance
  uint32_t misc[4]; // sample count (4..32), frame counter (noise), 0, 0
};
static_assert(sizeof(Params) == 16 * 4);

// Registers the task (game thread); false if it could not be.
bool ensure_task();
// Records the pass from the FIFO processor (GX_AURORA_PORT_SSAO), once ensure_task has returned true.
bool record(const Params& params);
void shutdown();
} // namespace aurora::gfx::ssao
