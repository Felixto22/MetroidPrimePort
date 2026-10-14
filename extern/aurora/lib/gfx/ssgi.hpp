#pragma once

#include <cstdint>

// Port extension: screen-space global illumination over the opaque world as drawn so far (see
// GXPortSSGI). Light bounces: a lit wall tints the floor beside it, a shot's glow and a lava
// pool colour what surrounds them. Not in Remastered.
//  - a compute pass at half the frame's size: per texel, the view-space position and normal from
//    the depth buffer, and rays over the cosine-weighted hemisphere around the normal, marched in
//    screen space through the depth buffer; a ray that finds a surface brings back its colour
//    from a copy of the frame (the light it shows), weaker the farther it went;
//  - a full-screen pass: that bounce light, blurred over texels at about the same depth, added
//    to each pixel tinted by the pixel's own colour (its albedo, as far as the frame can tell).
// The viewmodel and the sky neither receive nor give it; it fades out with distance.
namespace aurora::gfx::ssgi {
// The task's uniform, word for word (GXPortSsgiParams).
struct Params {
  float frustum[4]; // left, right, bottom, top at a view depth of 1
  float depth[4];   // near, far, and the GX z range the world draws in (min, max)
  float gi[4];      // ray length (world units), strength (0..2), thickness (world units), fade distance
  uint32_t misc[4]; // rays per texel (2..16), steps per ray (4..24), 0, 0
};
static_assert(sizeof(Params) == 16 * 4);

// Registers the task (game thread); false if it could not be.
bool ensure_task();
// Records the passes from the FIFO processor (GX_AURORA_PORT_SSGI), once ensure_task has returned true.
bool record(const Params& params);
void shutdown();
} // namespace aurora::gfx::ssgi
