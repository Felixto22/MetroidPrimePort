#pragma once

#include <cstdint>

// Port extension: geometry the GPU builds itself, drawn inline with the frame's opaque world (the
// work WebGPU's lack of mesh, hull and geometry shaders would otherwise have given those stages):
//  - grass: per area, a buffer of blades (a root each) the vertex shader turns into curved,
//    tapered blades bending in the wind and away from what moves through them;
//  - rocks: chunks of stone the vertex shader builds from a seed (an irregular, faceted lump),
//    for debris knocked loose by shots.
// Both are lit by their own colour (already the room's light, as the ground they come from shows
// it), shaded by their facing, and fogged with the GX fog of the draw.
namespace aurora::gfx::geo {
struct Blade {
  float pos[3];   // the root, world space
  float yaw;      // the blade's facing (radians)
  float height;   // metres
  float width;    // metres at the root
  uint32_t color; // 0xRRGGBB at the tip (the root is darker)
  uint32_t seed;  // phase and curl
};
static_assert(sizeof(Blade) == 32);

// The blades of key (an area), kept on the GPU until replaced; null/0 forgets them. Game thread.
void set_grass(uint64_t key, const Blade* blades, uint32_t count);

struct GrassDraw {
  uint64_t key = 0;
  uint32_t first = 0; // the blades drawn: [first, first + count)
  uint32_t count = 0;
  float time = 0.f;             // seconds, for the wind
  float wind[2] = {0.7f, 0.4f}; // world direction
  float windStrength = 1.f;
  float fadeStart = 25.f;       // blades shrink away between these distances (m)
  float fadeEnd = 40.f;
  // What moves through the grass (Samus, shots): world position, radius.
  static constexpr uint32_t MaxPushers = 8;
  uint32_t pusherCount = 0;
  float pushers[MaxPushers][4] = {};
};
// Records the blades into the current pass with the GX state's projection and position matrix
// (set it to the view: the blades are in world space).
void draw_grass(const GrassDraw& draw);

struct Rock {
  float pos[3];   // world space
  float scale;    // metres across
  float rot[4];   // orientation quaternion (x, y, z, w)
  uint32_t color; // 0xRRGGBB
  uint32_t seed;  // its shape
};
static_assert(sizeof(Rock) == 40);
// Records rocks into the current pass, as draw_grass.
void draw_rocks(const Rock* rocks, uint32_t count);

void shutdown();
} // namespace aurora::gfx::geo
