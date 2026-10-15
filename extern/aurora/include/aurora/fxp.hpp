#pragma once

#include <cstdint>

// Port extension: the GPU particle layer drawn over the game's own effects. Each particle is
// stateless: the vertex shader puts it where its launch (position, velocity, drag, gravity, a
// floor it bounces off) has taken it by now and draws it as a camera-facing quad, a streak along
// its motion, or flat on the ground; the fragment shader shapes it (glow, spark, smoke, drop,
// flake, ring, print, haze, arc, bubble) with no texture. Fields (rain, snow, dust, embers) are
// whole volumes around the camera made from the instance index alone, hidden under what roofs
// them (a height grid the game fills). Soft against the scene capture's depth; the haze bends it.
namespace aurora::gfx::fxp {
enum Shape : uint32_t {
  Glow = 0,    // soft round light (additive)
  Spark = 1,   // streak along its motion, hot core (additive)
  Smoke = 2,   // billowing puff that grows (blended)
  Drop = 3,    // water drop streak (blended)
  Flake = 4,   // soft snowflake (blended)
  Ring = 5,    // shockwave ring that grows (additive)
  Print = 6,   // flat on the ground: a footprint or a track (blended, dark)
  Haze = 7,    // heat shimmer: bends what is behind it (needs the scene capture)
  Arc = 8,     // lightning: a jagged streak (additive)
  Bubble = 9,  // rising bubble outline (blended)
  Ember = 10,  // glowing mote that flickers (additive)
  Flame = 11,  // a tongue of fire, white-hot core to red edges, licking upwards (additive)
  Wisp = 12,   // glowing smoke: energy streaming off a shot or a charge (additive)
};

// Flags beside the shape (bits 8+).
constexpr uint32_t Attached = 1u << 8; // its position is around the anchor (see Frame), spiralling in
constexpr uint32_t Shrink = 1u << 9;   // gets smaller over its life (else sparks/glows keep, smoke grows)
constexpr uint32_t Static = 1u << 10;  // stays where it starts; its velocity only orients it (a lightning bolt's segments)

struct Particle {
  float pos[3];   // where it starts (world); attached: its offset from the anchor
  float t0;       // the clock it starts at (Frame::time)
  float vel[3];   // its start velocity (world, m/s); attached: the spiral's axis
  float life;     // seconds
  float size;     // metres across
  float drag;     // 1/s
  float gravity;  // m/s^2 down (negative rises)
  float floorZ;   // it bounces off this height (-1e9: none)
  uint32_t color; // 0xAARRGGBB, alpha its opacity
  uint32_t flags; // Shape | flags
  float stretch;  // a streak's length per m/s
  float seed;     // 0..1
};
static_assert(sizeof(Particle) == 64);

struct Frame {
  float time = 0.f;        // the clock the particles count from (seconds)
  float anchor[3] = {};    // the attached particles' centre (the muzzle)
  float anchorPulse = 0.f; // 0..1: how much the attached particles brighten (the charge)
  bool sceneDepth = false; // the scene capture is this frame's (soft edges, the haze)
};

// Records the particles into the current pass (sorted into additive, blended and haze draws by
// the caller is not needed: they are split here) with the GX state's projection and position
// matrix (set it to the view).
void draw(const Frame& frame, const Particle* particles, uint32_t count);

enum FieldKind : uint32_t {
  Rain = 0,
  Snow = 1,
  Dust = 2,
  Embers = 3,
  Splashes = 4, // where the rain lands, on the height grid
};
// A volume of falling (or floating) particles around the camera.
struct Field {
  uint32_t kind = Rain;
  uint32_t count = 0;   // instances
  float time = 0.f;
  float radius = 25.f;  // half the box's width (m)
  float height = 20.f;  // the box's height (m)
  float wind[3] = {0.f, 0.f, 0.f};
  float speed = 20.f;   // fall speed (m/s); embers rise
  float size = 0.015f;  // width (m)
  float length = 0.7f;  // streak length (m)
  uint32_t color = 0x80c8d8e8;
  // The height grid: GridSize x GridSize cells of `cell` metres from (originX, originY); each the
  // top surface's height there (nothing falls below it; the splashes sit on it), -1e9 for open.
  static constexpr uint32_t GridSize = 32;
  float originX = 0.f, originY = 0.f, cell = 2.f;
  bool useGrid = false;
  bool sceneDepth = false; // as Frame::sceneDepth
  float grid[GridSize * GridSize] = {};
};
void draw_field(const Field& field);

void shutdown();
} // namespace aurora::gfx::fxp
