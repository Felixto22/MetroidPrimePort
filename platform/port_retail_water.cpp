// Enhanced retail water (port_retail_water.h): a generated surface for aurora's water shader.
#include "port_retail_water.h"

#include "port_debug.h"
#include "port_room_env.h"

#include "Kyoto/Graphics/CColor.hpp"
#include "Kyoto/Graphics/CGraphics.hpp"
#include "Kyoto/Math/CTransform4f.hpp"
#include "Kyoto/Math/CVector3f.hpp"
#include "MetaRender/CCubeRenderer.hpp"
#include "MetroidPrime/CFluidPlaneManager.hpp"
#include "MetroidPrime/CRippleManager.hpp"
#include "MetroidPrime/CStateManager.hpp"
#include "MetroidPrime/Player/CPlayer.hpp"
#include "MetroidPrime/Player/CPlayerState.hpp"

#include "aurora/water.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace PortRetailWater {
namespace {

constexpr float kPi = 3.14159265358979f;
constexpr int kNormalSize = 256;
// World metres one repeat of the ripple map covers.
constexpr float kRippleTile = 3.f;
// How far in from the box's edge the waves fade out, so they don't lift off the walls.
constexpr float kEdgeFade = 1.5f;
constexpr int kMaxCells = 256;

struct Mesh {
  float key[5] = {};
  std::vector< aurora::gfx::water::Vertex > verts;
  std::vector< uint32_t > indices;
};

std::unordered_map< uint32_t, Mesh >& Meshes() {
  static auto* meshes = new std::unordered_map< uint32_t, Mesh >();
  return *meshes;
}

// A tileable ripple height field: directional waves with whole numbers of periods across the
// tile, plus two octaves of wrapped value noise.
float Hash(int x, int y) {
  uint32_t h = uint32_t(x) * 374761393u + uint32_t(y) * 668265263u;
  h = (h ^ (h >> 13)) * 1274126177u;
  return float((h ^ (h >> 16)) & 0xFFFF) / 65535.f;
}

float ValueNoise(float x, float y, int period) {
  const int xi = int(std::floor(x)), yi = int(std::floor(y));
  const float fx = x - float(xi), fy = y - float(yi);
  const auto at = [period](int a, int b) { return Hash(((a % period) + period) % period, ((b % period) + period) % period); };
  const float ux = fx * fx * (3.f - 2.f * fx), uy = fy * fy * (3.f - 2.f * fy);
  const float a = at(xi, yi) + (at(xi + 1, yi) - at(xi, yi)) * ux;
  const float b = at(xi, yi + 1) + (at(xi + 1, yi + 1) - at(xi, yi + 1)) * ux;
  return a + (b - a) * uy;
}

float Height(float u, float v) {
  static const struct {
    int kx, ky;
    float amp, phase;
  } kWaves[] = {{3, 1, 1.f, 0.f}, {-2, 3, 0.8f, 1.3f}, {5, -2, 0.5f, 2.1f}, {1, 6, 0.4f, 0.7f},
                {-7, -3, 0.25f, 4.f}, {9, 4, 0.18f, 5.2f}};
  float h = 0.f;
  for (const auto& w : kWaves) {
    h += w.amp * std::sin(2.f * kPi * (w.kx * u + w.ky * v) + w.phase);
  }
  h += 0.9f * ValueNoise(u * 8.f, v * 8.f, 8) + 0.45f * ValueNoise(u * 16.f, v * 16.f, 16);
  return h;
}

// The ripple map as GX RGBA8 (4x4 tiles: 16 AR pairs then 16 GB pairs), xy the normal's
// tangent components as the shader reads them (texel / 255 * 1.9921875 - 1).
const GXTexObj* RippleMap() {
  static GXTexObj sObj;
  static std::vector< uint8_t > sData;
  if (!sData.empty()) {
    return &sObj;
  }
  const int n = kNormalSize;
  std::vector< float > h(size_t(n) * n);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      h[size_t(y) * n + x] = Height(float(x) / n, float(y) / n);
    }
  }
  const float strength = 0.9f;
  const auto at = [&](int x, int y) { return h[size_t((y + n) % n) * n + size_t((x + n) % n)]; };
  const auto enc = [](float c) { return uint8_t(std::clamp((c + 1.f) / 1.9921875f * 255.f + 0.5f, 0.f, 255.f)); };
  sData.assign(size_t(n) * n * 4, 0);
  for (int y = 0; y < n; ++y) {
    for (int x = 0; x < n; ++x) {
      const float dx = (at(x + 1, y) - at(x - 1, y)) * strength;
      const float dy = (at(x, y + 1) - at(x, y - 1)) * strength;
      const float len = std::sqrt(dx * dx + dy * dy + 1.f);
      const float nx = -dx / len, ny = -dy / len, nz = 1.f / len;
      const size_t tile = size_t(y / 4) * (n / 4) + size_t(x / 4);
      const size_t i = size_t((y % 4) * 4 + (x % 4));
      uint8_t* const block = sData.data() + tile * 64;
      block[i * 2] = 255;          // A
      block[i * 2 + 1] = enc(nx);  // R
      block[32 + i * 2] = enc(ny); // G
      block[32 + i * 2 + 1] = uint8_t(std::clamp(nz * 255.f, 0.f, 255.f)); // B
    }
  }
  GXInitTexObj(&sObj, sData.data(), n, n, GX_TF_RGBA8, GX_REPEAT, GX_REPEAT, GX_FALSE);
  GXInitTexObjLOD(&sObj, GX_LINEAR, GX_LINEAR, 0.f, 0.f, 0.f, GX_FALSE, GX_FALSE, GX_ANISO_4);
  return &sObj;
}

// A grid over the box's top in Remastered's mesh space (y up): mesh (x, y, z) is world
// (x, -z, y) about the box's centre at the surface.
const Mesh& BuildMesh(uint32_t uid, float minX, float minY, float maxX, float maxY) {
  Mesh& m = Meshes()[uid];
  const float key[5] = {minX, minY, maxX, maxY, 1.f};
  if (!m.verts.empty() && std::memcmp(m.key, key, sizeof(key)) == 0) {
    return m;
  }
  std::memcpy(m.key, key, sizeof(key));
  m.verts.clear();
  m.indices.clear();
  const float sx = std::max(maxX - minX, 0.1f), sy = std::max(maxY - minY, 0.1f);
  // Fine enough for the rings' swell (about 1.6 m from crest to crest) to show in the surface.
  const float cell = std::clamp(std::max(sx, sy) / 160.f, 0.25f, 3.f);
  const int nx = std::clamp(int(std::ceil(sx / cell)), 1, kMaxCells);
  const int ny = std::clamp(int(std::ceil(sy / cell)), 1, kMaxCells);
  const float cx = 0.5f * (minX + maxX), cy = 0.5f * (minY + maxY);
  m.verts.reserve(size_t(nx + 1) * (ny + 1));
  for (int j = 0; j <= ny; ++j) {
    for (int i = 0; i <= nx; ++i) {
      const float X = minX + sx * float(i) / float(nx);
      const float Y = minY + sy * float(j) / float(ny);
      aurora::gfx::water::Vertex v{};
      v.pos[0] = X - cx;
      v.pos[1] = 0.f;
      v.pos[2] = -(Y - cy);
      v.uv[0] = X / kRippleTile;
      v.uv[1] = Y / kRippleTile;
      const float edge = std::min(std::min(X - minX, maxX - X), std::min(Y - minY, maxY - Y));
      const uint8_t wave = uint8_t(std::clamp(edge / kEdgeFade, 0.f, 1.f) * 255.f);
      v.color[0] = v.color[1] = v.color[2] = wave;
      v.color[3] = 255;
      m.verts.push_back(v);
    }
  }
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const uint32_t a = uint32_t(j * (nx + 1) + i), b = a + 1, c = a + uint32_t(nx + 1), d = c + 1;
      m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
    }
  }
  return m;
}

// Port ripples on top of the game's: Samus's wake as she wades or swims, and the splash where
// she enters or leaves the surface. Per water body: world x, y, the clock at birth, strength.
struct WakeRipple {
  float x, y, born, strength;
};
struct Wake {
  std::vector< WakeRipple > ripples;
  float lastX = 0.f, lastY = 0.f;
  bool have = false;
  bool wasIn = false;
};
std::unordered_map< uint32_t, Wake >& Wakes() {
  static auto* wakes = new std::unordered_map< uint32_t, Wake >();
  return *wakes;
}
constexpr float kRippleLife = 2.6f;  // seconds a ring keeps spreading
constexpr float kWakeStep = 0.45f;   // metres between wake rings

float Age(float clock, float born) {
  float a = clock - born;
  return a < 0.f ? a + 900.f : a;
}

void UpdateWake(const CStateManager& mgr, uint32_t uid, const CVector3f& boxMin, const CVector3f& boxMax,
                float surfaceZ, float clock) {
  Wake& w = Wakes()[uid];
  for (size_t i = w.ripples.size(); i-- > 0;) {
    if (Age(clock, w.ripples[i].born) > kRippleLife) {
      w.ripples.erase(w.ripples.begin() + std::ptrdiff_t(i));
    }
  }
  const CPlayer* player = mgr.GetPlayer();
  if (player == nullptr) {
    return;
  }
  const CVector3f feet = player->GetTranslation();
  const bool ball = player->GetMorphballTransitionState() == CPlayer::kMS_Morphed;
  const float x = feet.GetX(), y = feet.GetY();
  const bool overBox = x >= boxMin.GetX() && x <= boxMax.GetX() && y >= boxMin.GetY() && y <= boxMax.GetY();
  // In the water and near enough the top for it to show: from wading to a little under.
  const float depth = surfaceZ - feet.GetZ();
  const bool in = overBox && depth > -0.1f && depth < (ball ? 1.5f : 3.f);
  const auto add = [&](float strength) {
    if (w.ripples.size() >= 12) {
      w.ripples.erase(w.ripples.begin());
    }
    w.ripples.push_back({x, y, clock, strength});
  };
  if (in != w.wasIn && overBox && depth > -1.f && depth < 3.5f) {
    add(1.f); // the splash in or out
    w.lastX = x;
    w.lastY = y;
    w.have = true;
  }
  w.wasIn = in;
  if (!in) {
    w.have = false;
    return;
  }
  if (!w.have) {
    w.lastX = x;
    w.lastY = y;
    w.have = true;
    return;
  }
  const float moved = std::hypot(x - w.lastX, y - w.lastY);
  if (moved > 6.f) {
    w.lastX = x;
    w.lastY = y;
  } else if (moved > kWakeStep) {
    // Stronger the nearer the top she moves.
    add(std::clamp(1.1f - depth * 0.25f, 0.45f, 1.f));
    w.lastX = x;
    w.lastY = y;
  }
}

} // namespace

bool Draw(const CStateManager& mgr, uint32_t uid, const CVector3f& boxMin, const CVector3f& boxMax, float surfaceZ,
          int fluidType, const CColor& insideColor) {
  if (!PortDebug::EnhancedWater() || (fluidType != 0 && fluidType != 1)) {
    return false;
  }
  const CPlayerState::EPlayerVisor visor = mgr.GetPlayerState()->GetActiveVisor(mgr);
  if (visor == CPlayerState::kPV_XRay || visor == CPlayerState::kPV_Thermal) {
    return false;
  }
  const bool poison = fluidType == 1;
  const Mesh& mesh = BuildMesh(uid, boxMin.GetX(), boxMin.GetY(), boxMax.GetX(), boxMax.GetY());
  const float cx = 0.5f * (boxMin.GetX() + boxMax.GetX());
  const float cy = 0.5f * (boxMin.GetY() + boxMax.GetY());
  // Mesh (x, y, z) -> world (x, -z, y), at the surface.
  const CTransform4f xf(1.f, 0.f, 0.f, cx, 0.f, 0.f, -1.f, cy, 0.f, 1.f, 0.f, surfaceZ);

  aurora::gfx::water::DrawDesc d;
  d.normalMap = RippleMap();
  d.cull = aurora::gfx::water::Cull::None;
  const CTransform4f& view = CGraphics::GetViewMatrix();
  d.bottom = view.GetTranslation().GetZ() < surfaceZ;
  d.screenReflections = true;
  d.foam = poison ? 0.35f : 0.5f;

  // The rings on this surface, newest first: the game's ripples (shots landing, enemies and
  // Samus touching it) and Samus's own wake.
  {
    const float now = CGraphics::GetSecondsMod900();
    UpdateWake(mgr, uid, boxMin, boxMax, surfaceZ, now);
    struct Ring {
      float x, y, t, s;
    };
    Ring rings[64];
    int count = 0;
    if (const CFluidPlaneManager* fluids = mgr.GetFluidPlaneManager()) {
      const rstl::vector< CRipple >& ripples = fluids->GetRippleManager().GetRipples();
      for (int i = ripples.size() - 1; i >= 0 && count < 48; --i) {
        const CRipple& r = ripples[i];
        const float t = r.GetTime();
        const float life = std::max(r.GetTimeFalloff(), kRippleLife);
        const CVector3f& c = r.GetCenter();
        if (t < 0.f || t >= life || c.GetX() < boxMin.GetX() - 1.f || c.GetX() > boxMax.GetX() + 1.f ||
            c.GetY() < boxMin.GetY() - 1.f || c.GetY() > boxMax.GetY() + 1.f ||
            std::fabs(c.GetZ() - surfaceZ) > 3.f) {
          continue;
        }
        // CRipple's amplitude runs 0.1..0.25 with the splash's intensity: every one shows, the
        // strongest the most. Fades over its life.
        const float s = std::clamp(0.45f + r.GetLookupAmplitude() * 255.f / 0.25f * 0.55f, 0.f, 1.f);
        rings[count++] = {c.GetX(), c.GetY(), t, s * (1.f - t / life)};
      }
    }
    for (const WakeRipple& w : Wakes()[uid].ripples) {
      if (count < 64) {
        const float t = Age(now, w.born);
        rings[count++] = {w.x, w.y, t, w.strength * (1.f - t / kRippleLife)};
      }
    }
    std::sort(rings, rings + count, [](const Ring& a, const Ring& b) { return a.t < b.t; });
    d.rippleTile = kRippleTile;
    for (int i = 0; i < count && d.rippleCount < aurora::gfx::water::DrawDesc::MaxRipples; ++i) {
      float(&slot)[4] = d.ripples[d.rippleCount++];
      slot[0] = rings[i].x;
      slot[1] = rings[i].y;
      slot[2] = rings[i].t;
      slot[3] = rings[i].s;
    }
  }

  const float clock = CGraphics::GetSecondsMod900();
  // The underwater colour, as the colour light keeps through the water's depth.
  float deep[3] = {insideColor.GetRed(), insideColor.GetGreen(), insideColor.GetBlue()};
  const float top = std::max(std::max(deep[0], deep[1]), deep[2]);
  for (float& c : deep) {
    c = top > 0.01f ? std::clamp(c / top * 0.75f, 0.06f, 0.9f) : 0.4f;
  }
  if (top <= 0.01f) {
    deep[0] = poison ? 0.45f : 0.2f;
    deep[1] = poison ? 0.6f : 0.5f;
    deep[2] = poison ? 0.2f : 0.55f;
  }
  float(&S)[8][4] = d.surface;
  S[0][0] = clock;
  S[0][1] = 0.92f;           // smoothness
  S[0][2] = 1.f / 0.25f;     // the soft edge where the surface meets the ground (m)
  S[0][3] = 1.f;             // the ripple map's repeat (uv is already in tiles)
  S[1][0] = 0.7f;            // ripple drift direction
  S[1][1] = 0.4f;
  S[1][2] = poison ? 0.05f : 0.08f; // ripple drift speed (tiles a second)
  S[1][3] = 0.92f;
  S[2][0] = deep[0] * 0.35f; // the surface's own colour in the light
  S[2][1] = deep[1] * 0.35f;
  S[2][2] = deep[2] * 0.35f;
  S[2][3] = poison ? 0.12f : 0.22f; // reflectance facing the camera: a still pool mirrors the room
  S[3][0] = 3.5f;            // Fresnel power
  S[3][1] = poison ? 0.75f : 1.f; // reflection strength
  S[3][2] = 1.f;
  S[3][3] = 1.f;
  S[4][0] = 0.25f;           // the vertex normal's finite difference (m)
  S[4][1] = 0.f;
  S[4][2] = 0.12f;           // the ripple cross-fade's speed
  S[4][3] = 0.f;
  S[6][0] = S[6][1] = S[6][2] = 1.f;
  S[6][3] = 0.1f;            // reflectance from below
  S[7][0] = deep[0];
  S[7][1] = deep[1];
  S[7][2] = deep[2];
  S[7][3] = 1.f / (poison ? 2.5f : 6.f); // depth to the full underwater colour (m)

  // Two gentle waves: angle (degrees), sideways and vertical amplitude, length, speed.
  static const float kWaves[2][5] = {{25.f, 0.05f, 0.04f, 7.f, 1.2f}, {115.f, 0.03f, 0.025f, 4.3f, 0.9f}};
  constexpr float kDegrees = 0.017453292f;
  d.waves[0][0] = std::cos(kWaves[1][0] * kDegrees);
  d.waves[0][1] = std::sin(kWaves[1][0] * kDegrees);
  d.waves[0][2] = std::cos(kWaves[0][0] * kDegrees);
  d.waves[0][3] = std::sin(kWaves[0][0] * kDegrees);
  for (int i = 0; i < 2; ++i) {
    const float number = 2.f * kPi / kWaves[i][3];
    d.waves[1 + i][0] = kWaves[i][1];
    d.waves[1 + i][1] = kWaves[i][2];
    d.waves[1 + i][2] = kWaves[i][4] * number * clock;
    d.waves[1 + i][3] = number;
  }

  // No probe or volume: an even light, and the reflection marched through the scene.
  for (int i = 0; i < 3; ++i) {
    d.fallbackMean[i] = 1.f;
    d.volumeScale[i] = 0.6f;
  }
  d.fallbackMean[3] = 1.f;
  d.volumeScale[3] = 1.f;
  if (!PortRoomEnv::Tone(d.tone)) {
    // The retail EFB is display-referred: a straight line through the curve and its inverse.
    std::memset(d.tone, 0, sizeof(d.tone));
    d.tone[1][0] = 1.f;
    d.tone[1][3] = 1e9f;
    d.tone[2][3] = 1e9f;
  }
  d.tone[0][3] = 1.f;
  const CGraphics::CProjectionState& proj = CGraphics::GetProjectionState();
  d.zNear = proj.GetNear();
  d.zFar = proj.GetFar();
  d.zMin = 0.125f;
  d.zMax = 1.f;

  gpRender->SetModelMatrix(xf);
  aurora::gfx::water::draw(d, mesh.verts.data(), uint32_t(mesh.verts.size()), mesh.indices.data(),
                           uint32_t(mesh.indices.size()));
  return true;
}

void Reset() {
  Meshes().clear();
  Wakes().clear();
}
} // namespace PortRetailWater
