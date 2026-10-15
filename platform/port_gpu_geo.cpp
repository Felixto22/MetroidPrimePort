// GPU-built geometry: grass, debris and relief (port_gpu_geo.h).
#include "port_gpu_geo.h"

#include "port_debug.h"
#include "port_env.h"
#include "port_log.h"

#include "Collision/CMaterialFilter.hpp"
#include "Collision/CRayCastResult.hpp"
#include "Kyoto/Graphics/CGraphics.hpp"
#include "MetaRender/CCubeRenderer.hpp"
#include "MetroidPrime/CGameArea.hpp"
#include "MetroidPrime/CGameCollision.hpp"
#include "MetroidPrime/CObjectList.hpp"
#include "MetroidPrime/CStateManager.hpp"
#include "MetroidPrime/Player/CPlayer.hpp"
#include "MetroidPrime/TCastTo.hpp"
#include "MetroidPrime/Weapons/CGameProjectile.hpp"

#include <aurora/geo.hpp>
#include <dolphin/gx/GXExtra.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace PortGpuGeo {
namespace {
using aurora::gfx::geo::Blade;
using aurora::gfx::geo::GrassDraw;
using aurora::gfx::geo::Rock;

constexpr float kCell = 12.f;          // grass chunk size (m)
constexpr float kBladesPerM2 = 16.f;   // at 100%
constexpr uint32_t kMaxBlades = 900000; // per area
constexpr float kFadeStart = 32.f;
constexpr float kFadeEnd = 52.f;

struct Chunk {
  float cx, cy, z;
  uint32_t first, count;
};
struct AreaGrass {
  int density = -1; // the setting it was made at
  std::vector< Chunk > chunks;
};
std::unordered_map< uint64_t, AreaGrass > sGrass;
std::vector< float > sTriangles;
std::vector< uint > sColors;

// A small deterministic random sequence.
struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed * 2654435761u + 1u) {}
  uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float Unit() { return float(Next() & 0xffffff) / float(0x1000000); }
};

uint64_t AreaKey(const CGameArea& area) { return (uint64_t(2) << 62) | uint64_t(area.GetAreaAssetId()); }

// Whether a ground colour reads as grass (green above red and blue, not too dark).
bool Grassy(uint color) {
  const float r = float((color >> 16) & 255) / 255.f;
  const float g = float((color >> 8) & 255) / 255.f;
  const float b = float(color & 255) / 255.f;
  return g > r * 1.03f && g > b * 1.12f && g > 0.1f;
}

void BuildGrass(const CGameArea& area, int density, AreaGrass& out) {
  out.chunks.clear();
  out.density = density;
  const uint64_t key = AreaKey(area);
  if (density <= 0 || gpRender == nullptr ||
      !gpRender->PortAreaTriangles(area.GetId().Value(), sTriangles, &sColors) ||
      sColors.size() * 9 != sTriangles.size()) {
    aurora::gfx::geo::set_grass(key, nullptr, 0);
    return;
  }
  const float perM2 = kBladesPerM2 * float(density) / 100.f;
  Rng rng(uint32_t(area.GetAreaAssetId()));
  std::vector< Blade > blades;
  for (size_t t = 0; t < sColors.size() && blades.size() < kMaxBlades; ++t) {
    const float* v = sTriangles.data() + t * 9;
    const float e1[3] = {v[3] - v[0], v[4] - v[1], v[5] - v[2]};
    const float e2[3] = {v[6] - v[0], v[7] - v[1], v[8] - v[2]};
    const float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    const float len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (len < 1e-6f || std::fabs(n[2]) / len < 0.82f || !Grassy(sColors[t])) {
      continue;
    }
    const float areaM2 = 0.5f * len;
    float want = areaM2 * perM2;
    int count = int(want);
    if (rng.Unit() < want - float(count)) {
      ++count;
    }
    const uint c = sColors[t];
    const float r = float((c >> 16) & 255) / 255.f;
    const float g = float((c >> 8) & 255) / 255.f;
    const float b = float(c & 255) / 255.f;
    for (int k = 0; k < count && blades.size() < kMaxBlades; ++k) {
      float a = rng.Unit(), bb = rng.Unit();
      if (a + bb > 1.f) {
        a = 1.f - a;
        bb = 1.f - bb;
      }
      Blade blade;
      for (int i = 0; i < 3; ++i) {
        blade.pos[i] = v[i] + e1[i] * a + e2[i] * bb;
      }
      blade.pos[2] -= 0.03f;
      blade.yaw = rng.Unit() * 6.2831853f;
      // Clumps: taller and lusher patches, shorter in between (a slow pattern over the ground).
      const float clump = 0.5f + 0.5f * std::sin(blade.pos[0] * 0.55f + std::sin(blade.pos[1] * 0.41f) * 2.f) *
                                      std::cos(blade.pos[1] * 0.47f + std::sin(blade.pos[0] * 0.37f) * 2.f);
      blade.height = (0.25f + 0.55f * rng.Unit() * rng.Unit() + 0.15f * rng.Unit()) * (0.7f + 0.7f * clump);
      blade.width = 0.035f + 0.035f * rng.Unit();
      // The ground's colour, greener and brighter at the tip; some blades sun-dried, yellower.
      const float j = 0.85f + 0.3f * rng.Unit();
      const float dry = rng.Unit() < 0.12f ? 0.6f + 0.4f * rng.Unit() : 0.f;
      const uint tr = uint(std::min(1.f, (r * 0.95f + dry * 0.35f) * j) * 255.f);
      const uint tg = uint(std::min(1.f, g * (1.18f - dry * 0.1f) * j) * 255.f);
      const uint tb = uint(std::min(1.f, b * (0.85f - dry * 0.3f) * j) * 255.f);
      blade.color = (tr << 16) | (tg << 8) | tb;
      blade.seed = rng.Next();
      blades.push_back(blade);
    }
  }
  sTriangles.clear();
  sTriangles.shrink_to_fit();
  sColors.clear();
  sColors.shrink_to_fit();
  // In chunks, so whole cells far away or behind are skipped.
  const auto cellOf = [](const Blade& b) {
    return std::pair< int, int >(int(std::floor(b.pos[0] / kCell)), int(std::floor(b.pos[1] / kCell)));
  };
  std::sort(blades.begin(), blades.end(), [&](const Blade& a, const Blade& b) { return cellOf(a) < cellOf(b); });
  for (uint32_t i = 0; i < blades.size();) {
    const auto cell = cellOf(blades[i]);
    uint32_t j = i;
    float z = 0.f;
    while (j < blades.size() && cellOf(blades[j]) == cell) {
      z += blades[j].pos[2];
      ++j;
    }
    out.chunks.push_back({(float(cell.first) + 0.5f) * kCell, (float(cell.second) + 0.5f) * kCell, z / float(j - i), i,
                          j - i});
    i = j;
  }
  aurora::gfx::geo::set_grass(key, blades.data(), uint32_t(blades.size()));
  PortLog::Write("port: gpu grass: area %08X, %u blades in %u chunks\n", unsigned(area.GetAreaAssetId()),
                 unsigned(blades.size()), unsigned(out.chunks.size()));
}

// What parts the grass: Samus and the shots in flight.
void Pushers(const CStateManager& mgr, GrassDraw& draw) {
  const auto add = [&](const CVector3f& p, float radius) {
    if (draw.pusherCount < GrassDraw::MaxPushers) {
      float* slot = draw.pushers[draw.pusherCount++];
      slot[0] = p.GetX(), slot[1] = p.GetY(), slot[2] = p.GetZ(), slot[3] = radius;
    }
  };
  if (const CPlayer* player = mgr.GetPlayer()) {
    const bool ball = player->GetMorphballTransitionState() == CPlayer::kMS_Morphed;
    add(player->GetTranslation(), ball ? 0.9f : 1.3f);
  }
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    if (const CGameProjectile* proj = TCastToConstPtr< CGameProjectile >(actors.GetObjectByIndex(idx))) {
      if (proj->GetActive()) {
        add(proj->GetTranslation(), 1.1f);
      }
    }
  }
}

// ---- Debris ----
struct Shard {
  float pos[3], vel[3];
  float scale;
  float axis[3], angle, spin;
  float floorZ;
  float age;
  uint color, seed;
  bool resting;
};
std::vector< Shard > sShards;
struct Tracked {
  TUniqueId uid;
  CVector3f pos, prev;
  bool seen;
};
std::vector< Tracked > sTracked;
float sLastClock = -1.f;
Rng sRng(0x5eed);
constexpr size_t kMaxShards = 220;
constexpr float kShardLife = 7.f;

void Spawn(const CStateManager& mgr, const CVector3f& at) {
  const CMaterialFilter filter = CMaterialFilter::MakeIncludeExclude(
      CMaterialList(kMT_Solid), CMaterialList(kMT_ProjectilePassthrough, kMT_SeeThrough));
  const CRayCastResult floor =
      CGameCollision::RayStaticIntersection(mgr, at + CVector3f(0.f, 0.f, 0.3f), CVector3f::Down(), 8.f, filter);
  if (!floor.IsValid()) {
    return;
  }
  const float floorZ = floor.GetPoint().GetZ();
  const int count = 6 + int(sRng.Next() % 6);
  for (int i = 0; i < count; ++i) {
    if (sShards.size() >= kMaxShards) {
      sShards.erase(sShards.begin());
    }
    Shard s{};
    s.pos[0] = at.GetX(), s.pos[1] = at.GetY(), s.pos[2] = std::max(at.GetZ(), floorZ + 0.1f);
    const float yaw = sRng.Unit() * 6.2831853f;
    const float up = 0.35f + 0.65f * sRng.Unit();
    const float speed = 2.5f + 4.f * sRng.Unit();
    const float side = std::sqrt(1.f - up * up);
    s.vel[0] = std::cos(yaw) * side * speed;
    s.vel[1] = std::sin(yaw) * side * speed;
    s.vel[2] = up * speed;
    s.scale = 0.08f + 0.17f * sRng.Unit() * sRng.Unit() + 0.04f;
    const float ax[3] = {sRng.Unit() - 0.5f, sRng.Unit() - 0.5f, sRng.Unit() - 0.5f};
    const float al = std::sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2]) + 1e-4f;
    s.axis[0] = ax[0] / al, s.axis[1] = ax[1] / al, s.axis[2] = ax[2] / al;
    s.angle = sRng.Unit() * 6.2831853f;
    s.spin = 6.f + 10.f * sRng.Unit();
    s.floorZ = floorZ;
    // Grey-brown stone, a little different each.
    const float g = 0.32f + 0.18f * sRng.Unit();
    const uint r = uint(std::min(1.f, g * 1.08f) * 255.f), gg = uint(g * 255.f), b = uint(g * 0.88f * 255.f);
    s.color = (r << 16) | (gg << 8) | b;
    s.seed = sRng.Next();
    sShards.push_back(s);
  }
}

void UpdateDebris(const CStateManager& mgr, float dt) {
  // A shot gone since the last frame that was against the world: debris where it ended.
  for (Tracked& t : sTracked) {
    t.seen = false;
  }
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CGameProjectile* proj = TCastToConstPtr< CGameProjectile >(actors.GetObjectByIndex(idx));
    if (proj == nullptr || !proj->GetActive()) {
      continue;
    }
    auto it = std::find_if(sTracked.begin(), sTracked.end(),
                           [&](const Tracked& t) { return t.uid == proj->GetUniqueId(); });
    if (it == sTracked.end()) {
      sTracked.push_back({proj->GetUniqueId(), proj->GetTranslation(), proj->GetTranslation(), true});
    } else {
      it->prev = it->pos;
      it->pos = proj->GetTranslation();
      it->seen = true;
    }
  }
  const CMaterialFilter filter = CMaterialFilter::MakeIncludeExclude(
      CMaterialList(kMT_Solid), CMaterialList(kMT_ProjectilePassthrough, kMT_SeeThrough));
  for (size_t i = sTracked.size(); i-- > 0;) {
    if (sTracked[i].seen) {
      continue;
    }
    const Tracked t = sTracked[i];
    sTracked.erase(sTracked.begin() + std::ptrdiff_t(i));
    CVector3f dir = t.pos - t.prev;
    const float len = dir.Magnitude();
    if (len < 1e-3f) {
      continue;
    }
    dir = dir * (1.f / len);
    // It ended against the static world (not an enemy, not the end of its range).
    const CRayCastResult hit = CGameCollision::RayStaticIntersection(mgr, t.pos - dir * 0.6f, dir, 2.5f, filter);
    if (hit.IsValid()) {
      Spawn(mgr, hit.GetPoint() - dir * 0.1f);
    }
  }
  for (size_t i = sShards.size(); i-- > 0;) {
    Shard& s = sShards[i];
    s.age += dt;
    if (s.age > kShardLife) {
      sShards.erase(sShards.begin() + std::ptrdiff_t(i));
      continue;
    }
    if (s.resting) {
      continue;
    }
    s.vel[2] -= 14.f * dt;
    for (int k = 0; k < 3; ++k) {
      s.pos[k] += s.vel[k] * dt;
    }
    s.angle += s.spin * dt;
    const float lift = s.scale * 0.3f;
    if (s.pos[2] < s.floorZ + lift) {
      s.pos[2] = s.floorZ + lift;
      s.vel[2] = -s.vel[2] * 0.32f;
      s.vel[0] *= 0.55f;
      s.vel[1] *= 0.55f;
      s.spin *= 0.5f;
      if (std::fabs(s.vel[2]) < 0.6f && std::hypot(s.vel[0], s.vel[1]) < 0.4f) {
        s.resting = true;
      }
    }
  }
}

void DrawDebris() {
  static std::vector< Rock > rocks;
  rocks.clear();
  for (const Shard& s : sShards) {
    Rock r;
    // Sinking into the ground over its last second and a half.
    const float sink = std::clamp((kShardLife - s.age) / 1.5f, 0.f, 1.f);
    r.pos[0] = s.pos[0], r.pos[1] = s.pos[1];
    r.pos[2] = s.pos[2] - (1.f - sink) * s.scale * 0.5f;
    r.scale = s.scale * (0.6f + 0.4f * sink);
    const float h = s.angle * 0.5f;
    const float sn = std::sin(h);
    r.rot[0] = s.axis[0] * sn, r.rot[1] = s.axis[1] * sn, r.rot[2] = s.axis[2] * sn, r.rot[3] = std::cos(h);
    r.color = s.color;
    r.seed = s.seed;
    rocks.push_back(r);
  }
  aurora::gfx::geo::draw_rocks(rocks.data(), uint32_t(rocks.size()));
}
} // namespace

void DrawWorld(const CStateManager& mgr, const CGameArea* const* areas, int areaCount, const CTransform4f& view) {
  // The relief's strength, when it changes.
  static int sRelief = -1;
  if (sRelief != PortDebug::Relief()) {
    sRelief = PortDebug::Relief();
    GXPortSetRelief(float(sRelief) / 100.f);
  }
  const float clock = CGraphics::GetSecondsMod900();
  float dt = sLastClock < 0.f ? 0.f : clock - sLastClock;
  if (dt < 0.f) {
    dt += 900.f;
  }
  dt = std::clamp(dt, 0.f, 0.1f);
  sLastClock = clock;

  gpRender->SetModelMatrix(CTransform4f::Identity());
  const int density = PortDebug::GpuGrass();
  if (density > 0) {
    GrassDraw draw;
    draw.time = clock;
    draw.fadeStart = kFadeStart;
    draw.fadeEnd = kFadeEnd;
    Pushers(mgr, draw);
    const CVector3f eye = view.GetTranslation();
    const CVector3f fwd = view.GetForward();
    bool built = false;
    for (int i = 0; i < areaCount; ++i) {
      if (areas[i] == nullptr) {
        continue;
      }
      const uint64_t key = AreaKey(*areas[i]);
      AreaGrass& grass = sGrass[key];
      if (grass.density != density) {
        // One area's blades a frame (the walk over its triangles takes a moment).
        if (built) {
          continue;
        }
        BuildGrass(*areas[i], density, grass);
        built = true;
      }
      draw.key = key;
      for (const Chunk& c : grass.chunks) {
        const float dx = c.cx - eye.GetX(), dy = c.cy - eye.GetY(), dz = c.z - eye.GetZ();
        const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (d > kFadeEnd + kCell * 0.75f) {
          continue;
        }
        if (d > kCell && (dx * fwd.GetX() + dy * fwd.GetY() + dz * fwd.GetZ()) < -0.35f * d) {
          continue; // behind the camera
        }
        draw.first = c.first;
        draw.count = c.count;
        aurora::gfx::geo::draw_grass(draw);
      }
    }
  }
  if (PortDebug::Debris()) {
    UpdateDebris(mgr, dt);
    if (!sShards.empty()) {
      DrawDebris();
    }
  } else {
    sShards.clear();
    sTracked.clear();
  }
}

void Reset() {
  for (const auto& [key, grass] : sGrass) {
    aurora::gfx::geo::set_grass(key, nullptr, 0);
  }
  sGrass.clear();
  sShards.clear();
  sTracked.clear();
}
} // namespace PortGpuGeo
