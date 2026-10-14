// Interactive foliage (port_foliage.h): plants that bend, get wet in the rain and burn.
#include "port_foliage.h"

#include "port_debug.h"

#include "Kyoto/Audio/CSfxManager.hpp"
#include "Kyoto/Basics/CBasics.hpp"
#include "Kyoto/CSimplePool.hpp"
#include "Kyoto/Graphics/CCubeModel.hpp"
#include "Kyoto/Graphics/CCubeSurface.hpp"
#include "Kyoto/Graphics/CGraphics.hpp"
#include "Kyoto/Math/CAABox.hpp"
#include "Kyoto/Math/CRelAngle.hpp"
#include "Kyoto/Math/CTransform4f.hpp"
#include "Kyoto/Math/CVector3f.hpp"
#include "Kyoto/Particles/CElementGen.hpp"
#include "Kyoto/Particles/CGenDescription.hpp"
#include "MetroidPrime/CEnvFxManager.hpp"
#include "MetroidPrime/CObjectList.hpp"
#include "MetroidPrime/CRainSplashGenerator.hpp"
#include "MetroidPrime/CStateManager.hpp"
#include "MetroidPrime/CWorld.hpp"
#include "MetroidPrime/Player/CPlayer.hpp"
#include "MetroidPrime/SFX/Misc.h"
#include "MetroidPrime/TCastTo.hpp"
#include "MetroidPrime/Weapons/CGameProjectile.hpp"

#include <dolphin/gx/GXExtra.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <memory>
#include <vector>

namespace PortFoliage {
namespace {
// Plant textures (TXTR ids), picked by eye from the disc's cut-out textures: ferns, leaf sprays,
// bushes, grass fringes, hanging vines, roots and moss. Wall and floor decals (moss patches,
// debris) stay off the list: bending a decal off its wall would show the gap behind it.
const unsigned int kFoliageTextures[] = {
    0x0AF0DFD0, 0x10C5E944, 0x12ACA553, 0x1AB3E109, 0x1B7FFC3F, 0x1D19EFF8,
    0x22FC069C, 0x244C952A, 0x2520FD93, 0x280B89C9, 0x2A9AF9B0, 0x2AB88EC6,
    0x2DCF3C32, 0x3335C23F, 0x3F578E77, 0x426A772D, 0x45D5C2E1, 0x46D630EF,
    0x4CF9A85C, 0x50CFA33D, 0x526C63ED, 0x56954560, 0x5695CA37, 0x59EB930D,
    0x6114F166, 0x64A444DB, 0x6B211DB9, 0x72852E14, 0x779CEE5C, 0x81EA28CF,
    0x8DFB0DC8, 0x90DE0FF1, 0x91C5E27D, 0x96939AD5, 0x9DB417D9, 0xA670D1C9,
    0xA9374EE7, 0xA9E79300, 0xB2D2A624, 0xB3936FB4, 0xB782B593, 0xBD05E899,
    0xBDA1ACFB, 0xBE16E803, 0xC694D876, 0xC7D58962, 0xD9A89D7B, 0xE0548F3C,
    0xE2DD22D8, 0xF0910EF1, 0xF5963EE5, 0xFE0E9414, 0xFF2252D5, 0xFF9FA3CD,
};

// ---------------------------------------------------------------------------------------------
// The plants drawn lately: each foliage surface (of an area, or of an actor's model where it
// stands), its world bounds and some of its vertices. Rain splashes land on those vertices,
// shots set the bounds alight.
struct SurfKey {
  const void* surface;
  int x, y, z; // the model's translation in cm: a model drawn in several places is several plants
  bool operator<(const SurfKey& o) const {
    if (surface != o.surface) {
      return surface < o.surface;
    }
    if (x != o.x) {
      return x < o.x;
    }
    if (y != o.y) {
      return y < o.y;
    }
    return z < o.z;
  }
};

struct Plant {
  CVector3f min, max;
  std::vector< CVector3f > points;
  uint lastSeen = 0;
  float charAmount = 0.f; // 0..1, kept once burnt
  bool burning = false;
};

std::map< SurfKey, Plant >& Plants() {
  static auto* plants = new std::map< SurfKey, Plant >();
  return *plants;
}
uint sFrame = 0;
const uint kForgetFrames = 60 * 20;
const int kPointsPerPlant = 32;

CVector3f Center(const Plant& p) { return (p.min + p.max) * 0.5f; }

// ---------------------------------------------------------------------------------------------
// Pushers (Samus, shots) for the bend.
struct Pusher {
  CVector3f pos = CVector3f::Zero(); // feet
  float radius = 0.f;
  float height = 0.f;
  float strength = 0.f;
  float age = 0.f; // seconds since it let go
};

const int kMaxReleased = 24;
Pusher sReleased[kMaxReleased];
int sReleasedCount = 0;

struct Shot {
  TUniqueId uid = kInvalidUniqueId;
  CVector3f last = CVector3f::Zero();  // where it was last frame (its true position)
  CVector3f trail = CVector3f::Zero(); // feet of its last trail pusher
  bool seen = false;
  bool burns = false;
};
const int kMaxShots = 16;
Shot sShots[kMaxShots];
int sShotCount = 0;

bool sHavePlayer = false;
CVector3f sPlayerTrail = CVector3f::Zero();
float sLastClock = -1.f;
float sSeconds = 0.f;
bool sSentEmpty = false;

const float kReleaseLife = 1.3f;
const float kPlayerTrailStep = 0.8f;
const float kShotTrailStep = 1.0f;

void Release(const CVector3f& pos, float radius, float height, float strength) {
  if (sReleasedCount == kMaxReleased) {
    int oldest = 0;
    for (int i = 1; i < sReleasedCount; ++i) {
      if (sReleased[i].age > sReleased[oldest].age) {
        oldest = i;
      }
    }
    sReleased[oldest] = sReleased[--sReleasedCount];
  }
  Pusher& p = sReleased[sReleasedCount++];
  p.pos = pos;
  p.radius = radius;
  p.height = height;
  p.strength = strength;
  p.age = 1e-3f;
}

float Advance() {
  const float clock = CGraphics::GetSecondsMod900();
  float dt = sLastClock < 0.f ? 0.f : clock - sLastClock;
  if (dt < 0.f) {
    dt += 900.f;
  }
  sLastClock = clock;
  dt = std::clamp(dt, 0.f, 0.1f);
  sSeconds += dt;
  if (sSeconds > 3600.f) {
    sSeconds -= 3600.f;
  }
  return dt;
}

// ---------------------------------------------------------------------------------------------
// Fire: a shot into a plant lights it. Flames spread over it from where the shot went in while it
// burns; then it smokes and is left charred. Rain puts it out sooner, with more smoke.
TToken< CGenDescription >* sFlameDesc = nullptr;
TToken< CGenDescription >* sSmokeDesc = nullptr;

bool LoadEffects() {
  static bool tried = false;
  if (!tried) {
    tried = true;
    sFlameDesc = new TToken< CGenDescription >(gpSimplePool->GetObj("Effect_OnFire"));
    sFlameDesc->Lock();
    sSmokeDesc = new TToken< CGenDescription >(gpSimplePool->GetObj("NFTMainSmoke"));
    sSmokeDesc->Lock();
  }
  return sFlameDesc->IsLoaded() && sSmokeDesc->IsLoaded();
}

struct Fire {
  SurfKey key{};
  CVector3f origin = CVector3f::Zero();
  float maxRadius = 1.f;
  float t = 0.f;
  float burnTime = 10.f;  // flames
  float smokeTime = 6.f;  // smoke after
  float spawnTimer = 0.f;
  bool out = false;
  std::vector< std::unique_ptr< CElementGen > > flames;
  std::vector< std::unique_ptr< CElementGen > > smoke;
  CSfxHandle sfx;

  float Radius() const { return std::min(maxRadius, 0.35f + t * 0.45f); }
  float Glow() const {
    if (t >= burnTime) {
      return std::max(0.f, 0.6f - (t - burnTime) * 0.3f); // embers dying out
    }
    return std::min(1.f, t * 2.f);
  }
};
std::vector< std::unique_ptr< Fire > >& Fires() {
  static auto* fires = new std::vector< std::unique_ptr< Fire > >();
  return *fires;
}
const size_t kMaxFires = 6;

bool Raining(const CStateManager& mgr, float* magnitude, bool* exposed) {
  const CEnvFxManager* env = mgr.GetEnvFxManager();
  const CWorld* world = mgr.GetWorld();
  if (env == nullptr || world == nullptr || world->GetNeededEnvFx() != kEFX_Rain) {
    return false;
  }
  *magnitude = env->GetRainMagnitude();
  *exposed = env->IsSplashActive();
  return *magnitude > 0.02f;
}

// Segment p0-p1 against a box grown by `pad`: the entry point.
bool SegmentHitsBox(const CVector3f& p0, const CVector3f& p1, const CVector3f& mn, const CVector3f& mx, float pad,
                    CVector3f* hit) {
  float t0 = 0.f, t1 = 1.f;
  const float a[3] = {p0.GetX(), p0.GetY(), p0.GetZ()};
  const float b[3] = {p1.GetX(), p1.GetY(), p1.GetZ()};
  const float lo[3] = {mn.GetX() - pad, mn.GetY() - pad, mn.GetZ() - pad};
  const float hi[3] = {mx.GetX() + pad, mx.GetY() + pad, mx.GetZ() + pad};
  for (int i = 0; i < 3; ++i) {
    const float d = b[i] - a[i];
    if (std::fabs(d) < 1e-6f) {
      if (a[i] < lo[i] || a[i] > hi[i]) {
        return false;
      }
      continue;
    }
    float e0 = (lo[i] - a[i]) / d, e1 = (hi[i] - a[i]) / d;
    if (e0 > e1) {
      std::swap(e0, e1);
    }
    t0 = std::max(t0, e0);
    t1 = std::min(t1, e1);
    if (t0 > t1) {
      return false;
    }
  }
  *hit = p0 + (p1 - p0) * t0;
  return true;
}

void Ignite(const SurfKey& key, Plant& plant, const CVector3f& at, bool raining) {
  if (plant.burning || plant.charAmount > 0.8f || Fires().size() >= kMaxFires || !LoadEffects()) {
    return;
  }
  plant.burning = true;
  auto fire = std::make_unique< Fire >();
  fire->key = key;
  // Start at the plant's vertex nearest where the shot went in.
  CVector3f origin = at;
  float best = 1e9f;
  for (const CVector3f& p : plant.points) {
    const float d = (p - at).MagSquared();
    if (d < best) {
      best = d;
      origin = p;
    }
  }
  fire->origin = origin;
  fire->maxRadius = std::clamp((plant.max - plant.min).Magnitude() * 0.6f, 0.6f, 3.f);
  fire->burnTime = raining ? 3.5f : 11.f;
  fire->smokeTime = raining ? 9.f : 6.f;
  fire->sfx = CSfxManager::AddEmitter(SFXeff_x_fire_lp_01, origin, CVector3f::Zero(), true, true);
  Fires().push_back(std::move(fire));
}

CElementGen* MakeGen(TToken< CGenDescription >& desc, const CVector3f& pos, bool up) {
  CElementGen* gen = rs_new CElementGen(desc);
  if (up) {
    // The smoke streams along +y like the flamethrower it is made for: turn that up.
    gen->SetOrientation(CTransform4f::RotateX(CRelAngle::FromDegrees(90.f)));
  }
  gen->SetGlobalTranslation(pos);
  return gen;
}

void UpdateFires(const CStateManager& mgr, float dt, bool raining) {
  std::vector< std::unique_ptr< Fire > >& fires = Fires();
  for (size_t f = 0; f < fires.size();) {
    Fire& fire = *fires[f];
    auto it = Plants().find(fire.key);
    Plant* plant = it != Plants().end() ? &it->second : nullptr;
    fire.t += dt;
    if (raining && fire.burnTime > fire.t + 3.5f) {
      fire.burnTime = fire.t + 3.5f; // the rain came: it goes out soon
      fire.smokeTime = 9.f;
    }
    const bool burning = fire.t < fire.burnTime;
    if (plant != nullptr) {
      plant->charAmount = std::max(plant->charAmount, std::min(1.f, fire.t / std::max(fire.burnTime, 1.f) * 1.3f));
    }
    // Flames: new bursts at the plant's vertices within the spread, every quarter second.
    fire.spawnTimer -= dt;
    if (burning && fire.spawnTimer <= 0.f && plant != nullptr && !plant->points.empty()) {
      fire.spawnTimer = 0.25f;
      const float r = fire.Radius();
      for (int tries = 0; tries < 6; ++tries) {
        const CVector3f& p = plant->points[size_t(mgr.Random()->Range(0, int(plant->points.size()) - 1))];
        if ((p - fire.origin).Magnitude() <= r) {
          if (fire.flames.size() < 10) {
            fire.flames.emplace_back(MakeGen(*sFlameDesc, p, false));
          }
          break;
        }
      }
    }
    // Smoke: a column while it burns (thin), thicker when the flames go out.
    const float smokeEnd = fire.burnTime + fire.smokeTime;
    if (fire.t < smokeEnd && plant != nullptr) {
      const size_t want = burning ? 1 : (raining ? 3 : 2);
      while (fire.smoke.size() < want) {
        const CVector3f& p = plant->points.empty() ? fire.origin
                                                    : plant->points[fire.smoke.size() * 7 % plant->points.size()];
        fire.smoke.emplace_back(MakeGen(*sSmokeDesc, p, true));
      }
    } else {
      for (auto& g : fire.smoke) {
        g->SetParticleEmission(false);
      }
    }
    for (auto& g : fire.flames) {
      g->Update(dt);
    }
    for (auto& g : fire.smoke) {
      g->Update(dt);
    }
    fire.flames.erase(std::remove_if(fire.flames.begin(), fire.flames.end(),
                                     [](const std::unique_ptr< CElementGen >& g) { return g->IsSystemDeletable(); }),
                      fire.flames.end());
    if (fire.sfx) {
      if (burning) {
        CSfxManager::UpdateEmitter(fire.sfx, fire.origin, CVector3f::Zero(), 0x7f);
      } else {
        CSfxManager::RemoveEmitter(fire.sfx);
        fire.sfx.Clear();
      }
    }
    const bool smokeDone = fire.t >= smokeEnd &&
                           std::all_of(fire.smoke.begin(), fire.smoke.end(),
                                       [](const std::unique_ptr< CElementGen >& g) { return g->IsSystemDeletable(); });
    if (fire.t >= smokeEnd + 8.f || (smokeDone && fire.flames.empty())) {
      if (fire.sfx) {
        CSfxManager::RemoveEmitter(fire.sfx);
      }
      if (plant != nullptr) {
        plant->burning = false;
        plant->charAmount = 1.f;
      }
      fires.erase(fires.begin() + std::ptrdiff_t(f));
      continue;
    }
    ++f;
  }
}

void ClearFires() {
  for (auto& f : Fires()) {
    if (f->sfx) {
      CSfxManager::RemoveEmitter(f->sfx);
    }
  }
  Fires().clear();
}

// ---------------------------------------------------------------------------------------------
// Rain on the leaves: the gun's line splashes, at the plants' vertices around the camera.
CRainSplashGenerator* sSplashes = nullptr;
bool sSplashing = false;

void UpdateSplashes(const CStateManager& mgr, float dt, const CVector3f& eye) {
  if (sSplashes == nullptr) {
    sSplashes = rs_new CRainSplashGenerator(CVector3f(1.f, 1.f, 1.f), 160, 6, 0.f, 0.6f);
  }
  static std::vector< CVector3f > points, normals;
  points.clear();
  for (const auto& [key, plant] : Plants()) {
    if (plant.lastSeen + 2 < sFrame || (Center(plant) - eye).Magnitude() > 16.f) {
      continue;
    }
    for (const CVector3f& p : plant.points) {
      points.push_back(p);
      if (points.size() >= 1024) {
        break;
      }
    }
  }
  sSplashes->Update(dt, const_cast< CStateManager& >(mgr));
  sSplashing = !points.empty() && sSplashes->IsRaining();
  if (sSplashing) {
    normals.assign(points.size(), CVector3f(0.f, 0.f, 1.f));
    sSplashes->GeneratePoints(points.data(), normals.data(), int(points.size()));
  }
}
} // namespace

bool IsFoliageTexture(unsigned int assetId) {
  return std::binary_search(std::begin(kFoliageTextures), std::end(kFoliageTextures), assetId);
}

bool Active() { return PortDebug::InteractiveFoliage() || PortDebug::RainDetails() || PortDebug::BurnableFoliage(); }

void NoteSurface(const CCubeSurface& surface, const CCubeModel& model) {
  const CTransform4f& xf = CGraphics::GetModelMatrix();
  const CVector3f t = xf.GetTranslation();
  const SurfKey key{surface.GetDisplayList(), int(std::lround(t.GetX() * 100.f)), int(std::lround(t.GetY() * 100.f)),
                    int(std::lround(t.GetZ() * 100.f))};
  auto [it, fresh] = Plants().try_emplace(key);
  Plant& plant = it->second;
  plant.lastSeen = sFrame;
  if (!fresh) {
    return;
  }
  const CAABox local = surface.GetBounds();
  const CAABox world = local.GetTransformedAABox(xf);
  plant.min = world.GetMinPoint();
  plant.max = world.GetMaxPoint();
  // Some of the model's vertices inside the surface's bounds: the plant's own, near enough.
  const float* pos = static_cast< const float* >(model.GetPositions());
  const uint count = pos != nullptr ? model.GetModelInstance().GetVertexSize() / 12 : 0;
  const CVector3f& mn = local.GetMinPoint();
  const CVector3f& mx = local.GetMaxPoint();
  const uint step = std::max(1u, count / 6000);
  for (uint i = 0; i < count && plant.points.size() < size_t(kPointsPerPlant * 4); i += step) {
    const CVector3f v(CBasics::SwapBytes(pos[i * 3]), CBasics::SwapBytes(pos[i * 3 + 1]),
                      CBasics::SwapBytes(pos[i * 3 + 2]));
    if (v.GetX() >= mn.GetX() - 0.02f && v.GetX() <= mx.GetX() + 0.02f && v.GetY() >= mn.GetY() - 0.02f &&
        v.GetY() <= mx.GetY() + 0.02f && v.GetZ() >= mn.GetZ() - 0.02f && v.GetZ() <= mx.GetZ() + 0.02f) {
      plant.points.push_back(xf * v);
    }
  }
  // Thin to an even spread.
  if (plant.points.size() > size_t(kPointsPerPlant)) {
    std::vector< CVector3f > kept;
    const size_t n = plant.points.size();
    for (int i = 0; i < kPointsPerPlant; ++i) {
      kept.push_back(plant.points[size_t(i) * n / kPointsPerPlant]);
    }
    plant.points.swap(kept);
  }
  if (plant.points.empty()) {
    plant.points.push_back(Center(plant));
  }
}

void Upload(const CStateManager& mgr) {
  if (!Active()) {
    if (!sSentEmpty) {
      const f32 up[3] = {0.f, 1.f, 0.f};
      const f32 iv[3][4] = {};
      GXPortSetFoliagePushers(up, nullptr, nullptr, 0);
      GXPortSetFoliageSurface(0.f, 0.f, iv, nullptr, nullptr, 0);
      sSentEmpty = true;
    }
    sReleasedCount = 0;
    sShotCount = 0;
    sHavePlayer = false;
    sLastClock = -1.f;
    ClearFires();
    return;
  }
  sSentEmpty = false;
  ++sFrame;

  const float dt = Advance();
  const bool interactive = PortDebug::InteractiveFoliage();
  const bool burnable = PortDebug::BurnableFoliage();
  const bool rainDetails = PortDebug::RainDetails();
  float rainMagnitude = 0.f;
  bool exposed = false;
  const bool raining = Raining(mgr, &rainMagnitude, &exposed);

  // Forget plants long out of sight (their area may be gone).
  if ((sFrame & 63) == 0) {
    for (auto it = Plants().begin(); it != Plants().end();) {
      if (it->second.lastSeen + kForgetFrames < sFrame && !it->second.burning) {
        it = Plants().erase(it);
      } else {
        ++it;
      }
    }
  }

  for (int i = sReleasedCount - 1; i >= 0; --i) {
    sReleased[i].age += dt;
    if (sReleased[i].age > kReleaseLife) {
      sReleased[i] = sReleased[--sReleasedCount];
    }
  }

  Pusher live[GX_PORT_FOLIAGE_MAX_PUSHERS];
  int liveCount = 0;

  // Samus: her body, or the morph ball.
  const CPlayer* player = mgr.GetPlayer();
  if (player != nullptr) {
    const CVector3f feet = player->GetTranslation();
    const bool ball = player->GetMorphballTransitionState() == CPlayer::kMS_Morphed;
    const float radius = ball ? 0.8f : 0.75f;
    const float height = ball ? 0.9f : 2.6f;
    if (!sHavePlayer || (feet - sPlayerTrail).Magnitude() > 10.f) {
      sPlayerTrail = feet;
      sHavePlayer = true;
    } else if ((feet - sPlayerTrail).Magnitude() > kPlayerTrailStep) {
      Release(sPlayerTrail, radius, height, 1.f);
      sPlayerTrail = feet;
    }
    Pusher& p = live[liveCount++];
    p.pos = feet;
    p.radius = radius;
    p.height = height;
    p.strength = 1.f;
  }

  // Shots in flight push what they pass through, and Samus's beams and missiles set it alight.
  const CVector3f eye = CGraphics::GetViewMatrix().GetTranslation();
  for (int i = 0; i < sShotCount; ++i) {
    sShots[i].seen = false;
  }
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CGameProjectile* proj = TCastToConstPtr< CGameProjectile >(actors.GetObjectByIndex(idx));
    if (proj == nullptr || !proj->GetActive()) {
      continue;
    }
    const CVector3f pos = proj->GetTranslation();
    if ((pos - eye).Magnitude() > 40.f) {
      continue;
    }
    const CVector3f feet = pos - CVector3f(0.f, 0.f, 0.6f);
    Shot* shot = nullptr;
    for (int i = 0; i < sShotCount; ++i) {
      if (sShots[i].uid == proj->GetUniqueId()) {
        shot = &sShots[i];
        break;
      }
    }
    if (shot == nullptr && sShotCount < kMaxShots) {
      shot = &sShots[sShotCount++];
      shot->uid = proj->GetUniqueId();
      shot->trail = feet;
      // Its first frame: from the gun's muzzle, near enough the eye.
      shot->last = player != nullptr ? eye : pos;
      const EWeaponType type = proj->GetType();
      shot->burns = player != nullptr && proj->GetOwnerId() == player->GetUniqueId() &&
                    (type == kWT_Power || type == kWT_Wave || type == kWT_Plasma || type == kWT_Missile ||
                     type == kWT_Phazon);
    }
    if (shot != nullptr) {
      shot->seen = true;
      if (burnable && shot->burns) {
        for (auto& [key, plant] : Plants()) {
          CVector3f hit;
          if (plant.lastSeen + 2 >= sFrame && SegmentHitsBox(shot->last, pos, plant.min, plant.max, 0.15f, &hit)) {
            Ignite(key, plant, hit, raining);
          }
        }
      }
      shot->last = pos;
      if ((feet - shot->trail).Magnitude() > kShotTrailStep) {
        Release(shot->trail, 0.8f, 1.2f, 0.8f);
        shot->trail = feet;
      }
    }
    if (liveCount < 6) {
      Pusher& p = live[liveCount++];
      p.pos = feet;
      p.radius = 0.8f;
      p.height = 1.2f;
      p.strength = 0.8f;
    }
  }
  for (int i = sShotCount - 1; i >= 0; --i) {
    if (!sShots[i].seen) {
      Release(sShots[i].last - CVector3f(0.f, 0.f, 0.6f), 1.3f, 1.6f, 1.2f);
      sShots[i] = sShots[--sShotCount];
    }
  }

  if (burnable) {
    if (dt > 0.f) {
      UpdateFires(mgr, dt, raining && exposed);
    }
  } else {
    ClearFires();
  }
  if (rainDetails && raining && dt > 0.f) {
    UpdateSplashes(mgr, dt, eye);
  } else if (!(rainDetails && raining)) {
    sSplashing = false;
  }

  // GX's view space: the camera's right, up and backward rows, about its eye.
  const CTransform4f& view = CGraphics::GetViewMatrix();
  const float axes[3][3] = {
      {view.Get00(), view.Get10(), view.Get20()},
      {view.Get02(), view.Get12(), view.Get22()},
      {-view.Get01(), -view.Get11(), -view.Get21()},
  };
  const auto toView = [&](const CVector3f& w, f32 out[4]) {
    const float d[3] = {w.GetX() - eye.GetX(), w.GetY() - eye.GetY(), w.GetZ() - eye.GetZ()};
    for (int r = 0; r < 3; ++r) {
      out[r] = axes[r][0] * d[0] + axes[r][1] * d[1] + axes[r][2] * d[2];
    }
  };

  // Pushers: live first, then the freshest of the released ones.
  const f32 up[3] = {axes[0][2], axes[1][2], axes[2][2]};
  f32 pos[GX_PORT_FOLIAGE_MAX_PUSHERS][4];
  f32 params[GX_PORT_FOLIAGE_MAX_PUSHERS][4];
  u32 count = 0;
  if (interactive) {
    Pusher* released[kMaxReleased];
    for (int i = 0; i < sReleasedCount; ++i) {
      released[i] = &sReleased[i];
    }
    std::sort(released, released + sReleasedCount,
              [](const Pusher* a, const Pusher* b) { return a->age < b->age; });
    const auto add = [&](const Pusher& p, float age) {
      toView(p.pos, pos[count]);
      pos[count][3] = p.radius;
      params[count][0] = p.strength;
      params[count][1] = age;
      params[count][2] = p.height;
      params[count][3] = 0.f;
      ++count;
    };
    for (int i = 0; i < liveCount && count < GX_PORT_FOLIAGE_MAX_PUSHERS; ++i) {
      add(live[i], 0.f);
    }
    for (int i = 0; i < sReleasedCount && count < GX_PORT_FOLIAGE_MAX_PUSHERS; ++i) {
      add(*released[i], released[i]->age);
    }
  }
  GXPortSetFoliagePushers(up, pos, params, count);

  // Rain, burns and the view's world frame.
  f32 iv[3][4];
  for (int r = 0; r < 3; ++r) {
    iv[r][0] = axes[0][r];
    iv[r][1] = axes[1][r];
    iv[r][2] = axes[2][r];
  }
  iv[0][3] = eye.GetX();
  iv[1][3] = eye.GetY();
  iv[2][3] = eye.GetZ();
  struct Burn {
    CVector3f c;
    float r, glow, chr, dist;
  };
  std::vector< Burn > burns;
  if (burnable) {
    for (const auto& f : Fires()) {
      auto it = Plants().find(f->key);
      const float chr = it != Plants().end() ? it->second.charAmount : 1.f;
      burns.push_back({f->origin, f->Radius(), f->Glow(), chr, (f->origin - eye).Magnitude()});
    }
    // Plants burnt earlier stay black.
    for (const auto& [key, plant] : Plants()) {
      if (!plant.burning && plant.charAmount > 0.f && plant.lastSeen + 2 >= sFrame) {
        const CVector3f c = Center(plant);
        burns.push_back({c, std::max((plant.max - plant.min).Magnitude() * 0.6f, 0.6f), 0.f, plant.charAmount,
                         (c - eye).Magnitude()});
      }
    }
    std::sort(burns.begin(), burns.end(), [](const Burn& a, const Burn& b) { return a.dist < b.dist; });
  }
  f32 bpos[GX_PORT_FOLIAGE_MAX_BURNS][4];
  f32 bpar[GX_PORT_FOLIAGE_MAX_BURNS][4];
  u32 nb = 0;
  for (const Burn& b : burns) {
    if (nb == GX_PORT_FOLIAGE_MAX_BURNS) {
      break;
    }
    bpos[nb][0] = b.c.GetX();
    bpos[nb][1] = b.c.GetY();
    bpos[nb][2] = b.c.GetZ();
    bpos[nb][3] = b.r;
    bpar[nb][0] = b.glow;
    bpar[nb][1] = b.chr;
    bpar[nb][2] = 0.f;
    bpar[nb][3] = 0.f;
    ++nb;
  }
  const float rain = rainDetails && raining ? std::clamp(rainMagnitude * (exposed ? 1.f : 0.5f), 0.f, 1.f) : 0.f;
  GXPortSetFoliageSurface(rain, sSeconds, iv, bpos, bpar, nb);
}

void DrawEffects(const CStateManager& mgr) {
  (void)mgr;
  if (!Active()) {
    return;
  }
  for (const auto& f : Fires()) {
    for (const auto& g : f->smoke) {
      g->Render();
    }
    for (const auto& g : f->flames) {
      g->Render();
    }
  }
  if (sSplashing && sSplashes != nullptr) {
    sSplashes->Draw(CTransform4f::Identity());
  }
}

void Reset() {
  ClearFires();
  Plants().clear();
  sReleasedCount = 0;
  sShotCount = 0;
  sHavePlayer = false;
}
} // namespace PortFoliage
