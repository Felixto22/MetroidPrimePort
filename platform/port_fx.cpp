// Modern effects (port_fx.h).
#include "port_fx.h"

#include "port_enemy_ai.h"

#include "port_debug.h"
#include "port_log.h"
#include "port_room_geo.h"

#include "Collision/CMaterialFilter.hpp"
#include "Collision/CRayCastResult.hpp"
#include "Kyoto/Graphics/CGraphics.hpp"
#include "Kyoto/Graphics/CLight.hpp"
#include "MetaRender/CCubeRenderer.hpp"
#include "MetroidPrime/CEnvFxManager.hpp"
#include "MetroidPrime/CGameArea.hpp"
#include "MetroidPrime/CGameCollision.hpp"
#include "MetroidPrime/CObjectList.hpp"
#include "MetroidPrime/CStateManager.hpp"
#include "MetroidPrime/CWorld.hpp"
#include "MetroidPrime/Cameras/CCameraManager.hpp"
#include "MetroidPrime/Cameras/CGameCamera.hpp"
#include "MetroidPrime/Player/CMorphBall.hpp"
#include "MetroidPrime/Player/CPlayer.hpp"
#include "MetroidPrime/Player/CPlayerGun.hpp"
#include "MetroidPrime/Player/CPlayerState.hpp"
#include "MetroidPrime/TCastTo.hpp"
#include "MetroidPrime/Weapons/CGameProjectile.hpp"

#include <aurora/fxp.hpp>
#include <dolphin/gx/GXExtra.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace PortFx {
namespace {
using aurora::gfx::fxp::Particle;
namespace fxp = aurora::gfx::fxp;

constexpr size_t kMaxParticles = 40000;
constexpr float kNoFloor = -1e9f;
constexpr float kPi = 3.14159265f;

// ---- Small helpers ----
struct Rng {
  uint32_t s = 0x9e3779b9u;
  uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  }
  float Unit() { return float(Next() & 0xffffff) / float(0x1000000); }
  float Range(float a, float b) { return a + (b - a) * Unit(); }
  // A random direction (uniform on the sphere).
  CVector3f Dir() {
    const float z = Range(-1.f, 1.f);
    const float a = Unit() * 2.f * kPi;
    const float r = std::sqrt(std::max(0.f, 1.f - z * z));
    return CVector3f(r * std::cos(a), r * std::sin(a), z);
  }
};
Rng sRng;

float Len(const CVector3f& v) { return v.Magnitude(); }

CVector3f Norm(const CVector3f& v) {
  const float l = v.Magnitude();
  return l > 1e-6f ? v * (1.f / l) : CVector3f(0.f, 0.f, 1.f);
}

float SrgbToLinear(float c) {
  c = std::clamp(c, 0.f, 1.f);
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

const CMaterialFilter& SolidFilter() {
  static const CMaterialFilter filter = CMaterialFilter::MakeIncludeExclude(
      CMaterialList(kMT_Solid), CMaterialList(kMT_ProjectilePassthrough, kMT_SeeThrough));
  return filter;
}

// The floor's height below a point (kNoFloor when there is none within `reach`).
float FloorBelow(const CStateManager& mgr, const CVector3f& at, float reach) {
  const CRayCastResult hit = CGameCollision::RayStaticIntersection(mgr, at + CVector3f(0.f, 0.f, 0.3f),
                                                                    CVector3f::Down(), reach, SolidFilter());
  return hit.IsValid() ? hit.GetPoint().GetZ() : kNoFloor;
}

// ---- The clock and the particles ----
float sNow = 0.f;        // seconds of game time this module has seen
float sLastClock = -1.f; // GetSecondsMod900 at the last step
int sLastFrame = -1;     // the frame counter the step ran for
float sDt = 0.f;
std::vector< Particle > sParticles;
size_t sOverwrite = 0;

float Density() { return float(PortDebug::ParticleDensity()) / 200.f; }

// Scales a count by the density setting, carrying the fraction from frame to frame by chance.
int Count(float n) {
  n *= Density();
  int whole = int(n);
  if (sRng.Unit() < n - float(whole)) {
    ++whole;
  }
  return whole;
}

void Emit(const CVector3f& pos, const CVector3f& vel, float life, float size, uint32_t color, uint32_t flags,
          float drag = 0.f, float gravity = 0.f, float floorZ = kNoFloor, float stretch = 0.f) {
  Particle p{};
  p.pos[0] = pos.GetX(), p.pos[1] = pos.GetY(), p.pos[2] = pos.GetZ();
  p.t0 = sNow;
  p.vel[0] = vel.GetX(), p.vel[1] = vel.GetY(), p.vel[2] = vel.GetZ();
  p.life = life;
  p.size = size;
  p.drag = drag;
  p.gravity = gravity;
  p.floorZ = floorZ;
  p.color = color;
  p.flags = flags;
  p.stretch = stretch;
  p.seed = sRng.Unit();
  if (sParticles.size() < kMaxParticles) {
    sParticles.push_back(p);
  } else {
    sParticles[sOverwrite++ % kMaxParticles] = p;
  }
}

// ---- What the frame knows ----
struct Fluid {
  int type;
  CVector3f min, max;
  float surface;
};
std::vector< Fluid > sFluids, sFluidsLast;
int sFluidFrame = -1;
bool sSkyDrawn = false;
// Lightning (in a rainstorm under the sky): the flash's brightness, and the time to the next strike.
float sFlash = 0.f;
float sNextStrike = 6.f;

void RollFluids() {
  const int frame = CGraphics::GetFrameCounter();
  if (frame != sFluidFrame) {
    sFluidFrame = frame;
    sFluidsLast.swap(sFluids);
    sFluids.clear();
  }
}

// The fluid whose volume holds a point (nullptr for none).
const Fluid* FluidAt(const CVector3f& p, bool water) {
  for (const Fluid& f : sFluidsLast) {
    const bool isWater = f.type == 0 || f.type == 1 || f.type == 3;
    if (isWater != water) {
      continue;
    }
    if (p.GetX() >= f.min.GetX() && p.GetX() <= f.max.GetX() && p.GetY() >= f.min.GetY() &&
        p.GetY() <= f.max.GetY() && p.GetZ() <= f.surface && p.GetZ() >= f.min.GetZ() - 1.f) {
      return &f;
    }
  }
  return nullptr;
}

float BoxDistanceXY(const Fluid& f, const CVector3f& p) {
  const float dx = std::max({f.min.GetX() - p.GetX(), 0.f, p.GetX() - f.max.GetX()});
  const float dy = std::max({f.min.GetY() - p.GetY(), 0.f, p.GetY() - f.max.GetY()});
  return std::sqrt(dx * dx + dy * dy);
}

bool Lava(int type) { return type == 2 || type == 5; }

bool NormalView(const CStateManager& mgr) {
  const CPlayerState::EPlayerVisor visor = mgr.GetPlayerState()->GetActiveVisor(mgr);
  return visor != CPlayerState::kPV_Thermal && visor != CPlayerState::kPV_XRay;
}

// ---- Day, night and weather (outdoors: where the world's sky is drawn) ----
// The day lasts kDayLength seconds: 0 midnight, 0.25 sunrise, 0.5 noon, 0.75 sunset. The weather
// moves between clear, cloudy, rain and storm, each held a few minutes, easing over half a minute.
constexpr float kDayLength = 600.f;
float sDayClock = 0.32f;
float sSkyClock = -1.f;
int sWeatherState = 1;
float sWeatherHold = 150.f;
float sCover = 0.45f, sPrecip = 0.f, sStorm = 0.f;
uint32_t sWeatherRng = 0x51ed27u;

// The planet's worlds (Tallon Overworld, Chozo Ruins, Phendrana Drifts): not the frigate or the
// crater in space, not Magmoor's caverns or the mines underground.
bool sPlanetWorld = false;
// Retail's sky domes around the camera this frame (Tallon's rooms draw their sky that way): big
// scripted actors whose bounds hold the eye, so a big cliff or backdrop is never one.
TUniqueId sDomes[8];
int sDomeCount = 0;
// The new sky drawn first this frame, behind everything (DrawSkyFirst), so retail's skies are not.
bool sSkyFirst = false;
// DrawAtmosphere drawing only that sky, behind the world.
bool sSkyOnlyPass = false;
bool Outdoors() { return sPlanetWorld && (sSkyDrawn || sDomeCount > 0); }
bool DayNightOn() { return PortDebug::DayNight() && Outdoors(); }

// The day and night's parts, each on or off to find which one misdraws (port_fx.h DiagBits).
uint32_t sDiag = kDiagDefault;
bool WeatherOn() { return PortDebug::DynamicWeather() && Outdoors(); }

float WeatherUnit() {
  sWeatherRng ^= sWeatherRng << 13;
  sWeatherRng ^= sWeatherRng >> 17;
  sWeatherRng ^= sWeatherRng << 5;
  return float(sWeatherRng & 0xffffff) / float(0x1000000);
}

// Direction to the sun and the moon (world, z up), and the sun's elevation (sin).
void SkyBodies(float sun[3], float moon[3], float& elev) {
  const float a = (sDayClock - 0.25f) * 6.2831853f;
  const float s[3] = {std::cos(a), -0.38f, std::sin(a)};
  const float l = std::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]);
  for (int i = 0; i < 3; ++i) {
    sun[i] = s[i] / l;
  }
  // The moon on the same arc, half a day behind.
  const float m[3] = {-s[0], s[1], -s[2]};
  for (int i = 0; i < 3; ++i) {
    moon[i] = m[i] / l;
  }
  elev = sun[2];
}

// The key light outdoors: sun by day, moon by night (its intensity, colour).
void KeyLight(float dir[3], float& intensity, float color[3]) {
  float sun[3], moon[3], elev;
  SkyBodies(sun, moon, elev);
  // Each fades in over its first 17 degrees, so a low sun (or moon) lights and shadows little; at
  // the swap between them both are out.
  const auto rise = [](float e) {
    const float t = std::clamp(e / 0.3f, 0.f, 1.f);
    return t * t * (3.f - 2.f * t);
  };
  const float sunI = rise(elev) * (1.f - 0.6f * sStorm) * (1.f - 0.35f * sCover);
  const float moonI = rise(moon[2]) * 0.28f * (1.f - 0.7f * sCover);
  const float warm = 1.f - std::clamp(elev / 0.35f, 0.f, 1.f);
  if (sunI >= moonI) {
    for (int i = 0; i < 3; ++i) {
      dir[i] = sun[i];
    }
    intensity = sunI;
    color[0] = 1.f, color[1] = 0.95f - 0.45f * warm, color[2] = 0.88f - 0.68f * warm;
  } else {
    for (int i = 0; i < 3; ++i) {
      dir[i] = moon[i];
    }
    intensity = moonI;
    color[0] = 0.55f, color[1] = 0.65f, color[2] = 0.9f;
  }
  // The light (and its shadows) no lower than 15 degrees: no shadow a hundred metres long.
  const float minZ = 0.26f;
  if (dir[2] < minZ) {
    const float h = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1]);
    const float k = h > 1e-4f ? std::sqrt(1.f - minZ * minZ) / h : 0.f;
    dir[0] *= k, dir[1] *= k, dir[2] = minZ;
  }
}

// The sky's light on the relit world (GXPortSetRelightAmbient), against the key light's: by day
// the sun gives most of it, by night (which DayGrade darkens as a whole) the sky; more under cloud.
float RelightAmbient() {
  float sun[3], moon[3], elev;
  SkyBodies(sun, moon, elev);
  const float t = std::clamp((elev + 0.1f) / 0.35f, 0.f, 1.f);
  const float day = t * t * (3.f - 2.f * t);
  return (0.72f - 0.3f * day) * (1.f + 0.3f * sCover);
}

// The world's light outdoors for the time of day and the weather (a multiplier).
void DayGrade(float out[3]) {
  float sun[3], moon[3], elev;
  SkyBodies(sun, moon, elev);
  const float day = std::clamp((elev + 0.12f) / 0.32f, 0.f, 1.f);
  const float golden = (1.f - std::clamp((elev - 0.05f) / 0.3f, 0.f, 1.f)) * day;
  const float night[3] = {0.22f, 0.27f, 0.4f};
  const float warm[3] = {1.08f, 0.86f, 0.68f};
  const float dull = 1.f - 0.2f * sCover - 0.2f * sStorm;
  for (int i = 0; i < 3; ++i) {
    const float lit = 1.f + (warm[i] - 1.f) * golden;
    out[i] = (night[i] + (lit - night[i]) * day) * dull;
  }
}

EEnvFxType EnvFx(const CStateManager& mgr) {
  const EEnvFxType retail = mgr.GetWorld() != nullptr ? mgr.GetWorld()->GetNeededEnvFx() : kEFX_None;
  if (WeatherOn()) {
    return retail == kEFX_Snow ? kEFX_Snow : kEFX_Rain;
  }
  return retail;
}

float EnvMagnitude(const CStateManager& mgr) {
  if (WeatherOn()) {
    return sPrecip;
  }
  return mgr.GetEnvFxManager() != nullptr ? std::clamp(mgr.GetEnvFxManager()->GetRainMagnitude(), 0.f, 1.f) : 0.f;
}

void ViewRows(const CTransform4f& view, float out[12]) {
  // View space is GX's: right, up, towards the camera.
  const CVector3f eye = view.GetTranslation();
  const float rows[12] = {view.Get00(), view.Get02(), -view.Get01(), eye.GetX(),
                          view.Get10(), view.Get12(), -view.Get11(), eye.GetY(),
                          view.Get20(), view.Get22(), -view.Get21(), eye.GetZ()};
  std::memcpy(out, rows, sizeof(rows));
}

void FillCommon(GXPortPostParams& p, const CTransform4f& view) {
  std::memset(&p, 0, sizeof(p));
  const CGraphics::CProjectionState& proj = CGraphics::GetProjectionState();
  const float zNear = proj.GetNear();
  p.frustum[0] = proj.GetLeft() / zNear;
  p.frustum[1] = proj.GetRight() / zNear;
  p.frustum[2] = proj.GetBottom() / zNear;
  p.frustum[3] = proj.GetTop() / zNear;
  p.depth[0] = zNear;
  p.depth[1] = proj.GetFar();
  // SetupViewForDraw's depth range.
  p.depth[2] = 0.125f;
  p.depth[3] = 1.f;
  ViewRows(view, p.view);
  std::memcpy(p.prevView, p.view, sizeof(p.view));
  p.weather[1] = std::fmod(sNow, 3600.f);
  p.misc[2] = static_cast< u32 >(CGraphics::GetFrameCounter());
}

// ---- Shots ----
enum ShotKind { kPower, kIce, kWave, kPlasma, kPhazon, kMissile, kEnemy };
struct Shot {
  TUniqueId uid;
  CVector3f pos, prev;
  ShotKind kind;
  bool charged;
  bool seen;
  float carry; // trail distance not yet emitted for
};
std::vector< Shot > sShots;

struct Palette {
  uint32_t core, spark;
  uint32_t smoke;
};
Palette ColorsOf(ShotKind kind) {
  switch (kind) {
  case kPower:
    return {0xffffd060, 0xffffb030, 0x50a09080};
  case kIce:
    return {0xff9ee6ff, 0xffe6f8ff, 0x60d8ecf4};
  case kWave:
    return {0xffc070ff, 0xffe0b0ff, 0x50a080b0};
  case kPlasma:
    return {0xffff6a20, 0xffffa040, 0x60605048};
  case kPhazon:
    return {0xffff3060, 0xffff90a0, 0x50803050};
  case kMissile:
    return {0xffffa040, 0xffffc070, 0x80707070};
  case kEnemy:
  default:
    return {0xc0ffa060, 0xffffc080, 0x50908880};
  }
}

ShotKind KindOf(const CGameProjectile& proj, bool player) {
  if (!player) {
    return kEnemy;
  }
  switch (proj.GetType()) {
  case kWT_Power:
    return kPower;
  case kWT_Ice:
    return kIce;
  case kWT_Wave:
    return kWave;
  case kWT_Plasma:
    return kPlasma;
  case kWT_Phazon:
    return kPhazon;
  case kWT_Missile:
    return kMissile;
  default:
    return kPower;
  }
}

// Arcs of lightning crackling around the beams (more and larger when charged).
void EnergyTrail(const Shot& s, const CVector3f& from, const CVector3f& to) {
  if (s.kind == kMissile || s.kind == kEnemy) {
    return;
  }
  const Palette c = ColorsOf(s.kind);
  const float k = s.charged ? 1.8f : 1.f;
  const CVector3f d = to - from;
  const float len = Len(d);
  const int arcs = std::clamp(Count(len * (s.charged ? 3.f : 1.4f)), 0, 12);
  for (int i = 0; i < arcs; ++i) {
    const CVector3f at = to - d * (0.3f * sRng.Unit());
    Emit(at + sRng.Dir() * 0.08f * k, sRng.Dir() * sRng.Range(4.f, 8.f), sRng.Range(0.05f, 0.1f), 0.04f * k,
         s.kind == kWave ? c.spark : 0xffd8f0ff, fxp::Arc | fxp::Static, 0.f, 0.f, kNoFloor, 0.05f * k);
  }
}

void Trail(const Shot& s, const CVector3f& from, const CVector3f& to, bool haze) {
  EnergyTrail(s, from, to);
  const Palette c = ColorsOf(s.kind);
  const float k = s.charged ? 1.6f : 1.f;
  const CVector3f d = to - from;
  const float len = Len(d);
  const CVector3f dir = Norm(d);
  const float perMeter = (s.kind == kMissile ? 9.f : s.kind == kEnemy ? 3.f : 6.f) * (s.charged ? 1.8f : 1.f);
  const int n = std::clamp(Count(len * perMeter), 1, 60);
  for (int i = 0; i < n; ++i) {
    const CVector3f at = from + d * ((float(i) + sRng.Unit()) / float(n));
    switch (s.kind) {
    case kPower:
      Emit(at, sRng.Dir() * 0.4f, 0.22f, 0.16f * k, c.core, fxp::Glow | fxp::Shrink, 4.f);
      if ((i & 1) == 0) {
        Emit(at, sRng.Dir() * sRng.Range(1.5f, 3.5f) - dir * 1.5f, sRng.Range(0.2f, 0.4f), 0.03f * k, c.spark,
             fxp::Spark, 2.f, 5.f, kNoFloor, 0.03f);
      }
      break;
    case kIce:
      Emit(at, sRng.Dir() * 0.3f, 0.35f, 0.2f * k, c.core, fxp::Glow | fxp::Shrink, 3.f);
      Emit(at, sRng.Dir() * 0.8f, sRng.Range(0.6f, 1.1f), 0.05f * k, c.spark, fxp::Flake | fxp::Shrink, 2.f, 1.5f);
      break;
    case kWave:
      Emit(at, sRng.Dir() * 0.3f, 0.2f, 0.22f * k, c.core, fxp::Glow | fxp::Shrink, 4.f);
      if ((i & 1) == 0) {
        Emit(at, sRng.Dir() * sRng.Range(5.f, 9.f), sRng.Range(0.06f, 0.12f), 0.05f * k, c.spark, fxp::Arc, 0.f, 0.f,
             kNoFloor, 0.06f);
      }
      break;
    case kPlasma:
      Emit(at, sRng.Dir() * 0.4f, 0.3f, 0.26f * k, c.core, fxp::Glow | fxp::Shrink, 4.f);
      Emit(at, sRng.Dir() * 1.4f, sRng.Range(0.6f, 1.1f), 0.05f * k, c.spark, fxp::Ember | fxp::Shrink, 1.5f, -1.5f);
      if (haze && (i % 3) == 0) {
        Emit(at, CVector3f(0.f, 0.f, 0.3f), 0.5f, 0.7f * k, 0xa0ffffff, fxp::Haze, 1.f);
      }
      break;
    case kPhazon:
      Emit(at, sRng.Dir() * 0.4f, 0.3f, 0.24f * k, c.core, fxp::Glow | fxp::Shrink, 4.f);
      Emit(at, sRng.Dir() * 2.f, 0.3f, 0.03f * k, c.spark, fxp::Spark, 2.f, 3.f, kNoFloor, 0.03f);
      break;
    case kMissile:
      Emit(at, sRng.Dir() * 0.25f + CVector3f(0.f, 0.f, 0.1f), sRng.Range(1.2f, 1.8f), sRng.Range(0.25f, 0.4f),
           c.smoke, fxp::Smoke, 2.f, -0.3f);
      if ((i & 1) == 0) {
        Emit(at, -dir * 1.f, 0.1f, 0.3f, c.core, fxp::Glow | fxp::Shrink, 3.f);
      }
      break;
    case kEnemy:
      Emit(at, sRng.Dir() * 0.3f, 0.2f, 0.18f, c.core, fxp::Glow | fxp::Shrink, 4.f);
      break;
    }
  }
}

void Impact(const CStateManager& mgr, const Shot& s, const CVector3f& at, const CVector3f& dir, bool haze) {
  const Palette c = ColorsOf(s.kind);
  const float k = s.kind == kMissile ? 2.f : s.charged ? 1.6f : s.kind == kEnemy ? 0.7f : 1.f;
  const float floorZ = FloorBelow(mgr, at, 6.f);
  const CVector3f back = -dir;
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.12f, 1.2f * k, c.core, fxp::Glow);
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.35f, 1.6f * k, (c.core & 0x00ffffff) | 0xc0000000, fxp::Ring);
  const int sparks = Count(18.f * k);
  for (int i = 0; i < sparks; ++i) {
    const CVector3f v = Norm(back * 0.7f + sRng.Dir()) * sRng.Range(4.f, 10.f) * std::sqrt(k);
    Emit(at, v, sRng.Range(0.45f, 1.f), 0.035f, c.spark, fxp::Spark, 1.2f, 9.8f, floorZ, 0.035f);
  }
  const int smoke = Count(4.f * k);
  for (int i = 0; i < smoke; ++i) {
    const CVector3f v = sRng.Dir() * 0.6f + back * 0.5f + CVector3f(0.f, 0.f, 0.4f);
    Emit(at, v, sRng.Range(1.5f, 2.6f), sRng.Range(0.6f, 1.f) * k, c.smoke, fxp::Smoke, 1.5f, -0.4f);
  }
  switch (s.kind) {
  case kIce:
    for (int i = 0, n = Count(12.f * k); i < n; ++i) {
      Emit(at, Norm(back + sRng.Dir()) * sRng.Range(1.5f, 4.f), sRng.Range(0.8f, 1.4f), 0.07f, 0xe0e0f4ff,
           fxp::Flake | fxp::Shrink, 1.f, 6.f, floorZ);
    }
    break;
  case kWave:
    for (int i = 0, n = Count(8.f * k); i < n; ++i) {
      Emit(at, sRng.Dir() * sRng.Range(6.f, 10.f), sRng.Range(0.08f, 0.16f), 0.05f, c.spark, fxp::Arc, 0.f, 0.f,
           kNoFloor, 0.08f);
    }
    break;
  case kPlasma:
  case kMissile:
  case kPower:
    for (int i = 0, n = Count((s.kind == kPower ? 4.f : 10.f) * k); i < n; ++i) {
      Emit(at, sRng.Dir() * 2.f + CVector3f(0.f, 0.f, 1.5f), sRng.Range(1.f, 1.8f), 0.05f, c.spark,
           fxp::Ember | fxp::Shrink, 1.f, 2.f, floorZ);
    }
    break;
  default:
    break;
  }
  if (s.kind == kMissile) {
    for (int i = 0; i < 5; ++i) {
      Emit(at + sRng.Dir() * 0.3f, sRng.Dir() * 1.5f, sRng.Range(0.3f, 0.5f), sRng.Range(0.8f, 1.3f), 0xffff8030,
           fxp::Glow | fxp::Shrink, 3.f);
    }
  }
  if (haze && (s.kind == kMissile || s.kind == kPlasma)) {
    Emit(at, CVector3f(0.f, 0.f, 0.4f), 0.7f, 2.5f * k, 0xc0ffffff, fxp::Haze, 1.f);
  }
}

void UpdateShots(const CStateManager& mgr, bool haze) {
  for (Shot& s : sShots) {
    s.seen = false;
  }
  const TUniqueId playerId = mgr.GetPlayer() != nullptr ? mgr.GetPlayer()->GetUniqueId() : kInvalidUniqueId;
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CGameProjectile* proj = TCastToConstPtr< CGameProjectile >(actors.GetObjectByIndex(idx));
    if (proj == nullptr || !proj->GetActive()) {
      continue;
    }
    auto it = std::find_if(sShots.begin(), sShots.end(), [&](const Shot& s) { return s.uid == proj->GetUniqueId(); });
    if (it == sShots.end()) {
      Shot s{};
      s.uid = proj->GetUniqueId();
      s.pos = s.prev = proj->GetTranslation();
      const bool player = proj->GetOwnerId() == playerId;
      s.kind = KindOf(*proj, player);
      s.charged = proj->HasAttrib(CWeapon::kPA_Charged);
      s.seen = true;
      sShots.push_back(s);
      continue;
    }
    it->prev = it->pos;
    it->pos = proj->GetTranslation();
    it->seen = true;
    const float moved = Len(it->pos - it->prev);
    if (moved > 1e-3f && moved < 60.f) {
      Trail(*it, it->prev, it->pos, haze);
    }
  }
  for (size_t i = sShots.size(); i-- > 0;) {
    if (sShots[i].seen) {
      continue;
    }
    const Shot s = sShots[i];
    sShots.erase(sShots.begin() + std::ptrdiff_t(i));
    const CVector3f d = s.pos - s.prev;
    const float len = Len(d);
    const CVector3f dir = len > 1e-3f ? d * (1.f / len) : CVector3f(0.f, 1.f, 0.f);
    // Where it ended: against the world just ahead, or where it was last (an enemy, its range's end).
    CVector3f at = s.pos;
    const CRayCastResult hit = CGameCollision::RayStaticIntersection(mgr, s.pos - dir * 0.6f, dir, 2.5f, SolidFilter());
    if (hit.IsValid()) {
      at = hit.GetPoint() - dir * 0.08f;
    }
    Impact(mgr, s, at, dir, haze);
  }
}

// ---- The charge beam ----
CVector3f sAnchor;
float sPulse = 0.f;

void UpdateCharge(const CStateManager& mgr, const CTransform4f& view) {
  sPulse = 0.f;
  const CPlayer* player = mgr.GetPlayer();
  if (player == nullptr || player->GetMorphballTransitionState() != CPlayer::kMS_Unmorphed ||
      mgr.GetCameraManager()->IsInCinematicCamera()) {
    return;
  }
  const CPlayerGun* gun = player->GetPlayerGun();
  if (gun == nullptr || !gun->IsCharging()) {
    return;
  }
  const float charge = std::clamp(gun->GetChargePercentage(), 0.f, 1.f);
  if (charge < 0.03f) {
    return;
  }
  sAnchor = gun->PortMuzzleWorld();
  sPulse = charge;
  ShotKind kind = kPower;
  switch (mgr.GetPlayerState()->GetCurrentBeam()) {
  case CPlayerState::kBI_Ice:
    kind = kIce;
    break;
  case CPlayerState::kBI_Wave:
    kind = kWave;
    break;
  case CPlayerState::kBI_Plasma:
    kind = kPlasma;
    break;
  case CPlayerState::kBI_Phazon:
    kind = kPhazon;
    break;
  default:
    break;
  }
  const Palette c = ColorsOf(kind);
  const CVector3f fwd = view.GetForward();
  // Light gathering in on the muzzle, spiralling about the aim.
  for (int i = 0, n = Count((50.f + 160.f * charge) * sDt); i < n; ++i) {
    const CVector3f off = sRng.Dir() * sRng.Range(0.18f, 0.32f + 0.25f * charge);
    Emit(off, fwd, sRng.Range(0.3f, 0.55f), sRng.Range(0.025f, 0.05f) * (1.f + charge), sRng.Unit() < 0.5f ? c.core : c.spark,
         fxp::Glow | fxp::Attached);
  }
  // Arcs of lightning leaping off the muzzle, growing with the charge.
  for (int i = 0, n = Count((6.f + 40.f * charge) * sDt); i < n; ++i) {
    const float ball = 0.06f + 0.22f * charge;
    const CVector3f v = sRng.Dir() * sRng.Range(2.f, 3.f + 4.f * charge);
    Emit(sAnchor + v * (ball * 0.12f), v, sRng.Range(0.05f, 0.09f), 0.025f + 0.02f * charge,
         kind == kWave ? c.spark : 0xffd8f0ff, fxp::Arc | fxp::Static, 0.f, 0.f, kNoFloor, ball * 0.12f);
  }
  // Crackling once it is well charged.
  if (charge > 0.35f) {
    for (int i = 0, n = Count(30.f * charge * sDt); i < n; ++i) {
      Emit(sAnchor, sRng.Dir() * sRng.Range(2.f, 4.f), sRng.Range(0.04f, 0.08f), 0.02f, c.spark,
           kind == kWave ? fxp::Arc : fxp::Spark, 0.f, 0.f, kNoFloor, 0.04f);
    }
  }
}

// ---- Lightning ----
// A jagged chain of glowing static segments from a to b, with the odd branch off it.
void Bolt(CVector3f a, const CVector3f& b, int segments, float width, float life, bool branches) {
  const float span = Len(b - a);
  for (int i = 0; i < segments; ++i) {
    CVector3f next = a + (b - a) * (1.f / float(segments - i)) +
                     CVector3f(sRng.Range(-1.f, 1.f), sRng.Range(-1.f, 1.f), 0.f) * (span * 0.08f);
    if (i == segments - 1) {
      next = b;
    }
    const CVector3f d = next - a;
    // A static streak centred between the two: its length is speed x stretch (10 x 0.1).
    Emit((a + next) * 0.5f, d * 10.f, life, width, 0xffe8f0ff, fxp::Arc | fxp::Static, 0.f, 0.f, kNoFloor, 0.1f);
    if (branches && i > 1 && i < segments - 2 && sRng.Unit() < 0.3f) {
      const float rest = float(segments - i) / float(segments);
      const CVector3f end =
          next + CVector3f(sRng.Range(-1.f, 1.f), sRng.Range(-1.f, 1.f), -0.7f) * (span * 0.3f * rest);
      Bolt(next, end, 4, width * 0.6f, life * 0.8f, false);
    }
    a = next;
  }
}

// In a rainstorm under the open sky: now and then a bolt from the clouds to the ground some way
// off, a jagged chain of glowing segments with branches, and a flash that lights the clouds and air.
void UpdateLightning(const CStateManager& mgr, const CTransform4f& view) {
  sFlash = std::max(0.f, sFlash - sDt * 5.f);
  if (!PortDebug::ModernRain() || EnvFx(mgr) != kEFX_Rain || EnvMagnitude(mgr) < 0.25f || !sSkyDrawn ||
      (WeatherOn() && sStorm < 0.3f)) {
    return;
  }
  sNextStrike -= sDt;
  if (sNextStrike > 0.f) {
    return;
  }
  sNextStrike = sRng.Range(7.f, 18.f) / (0.5f + EnvMagnitude(mgr));
  const CVector3f eye = view.GetTranslation();
  // Somewhere ahead and to the side, 70-160 m away.
  const CVector3f fwd = view.GetForward();
  const float yaw = std::atan2(fwd.GetY(), fwd.GetX()) + sRng.Range(-0.9f, 0.9f);
  const float dist = sRng.Range(70.f, 160.f);
  const CVector3f ground(eye.GetX() + std::cos(yaw) * dist, eye.GetY() + std::sin(yaw) * dist, eye.GetZ() - 5.f);
  const CVector3f top = ground + CVector3f(sRng.Range(-15.f, 15.f), sRng.Range(-15.f, 15.f), sRng.Range(90.f, 130.f));
  const float life = 0.35f;
  Bolt(top, ground, 14, 0.35f, life, true);
  Emit(ground, CVector3f(0.f, 0.f, 0.f), 0.25f, 8.f, 0x60d0e0ff, fxp::Glow | fxp::Static, 0.f);
  sFlash = 0.6f;
}

// ---- Under water, lava, snow ----
void UpdateUnderwater(const CStateManager& mgr, const CVector3f& eye) {
  if (!PortDebug::UnderwaterEffects()) {
    return;
  }
  const Fluid* f = FluidAt(eye, true);
  if (f == nullptr) {
    return;
  }
  for (int i = 0, n = Count(26.f * sDt); i < n; ++i) {
    const CVector3f at = eye + CVector3f(sRng.Range(-6.f, 6.f), sRng.Range(-6.f, 6.f), sRng.Range(-4.f, 2.f));
    if (at.GetZ() >= f->surface - 0.1f) {
      continue;
    }
    // Rising at about 1.5 m/s: gone by the surface.
    const float life = std::min(sRng.Range(2.f, 4.f), (f->surface - at.GetZ()) / 1.5f + 0.4f);
    Emit(at, CVector3f(sRng.Range(-0.2f, 0.2f), sRng.Range(-0.2f, 0.2f), 0.3f), life, sRng.Range(0.03f, 0.1f),
         0x90ffffff, fxp::Bubble, 2.f, -3.f);
  }
  // From Samus' helmet now and then.
  if (const CPlayer* player = mgr.GetPlayer()) {
    for (int i = 0, n = Count(6.f * sDt); i < n; ++i) {
      const CVector3f at = player->GetTranslation() + CVector3f(sRng.Range(-0.2f, 0.2f), sRng.Range(-0.2f, 0.2f), 1.5f);
      if (at.GetZ() < f->surface - 0.1f) {
        Emit(at, CVector3f(0.f, 0.f, 0.5f), std::min(3.f, (f->surface - at.GetZ()) / 1.5f + 0.3f),
             sRng.Range(0.03f, 0.07f), 0xa0ffffff, fxp::Bubble, 2.f, -3.f);
      }
    }
  }
}

const Fluid* NearestLava(const CVector3f& eye, float within) {
  const Fluid* best = nullptr;
  float bestD = within;
  for (const Fluid& f : sFluidsLast) {
    if (!Lava(f.type)) {
      continue;
    }
    const float d = BoxDistanceXY(f, eye);
    if (d < bestD) {
      bestD = d;
      best = &f;
    }
  }
  return best;
}

void UpdateHeat(const CVector3f& eye, bool haze) {
  if (!haze) {
    return;
  }
  const Fluid* f = NearestLava(eye, 30.f);
  if (f == nullptr) {
    return;
  }
  const float x0 = std::max(f->min.GetX(), eye.GetX() - 18.f), x1 = std::min(f->max.GetX(), eye.GetX() + 18.f);
  const float y0 = std::max(f->min.GetY(), eye.GetY() - 18.f), y1 = std::min(f->max.GetY(), eye.GetY() + 18.f);
  if (x0 >= x1 || y0 >= y1) {
    return;
  }
  for (int i = 0, n = Count(14.f * sDt); i < n; ++i) {
    const CVector3f at(sRng.Range(x0, x1), sRng.Range(y0, y1), f->surface + sRng.Range(0.2f, 0.8f));
    Emit(at, CVector3f(0.f, 0.f, 0.4f), sRng.Range(1.4f, 2.2f), sRng.Range(2.f, 3.5f), 0xa0ffffff, fxp::Haze, 0.5f,
         -0.6f);
  }
}

// Footprints and the morph ball's track.
CVector3f sLastStep;
bool sHaveStep = false;
bool sLeftFoot = false;

void UpdatePrints(const CStateManager& mgr) {
  const CPlayer* player = mgr.GetPlayer();
  if (!PortDebug::SnowFootprints() || EnvFx(mgr) != kEFX_Snow || player == nullptr ||
      player->GetPlayerMovementState() != NPlayer::kMS_OnGround) {
    sHaveStep = false;
    return;
  }
  const bool ball = player->GetMorphballTransitionState() == CPlayer::kMS_Morphed;
  const CVector3f pos = player->GetTranslation();
  if (!sHaveStep) {
    sLastStep = pos;
    sHaveStep = true;
    return;
  }
  CVector3f d = pos - sLastStep;
  d = CVector3f(d.GetX(), d.GetY(), 0.f);
  const float moved = Len(d);
  const float stride = ball ? 0.16f : 0.8f;
  if (moved < stride) {
    return;
  }
  if (moved > 5.f) {
    sLastStep = pos; // a teleport
    return;
  }
  sLastStep = pos;
  const CVector3f fwd = d * (1.f / moved);
  const CVector3f side(fwd.GetY(), -fwd.GetX(), 0.f);
  const float heading = std::atan2(fwd.GetY(), fwd.GetX());
  sLeftFoot = !sLeftFoot;
  const CVector3f at = ball ? pos : pos + side * (sLeftFoot ? -0.17f : 0.17f);
  const float floorZ = FloorBelow(mgr, at + CVector3f(0.f, 0.f, 0.5f), 1.6f);
  if (floorZ == kNoFloor) {
    return;
  }
  // The print's long axis lies along its yaw + 90 degrees (see fxp.cpp quad()).
  float yaw = heading - kPi * 0.5f;
  yaw = std::fmod(yaw + 4.f * kPi, 2.f * kPi);
  Particle p{};
  p.pos[0] = at.GetX(), p.pos[1] = at.GetY(), p.pos[2] = floorZ;
  p.t0 = sNow;
  p.life = 40.f;
  p.size = ball ? 0.55f : 0.42f;
  p.floorZ = kNoFloor;
  p.color = 0x80586878;
  p.flags = fxp::Print;
  p.seed = yaw / (2.f * kPi);
  if (sParticles.size() < kMaxParticles) {
    sParticles.push_back(p);
  } else {
    sParticles[sOverwrite++ % kMaxParticles] = p;
  }
  // A little snow kicked up.
  for (int i = 0, n = Count(ball ? 1.f : 3.f); i < n; ++i) {
    Emit(at + CVector3f(0.f, 0.f, 0.05f), CVector3f(sRng.Range(-0.4f, 0.4f), sRng.Range(-0.4f, 0.4f), sRng.Range(0.6f, 1.2f)),
         sRng.Range(0.5f, 0.9f), 0.04f, 0xd0f4f8ff, fxp::Flake | fxp::Shrink, 1.5f, 5.f, floorZ);
  }
}

// ---- The weather's height grid ----
constexpr int kGrid = int(fxp::Field::GridSize);
constexpr float kCell = 2.f;
float sGrid[kGrid * kGrid];
bool sGridKnown[kGrid * kGrid];
int sGridX = INT32_MIN, sGridY = INT32_MIN; // the grid's origin, in cells
int sGridNext = 0;

void UpdateGrid(const CStateManager& mgr, const CVector3f& eye) {
  const int gx = int(std::floor(eye.GetX() / kCell)) - kGrid / 2;
  const int gy = int(std::floor(eye.GetY() / kCell)) - kGrid / 2;
  if (gx != sGridX || gy != sGridY) {
    // Keep what overlaps, forget the rest.
    float grid[kGrid * kGrid];
    bool known[kGrid * kGrid];
    for (int y = 0; y < kGrid; ++y) {
      for (int x = 0; x < kGrid; ++x) {
        const int ox = x + gx - sGridX, oy = y + gy - sGridY;
        const bool keep = sGridX != INT32_MIN && ox >= 0 && ox < kGrid && oy >= 0 && oy < kGrid;
        grid[y * kGrid + x] = keep ? sGrid[oy * kGrid + ox] : kNoFloor;
        known[y * kGrid + x] = keep && sGridKnown[oy * kGrid + ox];
      }
    }
    std::memcpy(sGrid, grid, sizeof(grid));
    std::memcpy(sGridKnown, known, sizeof(known));
    sGridX = gx;
    sGridY = gy;
  }
  // 64 rays a frame: the cells not yet known first, nearest the camera, then the rest in turn.
  int budget = 64;
  for (int ring = 0; ring <= kGrid / 2 && budget > 0; ++ring) {
    for (int y = kGrid / 2 - ring; y <= kGrid / 2 + ring && budget > 0; ++y) {
      for (int x = kGrid / 2 - ring; x <= kGrid / 2 + ring && budget > 0; ++x) {
        if (x < 0 || y < 0 || x >= kGrid || y >= kGrid ||
            (std::abs(x - kGrid / 2) != ring && std::abs(y - kGrid / 2) != ring)) {
          continue;
        }
        const int i = y * kGrid + x;
        if (sGridKnown[i]) {
          continue;
        }
        const CVector3f top((float(gx + x) + 0.5f) * kCell, (float(gy + y) + 0.5f) * kCell, eye.GetZ() + 30.f);
        const CRayCastResult hit = CGameCollision::RayStaticIntersection(mgr, top, CVector3f::Down(), 70.f, SolidFilter());
        sGrid[i] = hit.IsValid() ? hit.GetPoint().GetZ() : kNoFloor;
        sGridKnown[i] = true;
        --budget;
      }
    }
  }
  // Refresh in turn (doors open, platforms move).
  for (; budget > 0; --budget) {
    sGridKnown[sGridNext] = false;
    sGridNext = (sGridNext + 1) % (kGrid * kGrid);
  }
}

void FillGrid(fxp::Field& f) {
  f.originX = float(sGridX) * kCell;
  f.originY = float(sGridY) * kCell;
  f.cell = kCell;
  f.useGrid = true;
  std::memcpy(f.grid, sGrid, sizeof(sGrid));
}

void DrawFields(const CStateManager& mgr, const CVector3f& eye, bool captured) {
  const EEnvFxType env = EnvFx(mgr);
  const float mag = EnvMagnitude(mgr);
  const bool weather = PortDebug::ModernRain() && (env == kEFX_Rain || env == kEFX_Snow) && mag > 0.01f;
  if (weather) {
    UpdateGrid(mgr, eye);
  }
  static fxp::Field field;
  if (weather && env == kEFX_Rain) {
    field = fxp::Field{};
    field.kind = fxp::Rain;
    field.count = uint32_t(std::min(30000.f, 14000.f * mag * Density()));
    field.time = std::fmod(sNow, 3600.f);
    field.radius = 22.f;
    field.height = 18.f;
    field.wind[0] = 1.6f, field.wind[1] = 0.8f;
    field.speed = 22.f;
    field.size = 0.012f;
    field.length = 0.55f;
    field.color = 0x58b8c8d8;
    field.sceneDepth = captured;
    FillGrid(field);
    fxp::draw_field(field);
    field.kind = fxp::Splashes;
    field.count = uint32_t(std::min(6000.f, 2600.f * mag * Density()));
    field.radius = 20.f;
    field.size = 0.22f;
    field.color = 0x80d8e4f0;
    fxp::draw_field(field);
  } else if (weather && env == kEFX_Snow) {
    field = fxp::Field{};
    field.kind = fxp::Snow;
    field.count = uint32_t(std::min(20000.f, 8000.f * mag * Density()));
    field.time = std::fmod(sNow, 3600.f);
    field.radius = 20.f;
    field.height = 14.f;
    field.wind[0] = 0.5f, field.wind[1] = 0.25f;
    field.speed = 1.1f;
    field.size = 0.045f;
    field.color = 0xc8f4f8ff;
    field.sceneDepth = captured;
    FillGrid(field);
    fxp::draw_field(field);
  }
  if (!PortDebug::AmbientParticles()) {
    return;
  }
  if (NearestLava(eye, 40.f) != nullptr) {
    field = fxp::Field{};
    field.kind = fxp::Embers;
    field.count = uint32_t(900.f * Density());
    field.time = std::fmod(sNow, 3600.f);
    field.radius = 16.f;
    field.height = 12.f;
    field.wind[0] = 0.3f, field.wind[1] = 0.2f;
    field.speed = 1.f;
    field.size = 0.05f;
    field.color = 0xffff6a20;
    field.sceneDepth = captured;
    fxp::draw_field(field);
    // Ash drifting down.
    field.kind = fxp::Snow;
    field.count = uint32_t(1400.f * Density());
    field.speed = 0.5f;
    field.size = 0.03f;
    field.color = 0x90403834;
    fxp::draw_field(field);
  } else if (!weather && FluidAt(eye, true) == nullptr) {
    field = fxp::Field{};
    field.kind = fxp::Dust;
    field.count = uint32_t(1000.f * Density());
    field.time = std::fmod(sNow, 3600.f);
    field.radius = 10.f;
    field.height = 6.f;
    field.wind[0] = 0.05f, field.wind[1] = 0.03f, field.wind[2] = 0.02f;
    field.size = 0.018f;
    field.color = 0x58f0e8d8;
    field.sceneDepth = captured;
    fxp::draw_field(field);
  }
}

// ---- The finish ----
float sPrevView[12];
bool sHavePrev = false;
} // namespace

void NoteFluid(int fluidType, const CVector3f& boxMin, const CVector3f& boxMax, float surfaceZ) {
  RollFluids();
  if (sFluids.size() < 32) {
    sFluids.push_back({fluidType, boxMin, boxMax, surfaceZ});
  }
}

void ThrowPack(const CVector3f& from, const CVector3f& to, float seconds) {
  if (!PortDebug::ModernParticles()) {
    return;
  }
  seconds = std::max(seconds, 0.2f);
  constexpr float g = 9.8f;
  const CVector3f v = (to - from) * (1.f / seconds) + CVector3f(0.f, 0.f, 0.5f * g * seconds);
  // The pack: a bright green core and a softer halo, the same arc.
  Emit(from, v, seconds, 0.22f, 0xff60ff80, fxp::Glow, 0.f, g);
  Emit(from, v, seconds, 0.6f, 0x9040ff70, fxp::Glow, 0.f, g);
  // A trail: the same arc again, each a little later and shorter-lived.
  for (int i = 1; i <= 8; ++i) {
    Particle p{};
    const float delay = 0.035f * float(i);
    p.pos[0] = from.GetX(), p.pos[1] = from.GetY(), p.pos[2] = from.GetZ();
    p.t0 = sNow + delay;
    p.vel[0] = v.GetX(), p.vel[1] = v.GetY(), p.vel[2] = v.GetZ();
    p.life = std::max(0.05f, seconds - delay);
    p.size = 0.14f * (1.f - 0.08f * float(i));
    p.drag = 0.f;
    p.gravity = g;
    p.floorZ = kNoFloor;
    p.color = 0xc070ff90;
    p.flags = fxp::Glow | fxp::Shrink;
    p.seed = sRng.Unit();
    if (sParticles.size() < kMaxParticles) {
      sParticles.push_back(p);
    }
  }
  // Where it lands: a ring and sparkles, once it is there.
  for (int i = 0; i < 14; ++i) {
    Particle p{};
    p.pos[0] = to.GetX(), p.pos[1] = to.GetY(), p.pos[2] = to.GetZ();
    p.t0 = sNow + seconds;
    const CVector3f d = sRng.Dir() * sRng.Range(1.f, 2.5f) + CVector3f(0.f, 0.f, 1.f);
    p.vel[0] = d.GetX(), p.vel[1] = d.GetY(), p.vel[2] = d.GetZ();
    p.life = sRng.Range(0.5f, 0.9f);
    p.size = 0.06f;
    p.drag = 2.f;
    p.gravity = -0.5f;
    p.floorZ = kNoFloor;
    p.color = 0xff80ffa0;
    p.flags = fxp::Glow | fxp::Shrink;
    p.seed = sRng.Unit();
    if (sParticles.size() < kMaxParticles) {
      sParticles.push_back(p);
    }
  }
  Particle ring{};
  ring.pos[0] = to.GetX(), ring.pos[1] = to.GetY(), ring.pos[2] = to.GetZ();
  ring.t0 = sNow + seconds;
  ring.life = 0.45f;
  ring.size = 2.2f;
  ring.floorZ = kNoFloor;
  ring.color = 0xc060ff80;
  ring.flags = fxp::Ring;
  if (sParticles.size() < kMaxParticles) {
    sParticles.push_back(ring);
  }
}

void MeleeHit(const CVector3f& at, const CVector3f& dir, int beam) {
  if (!PortDebug::ModernParticles()) {
    return;
  }
  ShotKind kind = kPower;
  switch (beam) {
  case CPlayerState::kBI_Ice:
    kind = kIce;
    break;
  case CPlayerState::kBI_Wave:
    kind = kWave;
    break;
  case CPlayerState::kBI_Plasma:
    kind = kPlasma;
    break;
  case CPlayerState::kBI_Phazon:
    kind = kPhazon;
    break;
  default:
    break;
  }
  const Palette c = ColorsOf(kind);
  const CVector3f back = -Norm(dir);
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.1f, 1.6f, 0xffffffff, fxp::Glow);
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.3f, 2.2f, (c.core & 0x00ffffff) | 0xd0000000, fxp::Ring);
  for (int i = 0; i < 26; ++i) {
    const CVector3f v = Norm(back * 0.5f + sRng.Dir()) * sRng.Range(5.f, 11.f);
    Emit(at, v, sRng.Range(0.3f, 0.7f), 0.04f, c.spark, fxp::Spark, 1.5f, 9.8f, kNoFloor, 0.04f);
  }
}

void GrenadeTick(const CVector3f& at, float fuse, bool returned) {
  // Blinking faster toward the end of its fuse.
  const float rate = fuse < 0.8f ? 14.f : fuse < 1.6f ? 7.f : 3.5f;
  const bool on = std::fmod(sNow * rate, 1.f) < 0.5f;
  const uint32_t core = returned ? 0xff80d0ff : on ? 0xffff3020 : 0xffff9040;
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.05f, on ? 0.35f : 0.22f, core, fxp::Glow);
  Emit(at, sRng.Dir() * 0.3f, 0.35f, 0.12f, 0x50808080, fxp::Smoke, 1.f, -0.5f);
}

void Explosion(const CStateManager& mgr, const CVector3f& at, float scale) {
  Shot s{};
  s.kind = kMissile;
  s.charged = scale > 1.f;
  Impact(mgr, s, at, CVector3f(0.f, 0.f, -1.f), PortDebug::HeatDistortion());
}

void ShieldBreak(const CVector3f& at, const CVector3f& dir) {
  const CVector3f back = -Norm(dir);
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.12f, 2.f, 0xffffffff, fxp::Glow);
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.4f, 3.f, 0xd0a0e0ff, fxp::Ring);
  for (int i = 0; i < 30; ++i) {
    // Shards of the shield.
    Emit(at, Norm(back * 0.4f + sRng.Dir()) * sRng.Range(3.f, 8.f), sRng.Range(0.5f, 1.f), 0.06f, 0xe0c0f0ff,
         fxp::Flake | fxp::Shrink, 1.f, 9.8f);
  }
  for (int i = 0; i < 10; ++i) {
    Emit(at, sRng.Dir() * sRng.Range(6.f, 10.f), sRng.Range(0.08f, 0.15f), 0.05f, 0xffc0e8ff, fxp::Arc, 0.f, 0.f,
         kNoFloor, 0.07f);
  }
}

void Shatter(const CVector3f& at) {
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.15f, 2.5f, 0xffd0f0ff, fxp::Glow);
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.4f, 3.5f, 0xc0b0e8ff, fxp::Ring);
  for (int i = 0; i < 40; ++i) {
    Emit(at + sRng.Dir() * 0.3f, sRng.Dir() * sRng.Range(2.f, 7.f) + CVector3f(0.f, 0.f, 2.f), sRng.Range(0.8f, 1.5f),
         sRng.Range(0.05f, 0.12f), 0xf0e0f8ff, fxp::Flake | fxp::Shrink, 0.8f, 9.8f);
  }
}

void Rage(const CVector3f& at) {
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.3f, 3.f, 0xffff2010, fxp::Glow);
  Emit(at, CVector3f(0.f, 0.f, 0.f), 0.5f, 4.f, 0xd0ff3020, fxp::Ring);
  for (int i = 0; i < 24; ++i) {
    Emit(at, sRng.Dir() * sRng.Range(2.f, 5.f), sRng.Range(0.4f, 0.8f), 0.05f, 0xffff4020, fxp::Ember | fxp::Shrink,
         1.5f, -1.f);
  }
}

void RageAura(const CVector3f& at) {
  for (int i = 0; i < 4; ++i) {
    const CVector3f off(sRng.Range(-0.5f, 0.5f), sRng.Range(-0.5f, 0.5f), sRng.Range(-0.8f, 0.8f));
    Emit(at + off, CVector3f(0.f, 0.f, 1.2f), sRng.Range(0.4f, 0.7f), 0.06f, 0xffff3018, fxp::Ember | fxp::Shrink, 1.f,
         -0.5f);
  }
}

void NoteSky(bool drawn) { sSkyDrawn = drawn; }

void SkyTick(const CStateManager& mgr) {
  sPlanetWorld = false;
  if (const CWorld* world = mgr.GetWorld()) {
    const uint32_t id = uint32_t(world->IGetWorldAssetId());
    sPlanetWorld = id == 0x39F2DE28u || id == 0x83F6FF6Fu || id == 0xA8BE6291u;
  }
  sDomeCount = 0;
  sSkyFirst = false;
  if (sPlanetWorld && PortDebug::DayNight()) {
    const CVector3f eye = mgr.GetCameraManager()->GetCurrentCamera(mgr).GetTranslation();
    const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
    for (int idx = actors.GetFirstObjectIndex(); idx != -1 && sDomeCount < 8; idx = actors.GetNextObjectIndex(idx)) {
      const CActor* actor = TCastToConstPtr< CActor >(actors.GetObjectByIndex(idx));
      if (actor == nullptr || !actor->GetActive() || !PortRoomGeo::IsBigActor(*actor) ||
          !actor->GetModelData()->GetBounds(actor->GetTransform()).PointInside(eye)) {
        continue;
      }
      const bool dome = PortRoomGeo::IsSkyDome(*actor);
      // Each big actor around the eye once in the log, so a dome under another name can be found.
      static std::vector< uint32_t > sSeen;
      const uint32_t key = uint32_t(actor->GetEditorId().Value());
      if (std::find(sSeen.begin(), sSeen.end(), key) == sSeen.end() && sSeen.size() < 256) {
        sSeen.push_back(key);
        PortLog::Write("port: sky: big actor %08x \"%s\"%s\n", key, actor->GetDebugName().c_str(),
                       dome ? " (sky dome)" : "");
      }
      if (dome) {
        sDomes[sDomeCount++] = actor->GetUniqueId();
      }
    }
  }
  const float clock = CGraphics::GetSecondsMod900();
  float dt = sSkyClock < 0.f ? 0.f : clock - sSkyClock;
  if (dt < 0.f) {
    dt += 900.f;
  }
  dt = std::clamp(dt, 0.f, 0.1f);
  sSkyClock = clock;
  if (PortDebug::DayNight()) {
    sDayClock = std::fmod(sDayClock + dt / kDayLength, 1.f);
  }
  // The weather: hold, then move to a neighbouring state; ease toward its targets.
  sWeatherHold -= dt;
  if (sWeatherHold <= 0.f) {
    const float u = WeatherUnit();
    if (sWeatherState == 0) {
      sWeatherState = 1;
    } else if (sWeatherState == 3) {
      sWeatherState = 2;
    } else {
      sWeatherState += u < 0.5f ? 1 : -1;
    }
    sWeatherHold = 120.f + 180.f * WeatherUnit();
  }
  static float sLogIn = 0.f;
  sLogIn -= dt;
  if (sLogIn <= 0.f) {
    sLogIn = 10.f;
    PortLog::Write("port: sky: %05.2f h, planet %d, sky %d, domes %d, new sky %d, weather %d (cover %.2f rain %.2f storm %.2f)\n",
                   double(sDayClock * 24.f), int(sPlanetWorld), int(sSkyDrawn), sDomeCount, int(sSkyFirst), sWeatherState, double(sCover),
                   double(sPrecip), double(sStorm));
  }
  static const float kCover[4] = {0.3f, 0.6f, 0.82f, 0.95f};
  static const float kPrecip[4] = {0.f, 0.f, 0.6f, 1.f};
  static const float kStorm[4] = {0.f, 0.f, 0.f, 1.f};
  const float ease = 1.f - std::exp(-dt / 25.f);
  sCover += (kCover[sWeatherState] - sCover) * ease;
  sPrecip += (kPrecip[sWeatherState] - sPrecip) * ease;
  sStorm += (kStorm[sWeatherState] - sStorm) * ease;
}

bool ReplacesSkyDome(const CActor& actor) {
  if (!sSkyFirst || (sDiag & kDiagHideDomes) == 0) {
    return false;
  }
  for (int i = 0; i < sDomeCount; ++i) {
    if (sDomes[i] == actor.GetUniqueId()) {
      return true;
    }
  }
  return false;
}

bool SkyLight(float travel[3], float& intensity, float color[3]) {
  if (!DayNightOn() || (sDiag & kDiagKeyLight) == 0) {
    return false;
  }
  float dir[3];
  KeyLight(dir, intensity, color);
  for (int i = 0; i < 3; ++i) {
    travel[i] = -dir[i];
  }
  return true;
}

float DayHour() { return sDayClock * 24.f; }

void Fire(const CVector3f& at, float strength, float dt) {
  if (!PortDebug::ModernParticles() || strength <= 0.f) {
    return;
  }
  for (int i = 0, n = Count(70.f * strength * dt); i < n; ++i) {
    const CVector3f p = at + CVector3f(sRng.Range(-0.15f, 0.15f), sRng.Range(-0.15f, 0.15f), sRng.Range(0.f, 0.1f));
    Emit(p, CVector3f(sRng.Range(-0.2f, 0.2f), sRng.Range(-0.2f, 0.2f), sRng.Range(0.8f, 1.6f)), sRng.Range(0.45f, 0.8f),
         sRng.Range(0.35f, 0.65f) * (0.7f + 0.5f * strength), 0xffffffff, fxp::Flame, 1.2f, -1.5f);
  }
  for (int i = 0, n = Count(14.f * strength * dt); i < n; ++i) {
    Emit(at, CVector3f(sRng.Range(-0.6f, 0.6f), sRng.Range(-0.6f, 0.6f), sRng.Range(1.5f, 3.f)), sRng.Range(1.f, 2.2f),
         0.03f, 0xffffa040, fxp::Ember | fxp::Shrink, 0.8f, -0.6f);
  }
  for (int i = 0, n = Count(5.f * strength * dt); i < n; ++i) {
    Emit(at + CVector3f(0.f, 0.f, 0.6f), CVector3f(sRng.Range(-0.2f, 0.2f), sRng.Range(-0.2f, 0.2f), 1.2f),
         sRng.Range(2.f, 3.2f), sRng.Range(0.5f, 0.8f), 0x70302824, fxp::Smoke, 0.6f, -0.5f);
  }
}

bool ReplacesEnvFx(int envFxType) {
  return PortDebug::ModernRain() && (envFxType == kEFX_Rain || envFxType == kEFX_Snow);
}

bool WantsSceneCapture() { return PortDebug::ModernParticles() || PortDebug::HeatDistortion(); }

bool DrawSkyFirst(const CStateManager& mgr, const CTransform4f& view, const float sunTravel[3]) {
  sSkyFirst = false;
  if (!DayNightOn() || (sDiag & kDiagSkyFirst) == 0) {
    return false;
  }
  sSkyOnlyPass = true;
  DrawAtmosphere(mgr, view, sunTravel);
  sSkyOnlyPass = false;
  sSkyFirst = true;
  return true;
}

bool SkyFirstDrawn() { return sSkyFirst; }

float WorldAmbient() { return sSkyFirst && (sDiag & kDiagRelight) != 0 ? RelightAmbient() : 0.f; }

uint32_t Diag() { return sDiag; }
void SetDiag(uint32_t bits) { sDiag = bits; }

void DrawAtmosphere(const CStateManager& mgr, const CTransform4f& view, const float sunTravel[3]) {
  RollFluids();
  if (!NormalView(mgr)) {
    return;
  }
  const CVector3f eye = view.GetTranslation();
  const float sunLen = std::sqrt(sunTravel[0] * sunTravel[0] + sunTravel[1] * sunTravel[1] + sunTravel[2] * sunTravel[2]);
  const EEnvFxType env = EnvFx(mgr);
  const float mag = EnvMagnitude(mgr);

  // The area's fog: its colour (the in-scatter's) and how far it reaches.
  CVector3f fogColor(0.45f, 0.5f, 0.55f);
  float areaBase = eye.GetZ() - 3.f;
  float areaHeight = 20.f;
  if (mgr.GetWorld() != nullptr && mgr.GetNextAreaId() != kInvalidAreaId) {
    const CGameArea& area = mgr.GetWorld()->GetAreaAlways(mgr.GetNextAreaId());
    areaBase = area.GetAABB().GetMinPoint().GetZ() + 0.5f;
    areaHeight = std::max(4.f, area.GetAABB().GetMaxPoint().GetZ() - area.GetAABB().GetMinPoint().GetZ());
    if (area.IsPostConstructed() && area.GetAreaFog() != nullptr && !area.GetAreaFog()->IsFogDisabled()) {
      fogColor = area.GetAreaFog()->PortColor();
    }
  }
  const CVector3f fogLinear(SrgbToLinear(fogColor.GetX()), SrgbToLinear(fogColor.GetY()), SrgbToLinear(fogColor.GetZ()));

  GXPortPostParams p;
  // The sky and its clouds (port_fx.h; post.cpp's Clouds mode): the time of day's sky over the
  // painted one, and volumetric clouds lit by the sun or the moon.
  const bool dayNight = DayNightOn();
  if (!sSkyFirst && ((PortDebug::VolumetricClouds() && sSkyDrawn) || dayNight)) {
    FillCommon(p, view);
    p.misc[0] = 1;
    const bool rain = env == kEFX_Rain;
    float sun[3] = {0.f, 0.f, 1.f}, moon[3] = {0.f, 0.f, -1.f}, elev = 0.6f;
    float sunI = 0.3f, moonI = 0.f;
    if (dayNight) {
      SkyBodies(sun, moon, elev);
      sunI = std::clamp((elev + 0.05f) / 0.2f, 0.f, 1.f);
      moonI = std::clamp(moon[2] / 0.15f, 0.f, 1.f);
    } else if (sunLen > 0.f) {
      for (int i = 0; i < 3; ++i) {
        sun[i] = -sunTravel[i] / sunLen;
      }
      sunI = 1.f;
      elev = sun[2];
    }
    const float warm = 1.f - std::clamp(elev / 0.35f, 0.f, 1.f);
    float* l0 = p.lights[0];
    l0[0] = sun[0], l0[1] = sun[1], l0[2] = sun[2], l0[3] = sunI * (1.f - 0.5f * sStorm);
    l0[4] = 1.f, l0[5] = 0.95f - 0.45f * warm, l0[6] = 0.88f - 0.68f * warm, l0[7] = elev;
    float* l1 = p.lights[1];
    l1[0] = moon[0], l1[1] = moon[1], l1[2] = moon[2], l1[3] = moonI * 0.3f;
    l1[4] = 0.55f, l1[5] = 0.65f, l1[6] = 0.9f, l1[7] = 0.f;
    float* l2 = p.lights[2];
    // 3: the whole view, drawn first behind the world; 1: over the painted sky above the horizon.
    l2[0] = sSkyOnlyPass ? ((sDiag & kDiagMagenta) != 0 ? 4.f : 3.f) : !dayNight ? 0.f : 1.f;
    l2[1] = dayNight ? std::clamp(-elev / 0.15f, 0.f, 1.f) * (1.f - sCover) : 0.f;
    l2[2] = PortDebug::VolumetricClouds() ? 1.f : 0.f;
    l2[3] = rain ? sFlash : 0.f;
    // Big clouds: their base some 350 m over the eye, up to 900 m thick, drifting.
    p.clouds[0] = WeatherOn() ? sCover : rain ? 0.72f + 0.22f * mag : 0.5f;
    p.clouds[1] = 350.f;
    p.clouds[2] = 900.f;
    p.clouds[3] = 8.f;
    const float tint = 0.15f;
    p.cloudColor[0] = (1.f - tint) + fogLinear.GetX() * tint;
    p.cloudColor[1] = (1.f - tint) + fogLinear.GetY() * tint;
    p.cloudColor[2] = (1.f - tint) + fogLinear.GetZ() * tint;
    const float storm = WeatherOn() ? sStorm : rain ? mag : 0.f;
    p.cloudColor[0] *= 1.f - 0.45f * storm, p.cloudColor[1] *= 1.f - 0.43f * storm, p.cloudColor[2] *= 1.f - 0.4f * storm;
    p.cloudColor[3] = 0.9f + 0.6f * storm;
    GXPortPost(&p);
  }
  if (sSkyOnlyPass) {
    return;
  }

  // The fog, the caustics, the wet or snowy ground.
  FillCommon(p, view);
  p.misc[0] = 0;
  bool any = false;
  const int fogAmount = PortDebug::VolumetricFogAmount();
  const Fluid* under = FluidAt(eye, true);
  if (fogAmount > 0) {
    const float amount = float(fogAmount) / 100.f;
    p.fog[0] = 0.012f * amount;
    p.fog[1] = 1.f / (0.35f * areaHeight);
    p.fog[2] = areaBase;
    p.fog[3] = 0.6f;
    p.fogColor[0] = fogLinear.GetX() * 0.6f;
    p.fogColor[1] = fogLinear.GetY() * 0.6f;
    p.fogColor[2] = fogLinear.GetZ() * 0.6f;
    p.fogColor[3] = 0.35f;
    if (under != nullptr && PortDebug::UnderwaterEffects()) {
      p.fog[0] = 0.04f * amount;
      p.fog[1] = 0.f;
      p.fogColor[0] = 0.02f, p.fogColor[1] = 0.09f, p.fogColor[2] = 0.11f;
    }
    if (env == kEFX_Rain) {
      p.fog[0] *= 1.f + mag;
      // The flash lights the air too.
      p.fogColor[0] += sFlash * 0.12f, p.fogColor[1] += sFlash * 0.14f, p.fogColor[2] += sFlash * 0.18f;
    }
    if (sunLen > 0.f) {
      for (int i = 0; i < 3; ++i) {
        p.sun[i] = -sunTravel[i] / sunLen;
      }
      p.sun[3] = 1.f;
      p.sunColor[0] = 0.55f, p.sunColor[1] = 0.5f, p.sunColor[2] = 0.42f;
      if (DayNightOn() && (sDiag & kDiagFogTint) != 0) {
        float d[3], inten, col[3];
        KeyLight(d, inten, col);
        p.sunColor[0] = 0.45f * col[0] * inten, p.sunColor[1] = 0.45f * col[1] * inten, p.sunColor[2] = 0.45f * col[2] * inten;
      }
    }
    p.sunColor[3] = std::min(CGraphics::GetProjectionState().GetFar(), 140.f);
    // The nearest lights (shots, lamps, glows) scatter in it.
    struct Near {
      float d;
      const CLight* l;
    };
    Near nearest[8];
    int count = 0;
    const rstl::vector< CLight >& lights = mgr.GetDynamicLightList();
    for (int i = 0; i < lights.size(); ++i) {
      const CLight& l = lights[i];
      if (l.GetType() != kLT_Point && l.GetType() != kLT_Custom) {
        continue;
      }
      const float r = l.GetRadius();
      const float d = Len(l.GetPosition() - eye);
      if (r <= 0.05f || d - r > 60.f) {
        continue;
      }
      if (count < 8) {
        nearest[count++] = {d, &l};
      } else {
        int far = 0;
        for (int k = 1; k < 8; ++k) {
          if (nearest[k].d > nearest[far].d) {
            far = k;
          }
        }
        if (d < nearest[far].d) {
          nearest[far] = {d, &l};
        }
      }
    }
    for (int i = 0; i < count; ++i) {
      const CLight& l = *nearest[i].l;
      float* rec = p.lights[i];
      rec[0] = l.GetPosition().GetX(), rec[1] = l.GetPosition().GetY(), rec[2] = l.GetPosition().GetZ();
      rec[3] = std::min(l.GetRadius(), 30.f);
      rec[4] = SrgbToLinear(l.GetColor().GetRed());
      rec[5] = SrgbToLinear(l.GetColor().GetGreen());
      rec[6] = SrgbToLinear(l.GetColor().GetBlue());
      rec[7] = 0.6f;
    }
    p.misc[1] = u32(count);
    any = true;
  }
  if (PortDebug::UnderwaterEffects()) {
    // The water to light: the one the eye is in, else the nearest in view.
    const Fluid* water = under;
    float best = 40.f;
    if (water == nullptr) {
      for (const Fluid& f : sFluidsLast) {
        if (f.type != 0 && f.type != 1) {
          continue;
        }
        const float d = BoxDistanceXY(f, eye);
        if (d < best) {
          best = d;
          water = &f;
        }
      }
    }
    if (water != nullptr) {
      p.water[0] = water->surface;
      p.water[1] = 1.f;
      p.water[2] = under != nullptr ? 1.f : 0.f;
      p.water[3] = 7.f;
      p.waterBox[0] = water->min.GetX(), p.waterBox[1] = water->min.GetY();
      p.waterBox[2] = water->max.GetX(), p.waterBox[3] = water->max.GetY();
      any = true;
    }
  }
  if (PortDebug::ModernRain() && env == kEFX_Rain && mag > 0.01f && under == nullptr) {
    p.weather[0] = mag;
    any = true;
  }
  if (PortDebug::SnowFootprints() && env == kEFX_Snow && mag > 0.01f) {
    p.weather[2] = std::min(1.f, 0.4f + mag);
    any = true;
  }
  if (any) {
    GXPortPost(&p);
  }
}

void DrawParticles(const CStateManager& mgr, const CTransform4f& view, bool sceneCaptured) {
  const int frame = CGraphics::GetFrameCounter();
  const bool step = frame != sLastFrame;
  const CVector3f eye = view.GetTranslation();
  const bool haze = PortDebug::HeatDistortion() && sceneCaptured;
  if (step) {
    sLastFrame = frame;
    const float clock = CGraphics::GetSecondsMod900();
    float dt = sLastClock < 0.f ? 0.f : clock - sLastClock;
    if (dt < 0.f) {
      dt += 900.f;
    }
    sDt = std::clamp(dt, 0.f, 0.1f);
    sLastClock = clock;
    sNow += sDt;
    // The dead ones go.
    sParticles.erase(std::remove_if(sParticles.begin(), sParticles.end(),
                                    [](const Particle& q) { return sNow - q.t0 > q.life; }),
                     sParticles.end());
    if (PortDebug::ModernParticles()) {
      UpdateShots(mgr, haze);
      UpdateCharge(mgr, view);
    } else {
      sShots.clear();
      sPulse = 0.f;
    }
    if (sDt > 0.f) {
      UpdateLightning(mgr, view);
      UpdateUnderwater(mgr, eye);
      UpdateHeat(eye, haze);
      UpdatePrints(mgr);
    }
  }
  // The thermal and X-ray visors see none of it (the shots are still followed).
  if (!NormalView(mgr)) {
    return;
  }
  CGraphics::SetViewPointMatrix(view);
  CGraphics::SetModelMatrix(CTransform4f::Identity());
  CGraphics::SetDepthRange(0.125f, 1.f);
  // The particles (the heat haze among them) first, so the haze doesn't hide the weather.
  if (!sParticles.empty()) {
    fxp::Frame f;
    f.time = sNow;
    f.anchor[0] = sAnchor.GetX(), f.anchor[1] = sAnchor.GetY(), f.anchor[2] = sAnchor.GetZ();
    f.anchorPulse = sPulse;
    f.sceneDepth = sceneCaptured;
    fxp::draw(f, sParticles.data(), uint32_t(sParticles.size()));
  }
  DrawFields(mgr, eye, sceneCaptured);
}

void DrawFinish(const CStateManager& mgr, const CTransform4f& view) {
  float rows[12];
  ViewRows(view, rows);
  if (!NormalView(mgr)) {
    std::memcpy(sPrevView, rows, sizeof(rows));
    sHavePrev = true;
    return;
  }
  GXPortPostParams p;
  FillCommon(p, view);
  p.misc[0] = 2;
  // The time of day's light over the world outdoors.
  if (DayNightOn() && (sDiag & kDiagGrade) != 0) {
    DayGrade(p.lights[0]);
    p.lights[0][3] = 1.f;
  }
  if (sHavePrev) {
    std::memcpy(p.prevView, sPrevView, sizeof(sPrevView));
  }
  std::memcpy(sPrevView, rows, sizeof(rows));
  sHavePrev = true;
  bool any = DayNightOn();
  // A perfect dodge's slowing tints the screen (port_enemy_ai.h).
  const float slow = PortEnemyAi::SlowMotion();
  if (slow > 0.f) {
    p.misc[3] = u32(slow * 1000.f);
    any = true;
  }
  if (PortDebug::MotionBlur() > 0) {
    p.finish[0] = float(PortDebug::MotionBlur()) / 100.f;
    any = true;
  }
  if (PortDebug::DepthOfField()) {
    const CPlayer* player = mgr.GetPlayer();
    if (mgr.GetCameraManager()->IsInCinematicCamera()) {
      p.finish[1] = 1.f;
      p.finish[2] = -1.f; // focus on what is at the screen's centre
      p.finish[3] = 6.f;
      any = true;
    } else if (player != nullptr && player->GetPlayerScanState() == CPlayer::kSS_Scanning &&
               player->GetScanningObjectId() != kInvalidUniqueId) {
      if (const CActor* target = TCastToConstPtr< CActor >(mgr.GetObjectById(player->GetScanningObjectId()))) {
        p.finish[1] = 0.6f;
        p.finish[2] = Len(target->GetTranslation() - view.GetTranslation());
        p.finish[3] = std::max(2.f, p.finish[2] * 0.35f);
        any = true;
      }
    }
  }
  if (any) {
    GXPortPost(&p);
  }
}

namespace {
u32 sFsrMode = 0;
float sFsrJitter[2] = {};
float sFsrPrevView[12] = {};
bool sFsrHavePrev = false;
float sFsrClock = -1.f;
} // namespace

void FsrBegin() {
  sFsrMode = static_cast< u32 >(PortDebug::UpscalingActive());
  if (sFsrMode == 0) {
    sFsrHavePrev = false;
    return;
  }
  GXPortFsrJitter(static_cast< u64 >(CGraphics::GetFrameCounter()), sFsrMode, &sFsrJitter[0], &sFsrJitter[1]);
  GXPortSetJitter(GX_TRUE, sFsrJitter[0], sFsrJitter[1]);
}

float LightningFlash() { return sFlash; }

void FsrSolid() {
  if (sFsrMode != 0) {
    GXPortFsrCaptureSolid();
  }
}

void FsrEnd(const CTransform4f& view) {
  if (sFsrMode == 0) {
    return;
  }
  GXPortSetJitter(GX_FALSE, 0.f, 0.f);
  GXPortFsrParams p;
  std::memset(&p, 0, sizeof(p));
  const CGraphics::CProjectionState& proj = CGraphics::GetProjectionState();
  const float zNear = proj.GetNear();
  p.frustum[0] = proj.GetLeft() / zNear;
  p.frustum[1] = proj.GetRight() / zNear;
  p.frustum[2] = proj.GetBottom() / zNear;
  p.frustum[3] = proj.GetTop() / zNear;
  p.depth[0] = zNear;
  p.depth[1] = proj.GetFar();
  p.depth[2] = 0.125f;
  p.depth[3] = 1.f;
  ViewRows(view, p.view);
  // A cut (the first frame, a jump of the eye or a sudden turn) starts FSR's history over.
  bool reset = !sFsrHavePrev;
  if (!reset) {
    const float* a = p.view;
    const float* b = sFsrPrevView;
    const float dx = a[3] - b[3], dy = a[7] - b[7], dz = a[11] - b[11];
    const float rightDot = a[0] * b[0] + a[4] * b[4] + a[8] * b[8];
    const float upDot = a[1] * b[1] + a[5] * b[5] + a[9] * b[9];
    reset = dx * dx + dy * dy + dz * dz > 16.f || rightDot < 0.8f || upDot < 0.8f;
  }
  std::memcpy(p.prevView, reset ? p.view : sFsrPrevView, sizeof(p.prevView));
  std::memcpy(sFsrPrevView, p.view, sizeof(p.view));
  sFsrHavePrev = true;
  p.jitter[0] = sFsrJitter[0];
  p.jitter[1] = sFsrJitter[1];
  const float clock = CGraphics::GetSecondsMod900();
  float dt = sFsrClock < 0.f ? 1.f / 60.f : clock - sFsrClock;
  if (dt < 0.f) {
    dt += 900.f;
  }
  sFsrClock = clock;
  p.frameMs = std::clamp(dt, 0.001f, 0.1f) * 1000.f;
  p.vfov = 2.f * std::atan((proj.GetTop() - proj.GetBottom()) * 0.5f / zNear);
  p.mode = sFsrMode;
  p.reset = reset ? 1 : 0;
  GXPortFsrCapture(&p);
}

void Reset() {
  sFsrHavePrev = false;
  sParticles.clear();
  sShots.clear();
  sFluids.clear();
  sFluidsLast.clear();
  sGridX = sGridY = INT32_MIN;
  sHaveStep = false;
  sHavePrev = false;
  sPulse = 0.f;
}
} // namespace PortFx
