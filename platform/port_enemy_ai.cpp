// Enemy behaviour and the cannon strike (port_enemy_ai.h).
#include "port_enemy_ai.h"

#include "port_debug.h"
#include "port_fx.h"
#include "port_log.h"

#include "Collision/CMaterialFilter.hpp"
#include "Collision/CRayCastResult.hpp"
#include "Kyoto/Input/CFinalInput.hpp"
#include "Kyoto/Math/CRelAngle.hpp"
#include "MetroidPrime/BodyState/CBodyController.hpp"
#include "MetroidPrime/BodyState/CBodyStateCmdMgr.hpp"
#include "MetroidPrime/CDamageInfo.hpp"
#include "MetroidPrime/CDamageVulnerability.hpp"
#include "MetroidPrime/CHealthInfo.hpp"
#include "MetroidPrime/CGameCollision.hpp"
#include "MetroidPrime/CObjectList.hpp"
#include "MetroidPrime/CStateManager.hpp"
#include "MetroidPrime/Cameras/CCameraManager.hpp"
#include "MetroidPrime/Cameras/CFirstPersonCamera.hpp"
#include "MetroidPrime/Enemies/CPatterned.hpp"
#include "MetroidPrime/Enemies/CSeedling.hpp"
#include "MetroidPrime/Enemies/CWallWalker.hpp"
#include "MetroidPrime/Player/CPlayer.hpp"
#include "MetroidPrime/Player/CPlayerState.hpp"
#include "MetroidPrime/Player/CStaticInterference.hpp"
#include "MetroidPrime/TCastTo.hpp"
#include "MetroidPrime/Tweaks/CTweakPlayerGun.hpp"
#include "MetroidPrime/Weapons/CGameProjectile.hpp"
#include "MetroidPrime/Weapons/CProjectileInfo.hpp"

#include <dolphin/pad.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_keyboard.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace PortEnemyAi {
namespace {
constexpr float kPi = 3.14159265f;

struct Ai {
  uint32_t lastSeen = 0;
  float maxHp = 0.f;
  float lastHp = -1.f;
  // Running from Samus.
  bool fled = false;
  float fleeT = 0.f;
  // Sidestepping a shot.
  float dodgeCd = 0.f;
  float dashT = 0.f;
  CVector3f dashDir = CVector3f::Zero();
  TUniqueId lastThreat = kInvalidUniqueId;
  // A pirate charging to strike.
  float chargeT = 0.f;
  float chargeCd = 0.f;
  bool struck = false;
  float strikeT = 0.f;
  // Health packs.
  float healCd = 0.f;
  float pendingHeal = 0.f;
  float healAmount = 0.f;
  // A Seedling on Samus' visor.
  bool latched = false;
  float latchCd = 0.f;
  bool hadSolid = false;
  bool hadCharacter = false;
  // Hard+: its health raised once.
  bool hpScaled = false;
  // A pirate's partner, and its fury once the partner is dead.
  TUniqueId partner = kInvalidUniqueId;
  bool rage = false;
  float rageFx = 0.f;
  // Cover: 0 none, 1 moving to it, 2 hiding, 3 peeking out.
  int coverPhase = 0;
  float coverT = 0.f;
  float coverCd = 0.f;
  int coverCycles = 0;
  CVector3f coverPos = CVector3f::Zero();
  CVector3f peekPos = CVector3f::Zero();
  // Flanking: the bearing around Samus it heads for.
  float flankT = 0.f;
  float flankCd = 0.f;
  float flankBearing = 0.f;
  // Grenades.
  float grenadeCd = -1.f;
};
std::unordered_map< ushort, Ai > sAi;
uint32_t sCounter = 0;

Ai& StateOf(const CPatterned& self) {
  Ai& a = sAi[self.GetUniqueId().value];
  a.lastSeen = sCounter;
  return a;
}

void Prune() {
  if (sAi.size() < 384) {
    return;
  }
  for (auto it = sAi.begin(); it != sAi.end();) {
    if (sCounter - it->second.lastSeen > 6000) {
      it = sAi.erase(it);
    } else {
      ++it;
    }
  }
}

float Len(const CVector3f& v) { return v.Magnitude(); }
CVector3f Flat(const CVector3f& v) { return CVector3f(v.GetX(), v.GetY(), 0.f); }
CVector3f Norm(const CVector3f& v) {
  const float l = v.Magnitude();
  return l > 1e-5f ? v * (1.f / l) : CVector3f(0.f, 1.f, 0.f);
}

bool Hard() { return PortDebug::HardPlus(); }
bool Smart() { return PortDebug::SmarterEnemies() || Hard(); }
float Rand(CStateManager& mgr) { return mgr.Random()->Float(); }
float RandRange(CStateManager& mgr, float a, float b) { return a + (b - a) * Rand(mgr); }
CVector3f EyeOf(const CPlayer& player) { return player.GetTranslation() + CVector3f(0.f, 0.f, player.GetEyeHeight()); }

const CMaterialFilter& SolidFilter() {
  static const CMaterialFilter filter = CMaterialFilter::MakeIncludeExclude(
      CMaterialList(kMT_Solid), CMaterialList(kMT_ProjectilePassthrough, kMT_SeeThrough));
  return filter;
}

bool Blocked(const CStateManager& mgr, const CVector3f& from, const CVector3f& to) {
  const CVector3f d = to - from;
  const float len = Len(d);
  if (len < 1e-3f) {
    return false;
  }
  const CRayCastResult hit = CGameCollision::RayStaticIntersection(mgr, from, d * (1.f / len), len, SolidFilter());
  return hit.IsValid();
}

bool IsBoss(CPatterned::EPatternedAI kind) {
  switch (kind) {
  case CPatterned::kC_Flaahgra:
  case CPatterned::kC_FlaahgraTentacle:
  case CPatterned::kC_MetroidPrimeExo:
  case CPatterned::kC_MetroidPrimeStage2:
  case CPatterned::kC_NewIntroBoss:
  case CPatterned::kC_Ridley:
  case CPatterned::kC_Thardus:
  case CPatterned::kC_ThardusRockProjectile:
    return true;
  default:
    return false;
  }
}

// The enemies that walk or fly freely (dodge and flee make sense for them).
bool Mobile(CPatterned::EPatternedAI kind) {
  switch (kind) {
  case CPatterned::kC_SpacePirate:
  case CPatterned::kC_FlyingPirate:
  case CPatterned::kC_Beetle:
  case CPatterned::kC_Babygoth:
  case CPatterned::kC_WarWasp:
  case CPatterned::kC_Puffer:
  case CPatterned::kC_Drone:
    return true;
  default:
    return false;
  }
}

bool Pirate(const CPatterned& p) { return p.GetCharacterType() == CPatterned::kC_SpacePirate; }

float HpOf(CPatterned& self, CStateManager& mgr) {
  const CHealthInfo* h = self.HealthInfo(mgr);
  return h != nullptr ? h->GetHP() : 0.f;
}

// The nearest other living space pirate within `range` (nullptr for none).
CPatterned* NearestPirate(CPatterned& self, CStateManager& mgr, float range) {
  CPatterned* best = nullptr;
  float bestD = range;
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CEntity* e = actors.GetObjectByIndex(idx);
    if (e == nullptr || e->GetUniqueId() == self.GetUniqueId()) {
      continue;
    }
    CPatterned* p = TCastToPtr< CPatterned >(mgr.ObjectById(e->GetUniqueId()));
    if (p == nullptr || !Pirate(*p) || !p->IsAlive() || !p->GetActive()) {
      continue;
    }
    const float d = Len(p->GetTranslation() - self.GetTranslation());
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  return best;
}

// Overrides the body's commands for this tick: move along `move`, face `face`.
void Steer(CPatterned& self, const CVector3f& move, const CVector3f& face) {
  CBodyStateCmdMgr& cmds = self.BodyCtrl()->CommandMgr();
  cmds.Reset();
  cmds.DeliverCmd(CBCLocomotionCmd(move, face, 1.f));
}

// ---- Dodging ----
// A player shot that will pass close in the next moments: the side to step to (zero for none).
CVector3f Threat(CPatterned& self, CStateManager& mgr, TUniqueId& shotId) {
  const TUniqueId playerId = mgr.GetPlayer() != nullptr ? mgr.GetPlayer()->GetUniqueId() : kInvalidUniqueId;
  const CVector3f center = self.GetTranslation() + CVector3f(0.f, 0.f, 1.f);
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CGameProjectile* proj = TCastToConstPtr< CGameProjectile >(actors.GetObjectByIndex(idx));
    if (proj == nullptr || !proj->GetActive() || proj->GetOwnerId() != playerId) {
      continue;
    }
    const CVector3f step = proj->GetVelocity();
    if (Len(step) < 1e-4f) {
      continue;
    }
    const CVector3f dir = Norm(step);
    const CVector3f rel = center - proj->GetTranslation();
    const float along = CVector3f::Dot(rel, dir);
    if (along < 0.5f || along > 14.f) {
      continue;
    }
    const CVector3f perp = rel - dir * along;
    if (Len(perp) > 1.7f) {
      continue;
    }
    CVector3f side = Norm(CVector3f::Cross(dir, CVector3f(0.f, 0.f, 1.f)));
    if (CVector3f::Dot(perp, side) < 0.f) {
      side = -side;
    }
    shotId = proj->GetUniqueId();
    return side;
  }
  return CVector3f::Zero();
}

void Dodge(CPatterned& self, Ai& a, float dt, CStateManager& mgr) {
  if (a.dashT > 0.f) {
    a.dashT -= dt;
    const CVector3f pos = self.GetTranslation();
    const CVector3f next = pos + a.dashDir * (8.f * dt);
    const CVector3f waist(0.f, 0.f, 0.8f);
    if (Blocked(mgr, pos + waist, next + waist + a.dashDir * 0.6f)) {
      a.dashT = 0.f;
      return;
    }
    if (!self.PortIsFlyer()) {
      const CRayCastResult floor = CGameCollision::RayStaticIntersection(
          mgr, next + CVector3f(0.f, 0.f, 1.f), CVector3f::Down(), 2.6f, SolidFilter());
      if (!floor.IsValid()) {
        a.dashT = 0.f;
        return;
      }
    }
    self.SetTranslation(next);
    return;
  }
  if (a.dodgeCd > 0.f) {
    a.dodgeCd -= dt;
    return;
  }
  TUniqueId shot = kInvalidUniqueId;
  const CVector3f side = Threat(self, mgr, shot);
  if (shot == kInvalidUniqueId || shot == a.lastThreat) {
    return;
  }
  a.lastThreat = shot;
  // Not every time: about half the shots (more when enraged or on Hard+).
  const float chance = a.rage ? 0.85f : Hard() ? 0.75f : 0.55f;
  if (Rand(mgr) >= chance) {
    return;
  }
  // Room to step into: no wall, ground under it.
  const CVector3f waist(0.f, 0.f, 0.8f);
  const CVector3f target = self.GetTranslation() + side * 3.f;
  if (Blocked(mgr, self.GetTranslation() + waist, target + waist)) {
    return;
  }
  if (!self.PortIsFlyer() &&
      !CGameCollision::RayStaticIntersection(mgr, target + CVector3f(0.f, 0.f, 1.f), CVector3f::Down(), 2.6f,
                                             SolidFilter())
           .IsValid()) {
    return;
  }
  a.dodgeCd = a.rage ? 1.f : 1.6f;
  if (Pirate(self)) {
    const bool right = CVector3f::Dot(side, self.GetTransform().GetRight()) > 0.f;
    self.BodyCtrl()->CommandMgr().DeliverCmd(CBCStepCmd(right ? pas::kSD_Right : pas::kSD_Left, pas::kStep_Dodge));
  } else {
    a.dashDir = side;
    a.dashT = 0.28f;
  }
}

// ---- Fleeing ----
void Flee(CPatterned& self, Ai& a, float dt, CStateManager& mgr) {
  a.fleeT -= dt;
  const CVector3f away = self.GetTranslation() - mgr.GetPlayer()->GetTranslation();
  if (Len(Flat(away)) > 24.f) {
    a.fleeT = 0.f;
    return;
  }
  CVector3f dir = Norm(self.PortIsFlyer() ? away + CVector3f(0.f, 0.f, 0.3f * Len(away)) : Flat(away));
  // A wall ahead: run along it.
  const CVector3f waist(0.f, 0.f, 0.9f);
  if (Blocked(mgr, self.GetTranslation() + waist, self.GetTranslation() + waist + dir * 1.8f)) {
    const CVector3f along = Norm(CVector3f::Cross(dir, CVector3f(0.f, 0.f, 1.f)));
    dir = Blocked(mgr, self.GetTranslation() + waist, self.GetTranslation() + waist + along * 1.8f) ? -along : along;
  }
  Steer(self, dir, dir);
}

// ---- Pirates: the partner's charge, health packs ----
float MeleeDamage(CPatterned& self) {
  float shot = 10.f;
  if (CProjectileInfo* info = self.ProjectileInfo()) {
    shot = info->GetDamage().GetDamage();
  }
  return 2.f * std::max(shot, 4.f);
}

void Charge(CPatterned& self, Ai& a, float dt, CStateManager& mgr) {
  a.chargeT -= dt;
  CPlayer& player = *mgr.Player();
  const CVector3f to = player.GetTranslation() - self.GetTranslation();
  const float dist = Len(Flat(to));
  const CVector3f dir = Norm(Flat(to));
  if (a.struck) {
    // The blow lands partway through the swing, if Samus is still in reach.
    if (a.strikeT > 0.f) {
      a.strikeT -= dt;
      if (a.strikeT <= 0.f && dist < 3.6f && CVector3f::Dot(self.GetTransform().GetForward(), dir) > 0.2f) {
        const CDamageInfo info(CWeaponMode(kWT_AI), MeleeDamage(self) * (a.rage ? 1.5f : 1.f), 0.f, 12.f);
        mgr.ApplyDamage(self.GetUniqueId(), player.GetUniqueId(), self.GetUniqueId(), info,
                        CMaterialFilter::MakeIncludeExclude(CMaterialList(kMT_Solid), CMaterialList()), dir * 12.f);
      }
    }
    return;
  }
  if (dist > 45.f) {
    a.chargeT = 0.f;
    return;
  }
  if (dist > 2.7f) {
    Steer(self, dir, dir);
    return;
  }
  // In reach: the rifle's blade.
  CBodyStateCmdMgr& cmds = self.BodyCtrl()->CommandMgr();
  cmds.Reset();
  cmds.DeliverCmd(CBCMeleeAttackCmd(pas::kS_One, player.GetTranslation() + CVector3f(0.f, 0.f, 1.f)));
  a.struck = true;
  a.strikeT = 0.45f;
  a.chargeT = std::min(a.chargeT, 1.4f);
}

void Hurt(CPatterned& self, CStateManager& mgr) {
  if (!Pirate(self)) {
    return;
  }
  CPatterned* buddy = NearestPirate(self, mgr, 30.f);
  if (buddy == nullptr) {
    return;
  }
  Ai& b = StateOf(*buddy);
  if (b.chargeT > 0.f || b.fleeT > 0.f || b.chargeCd > 0.f) {
    return;
  }
  b.chargeT = 7.f;
  b.chargeCd = 9.f;
  b.struck = false;
  buddy->BodyCtrl()->SetLocomotionType(pas::kLT_Combat);
}

void Heal(CPatterned& self, Ai& a, float dt, CStateManager& mgr) {
  CHealthInfo* h = self.HealthInfo(mgr);
  if (h == nullptr) {
    return;
  }
  if (a.pendingHeal > 0.f) {
    a.pendingHeal -= dt;
    if (a.pendingHeal <= 0.f) {
      h->SetHP(std::min(a.maxHp, h->GetHP() + a.healAmount));
      a.lastHp = h->GetHP();
    }
    return;
  }
  if (a.healCd > 0.f) {
    a.healCd -= dt;
    return;
  }
  if (h->GetHP() > 0.5f * a.maxHp) {
    return;
  }
  CPatterned* buddy = NearestPirate(self, mgr, 18.f);
  if (buddy == nullptr) {
    return;
  }
  const CVector3f from = buddy->GetTranslation() + CVector3f(0.f, 0.f, 1.5f);
  const CVector3f to = self.GetTranslation() + CVector3f(0.f, 0.f, 1.2f);
  if (Blocked(mgr, from, to)) {
    return;
  }
  const float t = std::clamp(Len(to - from) / 14.f + 0.45f, 0.5f, 1.3f);
  PortFx::ThrowPack(from, to, t);
  a.pendingHeal = t;
  a.healAmount = 0.35f * a.maxHp;
  a.healCd = 10.f;
  // The thrower turns to its partner.
  buddy->BodyCtrl()->CommandMgr().DeliverCmd(CBCLocomotionCmd(CVector3f::Zero(), Norm(Flat(to - from)), 1.f));
}

// ---- Grenades (pirates throw them; the cannon strike knocks them back) ----
struct Grenade {
  CVector3f pos;
  CVector3f vel;
  float fuse;
  float damage;
  TUniqueId thrower;
  bool returned;
};
std::vector< Grenade > sGrenades;

void Throw(CPatterned& self, CStateManager& mgr) {
  CPlayer& player = *mgr.Player();
  const CVector3f from = self.GetTranslation() + CVector3f(0.f, 0.f, 1.8f) + self.GetTransform().GetForward() * 0.5f;
  // Where Samus will be, roughly.
  const CVector3f target = player.GetTranslation() + player.GetVelocityWR() * 0.4f;
  const float dist = Len(Flat(target - from));
  const float t = std::clamp(dist / 12.f, 0.8f, 1.6f);
  constexpr float g = 14.f;
  Grenade gr;
  gr.pos = from;
  gr.vel = (target - from) * (1.f / t) + CVector3f(0.f, 0.f, 0.5f * g * t);
  gr.fuse = t + 0.9f;
  gr.damage = MeleeDamage(self) * (Hard() ? 1.5f : 1.f);
  gr.thrower = self.GetUniqueId();
  gr.returned = false;
  if (sGrenades.size() < 16) {
    sGrenades.push_back(gr);
  }
  // The throw: it turns to Samus.
  self.BodyCtrl()->CommandMgr().DeliverCmd(
      CBCLocomotionCmd(CVector3f::Zero(), Norm(Flat(player.GetTranslation() - self.GetTranslation())), 1.f));
}

void Explode(const Grenade& gr, CStateManager& mgr) {
  PortFx::Explosion(mgr, gr.pos, gr.returned ? 1.3f : 1.f);
  CPlayer& player = *mgr.Player();
  const CMaterialFilter filter = CMaterialFilter::MakeIncludeExclude(CMaterialList(kMT_Solid), CMaterialList());
  constexpr float radius = 4.5f;
  if (!gr.returned) {
    const CVector3f d = player.GetTranslation() + CVector3f(0.f, 0.f, 1.f) - gr.pos;
    const float dist = Len(d);
    if (dist < radius && !Blocked(mgr, gr.pos + CVector3f(0.f, 0.f, 0.2f), player.GetTranslation() + CVector3f(0.f, 0.f, 1.f))) {
      const CDamageInfo info(CWeaponMode(kWT_AI), gr.damage * (1.f - 0.5f * dist / radius), 0.f, 14.f);
      const TUniqueId from = mgr.GetObjectById(gr.thrower) != nullptr ? gr.thrower : player.GetUniqueId();
      mgr.ApplyDamage(from, player.GetUniqueId(), from, info, filter, Norm(d) * 14.f);
    }
    return;
  }
  // Knocked back: it hurts the enemies around it.
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  const CDamageInfo info(CWeaponMode(kWT_Missile), 45.f, 0.f, 15.f);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1; idx = actors.GetNextObjectIndex(idx)) {
    const CEntity* e = actors.GetObjectByIndex(idx);
    if (e == nullptr) {
      continue;
    }
    CPatterned* p = TCastToPtr< CPatterned >(mgr.ObjectById(e->GetUniqueId()));
    if (p == nullptr || !p->IsAlive()) {
      continue;
    }
    const CVector3f d = p->GetTranslation() + CVector3f(0.f, 0.f, 1.f) - gr.pos;
    if (Len(d) < radius + 0.5f) {
      mgr.ApplyDamage(player.GetUniqueId(), p->GetUniqueId(), player.GetUniqueId(), info, filter, Norm(d) * 15.f);
    }
  }
}

void UpdateGrenades(CStateManager& mgr, float dt) {
  constexpr float g = 14.f;
  for (size_t i = sGrenades.size(); i-- > 0;) {
    Grenade& gr = sGrenades[i];
    gr.fuse -= dt;
    gr.vel = gr.vel - CVector3f(0.f, 0.f, g * dt);
    const CVector3f step = gr.vel * dt;
    const float len = Len(step);
    bool boom = gr.fuse <= 0.f;
    if (len > 1e-5f) {
      const CRayCastResult hit =
          CGameCollision::RayStaticIntersection(mgr, gr.pos, step * (1.f / len), len + 0.12f, SolidFilter());
      if (hit.IsValid()) {
        const CVector3f n = hit.GetPlane().GetNormal();
        gr.pos = hit.GetPoint() + n * 0.13f;
        gr.vel = (gr.vel - n * (2.f * CVector3f::Dot(gr.vel, n))) * 0.45f;
        // Knocked back, it goes off where it lands.
        boom = boom || gr.returned;
      } else {
        gr.pos = gr.pos + step;
      }
    }
    if (gr.returned && !boom) {
      // Or against an enemy.
      const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
      for (int idx = actors.GetFirstObjectIndex(); idx != -1 && !boom; idx = actors.GetNextObjectIndex(idx)) {
        const CPatterned* p = TCastToConstPtr< CPatterned >(actors.GetObjectByIndex(idx));
        boom = p != nullptr && p->IsAlive() && Len(p->GetTranslation() + CVector3f(0.f, 0.f, 1.f) - gr.pos) < 1.4f;
      }
    }
    if (boom) {
      const Grenade done = gr;
      sGrenades.erase(sGrenades.begin() + std::ptrdiff_t(i));
      Explode(done, mgr);
      continue;
    }
    PortFx::GrenadeTick(gr.pos, gr.fuse, gr.returned);
  }
}

// ---- Cover: behind a column, out of Samus' sight, stepping out to shoot ----
bool FloorAt(CStateManager& mgr, CVector3f& at, float maxRise) {
  const CRayCastResult floor =
      CGameCollision::RayStaticIntersection(mgr, at + CVector3f(0.f, 0.f, 1.5f), CVector3f::Down(), 3.5f, SolidFilter());
  if (!floor.IsValid() || std::fabs(floor.GetPoint().GetZ() - at.GetZ()) > maxRise) {
    return false;
  }
  at = floor.GetPoint();
  return true;
}

bool FindCover(CPatterned& self, Ai& a, CStateManager& mgr) {
  const CVector3f eye = EyeOf(*mgr.Player());
  const CVector3f pos = self.GetTranslation();
  const CVector3f waist(0.f, 0.f, 0.9f);
  const CVector3f chest(0.f, 0.f, 1.3f);
  float best = 1e9f;
  const float start = Rand(mgr) * 6.2831853f;
  for (int k = 0; k < 16; ++k) {
    const float ang = start + float(k) * (6.2831853f / 16.f);
    const float r = (k & 1) ? 3.5f : 6.5f;
    CVector3f c = pos + CVector3f(std::cos(ang) * r, std::sin(ang) * r, 0.f);
    if (!FloorAt(mgr, c, 1.2f) || Len(Flat(c - eye)) < 6.f || Blocked(mgr, pos + waist, c + waist) ||
        !Blocked(mgr, c + chest, eye)) {
      continue;
    }
    const CVector3f side = Norm(CVector3f::Cross(Norm(Flat(eye - c)), CVector3f(0.f, 0.f, 1.f)));
    for (const float sgn : {1.f, -1.f}) {
      CVector3f peek = c + side * (1.3f * sgn);
      if (!FloorAt(mgr, peek, 0.8f) || Blocked(mgr, c + waist, peek + waist) || Blocked(mgr, peek + chest, eye)) {
        continue;
      }
      const float d = Len(c - pos);
      if (d < best) {
        best = d;
        a.coverPos = c;
        a.peekPos = peek;
      }
      break;
    }
  }
  return best < 1e8f;
}

// True while cover takes over this tick.
bool Cover(CPatterned& self, Ai& a, float dt, CStateManager& mgr, bool hurt) {
  const CVector3f eye = EyeOf(*mgr.Player());
  const CVector3f face = Norm(Flat(eye - self.GetTranslation()));
  a.coverT -= dt;
  switch (a.coverPhase) {
  case 1: { // to the cover
    const CVector3f to = Flat(a.coverPos - self.GetTranslation());
    if (Len(to) < 0.6f) {
      a.coverPhase = 2;
      a.coverT = RandRange(mgr, 1.f, 1.8f);
    } else if (a.coverT <= 0.f) {
      a.coverPhase = 0;
      return false;
    }
    Steer(self, Norm(to), face);
    return true;
  }
  case 2: // hiding
    if (!Blocked(mgr, a.coverPos + CVector3f(0.f, 0.f, 1.3f), eye)) {
      a.coverPhase = 0; // Samus came round: no longer cover
      return false;
    }
    if (a.coverT <= 0.f) {
      if (a.coverCycles-- <= 0) {
        a.coverPhase = 0;
        return false;
      }
      a.coverPhase = 3;
      a.coverT = RandRange(mgr, 1.8f, 2.6f);
    }
    Steer(self, Norm(Flat(a.coverPos - self.GetTranslation())) * std::min(1.f, Len(a.coverPos - self.GetTranslation())),
          face);
    return true;
  case 3: { // peeking out: steps out, then its own fight takes over (it shoots)
    if (hurt || a.coverT <= 0.f) {
      a.coverPhase = 2;
      a.coverT = RandRange(mgr, 1.f, 2.f);
      return true;
    }
    const CVector3f to = Flat(a.peekPos - self.GetTranslation());
    if (Len(to) > 0.5f) {
      Steer(self, Norm(to), face);
      return true;
    }
    return false;
  }
  default:
    return false;
  }
}

void MaybeTakeCover(CPatterned& self, Ai& a, float dt, CStateManager& mgr) {
  if (a.coverCd > 0.f) {
    a.coverCd -= dt;
    return;
  }
  a.coverCd = RandRange(mgr, 6.f, 10.f) * (Hard() ? 0.7f : 1.f);
  const CVector3f eye = EyeOf(*mgr.Player());
  const float dist = Len(Flat(eye - self.GetTranslation()));
  if (dist < 7.f || dist > 28.f || Blocked(mgr, self.GetTranslation() + CVector3f(0.f, 0.f, 1.5f), eye) ||
      Rand(mgr) > 0.6f) {
    return;
  }
  if (FindCover(self, a, mgr)) {
    a.coverPhase = 1;
    a.coverT = 4.f;
    a.coverCycles = 2 + int(Rand(mgr) * 2.f);
  }
}

// ---- Flanking: coming at Samus from a side instead of straight on ----
bool Flank(CPatterned& self, Ai& a, float dt, CStateManager& mgr) {
  CPlayer& player = *mgr.Player();
  const CVector3f toSelf = Flat(self.GetTranslation() - player.GetTranslation());
  const float dist = Len(toSelf);
  const bool ranged = Pirate(self) || self.GetCharacterType() == CPatterned::kC_FlyingPirate ||
                      self.GetCharacterType() == CPatterned::kC_Drone;
  const float radius = ranged ? 9.f : 2.5f;
  if (a.flankT > 0.f) {
    a.flankT -= dt;
    const CVector3f target =
        player.GetTranslation() + CVector3f(std::cos(a.flankBearing), std::sin(a.flankBearing), 0.f) * radius;
    const CVector3f to = Flat(target - self.GetTranslation());
    const CVector3f waist(0.f, 0.f, 0.9f);
    if (Len(to) < 1.5f || Blocked(mgr, self.GetTranslation() + waist, self.GetTranslation() + waist + Norm(to) * 2.f)) {
      a.flankT = 0.f;
      return false;
    }
    // Blended over its own steering, heavier: it still aims and shoots on the way.
    self.BodyCtrl()->CommandMgr().DeliverCmd(CBCLocomotionCmd(Norm(to), Norm(-toSelf), 2.5f));
    return true;
  }
  if (a.flankCd > 0.f) {
    a.flankCd -= dt;
    return false;
  }
  a.flankCd = RandRange(mgr, 5.f, 8.f) * (Hard() ? 0.7f : 1.f);
  if (dist < (ranged ? 7.f : 5.f) || dist > 28.f ||
      Blocked(mgr, self.GetTranslation() + CVector3f(0.f, 0.f, 1.5f), EyeOf(player))) {
    return false;
  }
  // Partners split: one to each side.
  const float side = (self.GetUniqueId().value & 1) != 0 ? 1.f : -1.f;
  a.flankBearing = std::atan2(toSelf.GetY(), toSelf.GetX()) + side * RandRange(mgr, 1.0f, 1.35f);
  a.flankT = 4.5f;
  return false;
}

// ---- A pirate whose partner died ----
void CheckPartner(CPatterned& self, Ai& a, CStateManager& mgr) {
  if (a.rage) {
    return;
  }
  if (a.partner == kInvalidUniqueId) {
    if (CPatterned* p = NearestPirate(self, mgr, 25.f)) {
      a.partner = p->GetUniqueId();
      Ai& b = StateOf(*p);
      if (b.partner == kInvalidUniqueId) {
        b.partner = self.GetUniqueId();
      }
    }
    return;
  }
  const CPatterned* p = TCastToConstPtr< CPatterned >(mgr.GetObjectById(a.partner));
  if (p != nullptr && p->IsAlive()) {
    return;
  }
  a.partner = kInvalidUniqueId;
  a.rage = true;
  a.coverPhase = 0;
  a.fleeT = 0.f;
  a.fled = true; // it fights to the end
  a.chargeCd = 0.f;
  PortFx::Rage(self.GetTranslation() + CVector3f(0.f, 0.f, 1.2f));
  PortLog::Write("port: enemy ai: a space pirate lost its partner and is enraged\n");
}

// ---- The cannon strike ----
float sMeleeT = -1.f;
float sMeleeCool = 0.f;
bool sMeleeHit = false;
bool sLbHeld = false;
bool sKeyHeld = false;

void MeleeStrike(CPlayer& player, CStateManager& mgr) {
  const CFirstPersonCamera* cam = mgr.GetCameraManager()->GetFirstPersonCamera();
  if (cam == nullptr) {
    return;
  }
  const CTransform4f xf = cam->GetTransform();
  const CVector3f eye = xf.GetTranslation();
  const CVector3f fwd = xf.GetForward();
  const int beam = int(mgr.GetPlayerState()->GetCurrentBeam());
  CDamageInfo info = gpTweakPlayerGun->GetBeamInfo(beam >= 0 && beam <= 4 ? beam : 0).x20_charged;
  info.SetKnockBackPower(std::max(info.GetKnockBackPower(), 10.f));
  int hits = 0;
  const CObjectList& actors = mgr.GetObjectListById(kOL_Actor);
  for (int idx = actors.GetFirstObjectIndex(); idx != -1 && hits < 3; idx = actors.GetNextObjectIndex(idx)) {
    const CEntity* e = actors.GetObjectByIndex(idx);
    if (e == nullptr || e->GetUniqueId() == player.GetUniqueId() || !e->GetActive()) {
      continue;
    }
    CActor* act = TCastToPtr< CActor >(mgr.ObjectById(e->GetUniqueId()));
    if (act == nullptr) {
      continue;
    }
    if (const CPatterned* p = TCastToConstPtr< CPatterned >(act)) {
      if (!p->IsAlive()) {
        continue;
      }
    }
    const CHealthInfo* h = act->HealthInfo(mgr);
    if (h == nullptr || h->GetHP() <= 0.f) {
      continue;
    }
    const CVector3f at = act->GetAimPosition(mgr, 0.f);
    const CVector3f d = at - eye;
    const float dist = Len(d);
    if (dist > 3.3f || (dist > 0.6f && CVector3f::Dot(d * (1.f / dist), fwd) < 0.5f)) {
      continue;
    }
    if (dist > 0.6f && Blocked(mgr, eye, at - d * (0.4f / dist))) {
      continue;
    }
    const CVector3f fxAt = dist > 0.6f ? at - d * (0.35f / dist) : eye + fwd * 0.8f;
    const CMaterialFilter filter = CMaterialFilter::MakeIncludeExclude(CMaterialList(kMT_Solid), CMaterialList());
    CPatterned* enemy = TCastToPtr< CPatterned >(act);
    if (enemy != nullptr && enemy->GetBodyCtrl() != nullptr && enemy->GetBodyCtrl()->IsFrozen()) {
      // Frozen: it shatters.
      enemy->PortShatter();
      const CDamageInfo lethal(CWeaponMode(kWT_None), h->GetHP() + 1000.f, 0.f, 10.f);
      mgr.ApplyDamage(player.GetUniqueId(), act->GetUniqueId(), player.GetUniqueId(), lethal, filter, fwd);
      PortFx::Shatter(fxAt);
      ++hits;
      continue;
    }
    const CDamageVulnerability* vuln = act->GetDamageVulnerability(eye, fwd, info);
    const bool hurts = vuln == nullptr || vuln->WeaponHurts(info.GetWeaponMode(), CDamageVulnerability::kRD_No);
    if (enemy != nullptr && !hurts) {
      // A shield the beam can't get through: the blow breaks through it, at a little over half strength.
      CDamageInfo breaker(CWeaponMode(kWT_None), info.GetDamage() * 0.6f, 0.f, info.GetKnockBackPower());
      mgr.ApplyDamage(player.GetUniqueId(), act->GetUniqueId(), player.GetUniqueId(), breaker, filter, fwd);
      PortFx::ShieldBreak(fxAt, fwd);
    } else {
      mgr.ApplyDamage(player.GetUniqueId(), act->GetUniqueId(), player.GetUniqueId(), info, filter, fwd);
      PortFx::MeleeHit(fxAt, fwd, beam);
    }
    // The Ice Beam's blow freezes.
    if (enemy != nullptr && enemy->IsAlive() && beam == CPlayerState::kBI_Ice && enemy->GetBodyCtrl() != nullptr &&
        !enemy->GetBodyCtrl()->IsFrozen() && !IsBoss(enemy->GetCharacterType()) &&
        (vuln == nullptr || vuln->WeaponHurts(CWeaponMode(kWT_Ice), CDamageVulnerability::kRD_No))) {
      enemy->Freeze(mgr, at, CUnitVector3f(fwd), 5.f);
    }
    ++hits;
  }
  // Grenades within reach go back the way they came, along the aim.
  for (Grenade& gr : sGrenades) {
    const CVector3f d = gr.pos - eye;
    const float dist = Len(d);
    if (gr.returned || dist > 3.5f || (dist > 0.5f && CVector3f::Dot(d * (1.f / dist), fwd) < 0.2f)) {
      continue;
    }
    gr.returned = true;
    gr.vel = fwd * 22.f + CVector3f(0.f, 0.f, 2.5f);
    gr.fuse = std::max(gr.fuse, 1.6f);
    PortFx::MeleeHit(gr.pos, fwd, beam);
  }
}

float Ease(float x) {
  x = std::clamp(x, 0.f, 1.f);
  return x * x * (3.f - 2.f * x);
}

// The strike's pose at time t: offset (gun space) and turn (roll about forward, pitch), keyframed.
void Pose(float t, CVector3f& offset, float& roll, float& pitch) {
  struct Key {
    float t, x, y, z, roll, pitch;
  };
  static const Key keys[] = {
      {0.00f, 0.f, 0.f, 0.f, 0.f, 0.f},
      {0.07f, 0.03f, -0.12f, -0.05f, -14.f, 9.f},   // drawn back
      {0.15f, -0.07f, 0.42f, 0.04f, 30.f, -12.f},   // thrust
      {0.24f, -0.06f, 0.36f, 0.02f, 24.f, -8.f},    // held a moment
      {0.46f, 0.f, 0.f, 0.f, 0.f, 0.f},             // back
  };
  offset = CVector3f::Zero();
  roll = pitch = 0.f;
  if (t < 0.f) {
    return;
  }
  for (size_t i = 0; i + 1 < std::size(keys); ++i) {
    const Key& a = keys[i];
    const Key& b = keys[i + 1];
    if (t <= b.t) {
      const float k = Ease((t - a.t) / (b.t - a.t));
      offset = CVector3f(a.x + (b.x - a.x) * k, a.y + (b.y - a.y) * k, a.z + (b.z - a.z) * k);
      roll = a.roll + (b.roll - a.roll) * k;
      pitch = a.pitch + (b.pitch - a.pitch) * k;
      return;
    }
  }
}
} // namespace

float SpeedScale(const CPatterned& self) {
  if (!self.IsAlive() || IsBoss(self.GetCharacterType())) {
    return 1.f;
  }
  float scale = float(PortDebug::EnemySpeed()) / 100.f;
  if (Hard()) {
    scale = std::max(scale, 1.75f);
  }
  if (Smart()) {
    const auto it = sAi.find(self.GetUniqueId().value);
    if (it != sAi.end() && it->second.rage) {
      scale *= 1.35f;
    }
  }
  return scale;
}

float PlayerDamageScale() { return Hard() ? 2.f : 1.f; }

void Update(CStateManager& mgr, float dt) {
  if (dt <= 0.f || mgr.GetPlayer() == nullptr) {
    return;
  }
  if (!Smart()) {
    sGrenades.clear();
    return;
  }
  UpdateGrenades(mgr, dt);
}

void Think(CPatterned& self, float dt, CStateManager& mgr) {
  ++sCounter;
  if ((sCounter & 255) == 0) {
    Prune();
  }
  if (!Smart() || !self.IsAlive() || mgr.GetPlayer() == nullptr || IsBoss(self.GetCharacterType()) ||
      self.BodyCtrl() == nullptr || dt <= 0.f) {
    return;
  }
  Ai& a = StateOf(self);
  // Hard+: half as much health again, once.
  if (Hard() && !a.hpScaled) {
    a.hpScaled = true;
    if (CHealthInfo* h = self.HealthInfo(mgr)) {
      h->SetHP(h->GetHP() * 1.5f);
    }
  }
  const float hp = HpOf(self, mgr);
  if (a.lastHp < 0.f) {
    a.maxHp = a.lastHp = hp;
  }
  a.maxHp = std::max(a.maxHp, hp);
  const bool hurt = hp < a.lastHp - 0.01f;
  a.lastHp = hp;
  if (a.chargeCd > 0.f) {
    a.chargeCd -= dt;
  }
  if (hurt) {
    Hurt(self, mgr);
  }
  if (Pirate(self)) {
    Heal(self, a, dt, mgr);
    CheckPartner(self, a, mgr);
    if (a.rage) {
      a.rageFx -= dt;
      if (a.rageFx <= 0.f) {
        a.rageFx = 0.35f;
        PortFx::RageAura(self.GetTranslation() + CVector3f(0.f, 0.f, 1.f));
      }
      // Enraged: it keeps coming at Samus to strike.
      if (a.chargeT <= 0.f && a.chargeCd <= 0.f) {
        a.chargeT = 6.f;
        a.chargeCd = 3.f;
        a.struck = false;
        a.coverPhase = 0;
      }
    }
  }
  const bool mobile = Mobile(self.GetCharacterType());
  if (!mobile) {
    return;
  }
  // Nearly dead: run, once.
  if (!a.fled && a.maxHp > 0.f && hp < 0.25f * a.maxHp && hp > 0.f) {
    a.fled = true;
    a.fleeT = 5.f;
    a.chargeT = 0.f;
  }
  if (a.fleeT > 0.f) {
    Flee(self, a, dt, mgr);
    return;
  }
  if (a.chargeT > 0.f) {
    a.coverPhase = 0;
    Charge(self, a, dt, mgr);
    return;
  }
  if (Pirate(self)) {
    // Grenades now and then, when Samus is in sight and not too close.
    if (a.grenadeCd < 0.f) {
      a.grenadeCd = RandRange(mgr, 5.f, 12.f);
    }
    a.grenadeCd -= dt;
    if (a.grenadeCd <= 0.f && a.coverPhase != 1) {
      a.grenadeCd = RandRange(mgr, 8.f, 14.f) * (Hard() ? 0.65f : 1.f);
      const CVector3f eye = EyeOf(*mgr.Player());
      const float dist = Len(Flat(eye - self.GetTranslation()));
      if (dist > 7.f && dist < 22.f && Rand(mgr) < 0.5f &&
          !Blocked(mgr, self.GetTranslation() + CVector3f(0.f, 0.f, 1.8f), eye)) {
        Throw(self, mgr);
      }
    }
    if (a.coverPhase == 0) {
      MaybeTakeCover(self, a, dt, mgr);
    }
    if (a.coverPhase != 0 && Cover(self, a, dt, mgr, hurt)) {
      return;
    }
  }
  if (a.coverPhase == 0) {
    Flank(self, a, dt, mgr);
  }
  Dodge(self, a, dt, mgr);
}

bool SeedlingTouch(CSeedling& self, CStateManager& mgr) {
  if (!Smart() || !self.IsAlive()) {
    return false;
  }
  Ai& a = StateOf(self);
  if (a.latched || a.latchCd > 0.f) {
    return true;
  }
  CPlayer& player = *mgr.Player();
  if (player.GetMorphballTransitionState() != CPlayer::kMS_Unmorphed ||
      !mgr.GetPlayerState()->IsAlive() || !player.AttachActorToPlayer(self.GetUniqueId(), false)) {
    return false;
  }
  a.latched = true;
  a.hadSolid = self.GetMaterialList().HasMaterial(kMT_Solid);
  a.hadCharacter = self.GetMaterialList().HasMaterial(kMT_Character);
  if (a.hadSolid) {
    self.RemoveMaterial(kMT_Solid, mgr);
  }
  if (a.hadCharacter) {
    self.RemoveMaterial(kMT_Character, mgr);
  }
  PortLog::Write("port: enemy ai: a Seedling latched onto Samus\n");
  return true;
}

bool SeedlingThink(CSeedling& self, float dt, CStateManager& mgr) {
  if (!Smart()) {
    return false;
  }
  Ai& a = StateOf(self);
  if (a.latchCd > 0.f) {
    a.latchCd -= dt;
  }
  if (!a.latched) {
    return false;
  }
  CPlayer& player = *mgr.Player();
  const bool keep = self.IsAlive() && mgr.GetPlayerState()->IsAlive() &&
                    player.GetMorphballTransitionState() == CPlayer::kMS_Unmorphed &&
                    player.GetAttachedActor() == self.GetUniqueId();
  if (!keep) {
    // Shaken off: thrown down beside her.
    if (player.GetAttachedActor() == self.GetUniqueId()) {
      player.DetachActorFromPlayer();
    }
    a.latched = false;
    a.latchCd = 4.f;
    if (a.hadSolid) {
      self.AddMaterial(kMT_Solid, mgr);
    }
    if (a.hadCharacter) {
      self.AddMaterial(kMT_Character, mgr);
    }
    const float yaw = float(mgr.Random()->Next() % 628) / 100.f;
    CVector3f at = player.GetTranslation() + CVector3f(std::cos(yaw) * 1.8f, std::sin(yaw) * 1.8f, 0.6f);
    const CRayCastResult floor =
        CGameCollision::RayStaticIntersection(mgr, at, CVector3f::Down(), 4.f, SolidFilter());
    if (floor.IsValid()) {
      at = floor.GetPoint() + CVector3f(0.f, 0.f, 0.05f);
    }
    self.SetTransform(CTransform4f::Translate(at));
    self.SetVelocityWR(CVector3f::Zero());
    return true;
  }
  // On the visor, its belly against the glass.
  const CFirstPersonCamera* cam = mgr.GetCameraManager()->GetFirstPersonCamera();
  if (cam != nullptr) {
    const CTransform4f xf = cam->GetTransform();
    const CVector3f f = xf.GetForward();
    const CVector3f u = xf.GetUp();
    const float wobble = std::sin(float(sCounter) * 0.37f) * 0.03f;
    const CVector3f pos = xf.GetTranslation() + f * 0.95f - u * (0.3f + wobble);
    const CVector3f right = CVector3f::Cross(u, f);
    self.SetTransform(CTransform4f::FromColumns(right, u, f, pos));
  }
  self.SetVelocityWR(CVector3f::Zero());
  const CDamageInfo info(CWeaponMode(kWT_PoisonWater), (Hard() ? 14.f : 9.f) * dt, 0.f, 0.f, true);
  player.SetNoDamageLoopSfx(true);
  mgr.ApplyDamage(self.GetUniqueId(), player.GetUniqueId(), self.GetUniqueId(), info,
                  CMaterialFilter::MakeIncludeExclude(CMaterialList(kMT_Solid), CMaterialList()), CVector3f::Zero());
  player.SetNoDamageLoopSfx(false);
  mgr.PlayerState()->StaticInterference().AddSource(self.GetUniqueId(), 0.35f, 0.2f);
  return true;
}

float SeedlingCoolOffScale() { return Smart() ? 0.45f : 1.f; }

void PlayerInput(CPlayer& player, const CFinalInput& input, CStateManager& mgr) {
  const float dt = input.Time();
  if (sMeleeCool > 0.f) {
    sMeleeCool -= dt;
  }
  if (sMeleeT >= 0.f) {
    sMeleeT += dt;
    if (!sMeleeHit && sMeleeT >= 0.13f) {
      sMeleeHit = true;
      MeleeStrike(player, mgr);
    }
    if (sMeleeT > 0.46f) {
      sMeleeT = -1.f;
      sMeleeCool = 0.12f;
    }
  }
  bool lb = false;
  if (SDL_Gamepad* pad = PADGetSDLGamepadForIndex(0)) {
    lb = SDL_GetGamepadButton(pad, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
  }
  int keyCount = 0;
  const bool* keys = SDL_GetKeyboardState(&keyCount);
  const bool key = keys != nullptr && SDL_SCANCODE_V < keyCount && keys[SDL_SCANCODE_V];
  const bool pressed = (lb && !sLbHeld) || (key && !sKeyHeld);
  sLbHeld = lb;
  sKeyHeld = key;
  if (!pressed || !PortDebug::CannonMelee() || sMeleeT >= 0.f || sMeleeCool > 0.f) {
    return;
  }
  if (player.GetMorphballTransitionState() != CPlayer::kMS_Unmorphed ||
      mgr.GetCameraManager()->IsInCinematicCamera() || !mgr.GetPlayerState()->IsAlive()) {
    return;
  }
  sMeleeT = 0.f;
  sMeleeHit = false;
}

CVector3f MeleeGunOffset() {
  CVector3f offset;
  float roll, pitch;
  Pose(sMeleeT, offset, roll, pitch);
  return offset;
}

CTransform4f MeleeGunTurn() {
  CVector3f offset;
  float roll, pitch;
  Pose(sMeleeT, offset, roll, pitch);
  if (roll == 0.f && pitch == 0.f) {
    return CTransform4f::Identity();
  }
  return CTransform4f::RotateY(CRelAngle::FromDegrees(roll)) * CTransform4f::RotateX(CRelAngle::FromDegrees(pitch));
}

void Reset() {
  sAi.clear();
  sGrenades.clear();
  sMeleeT = -1.f;
}
} // namespace PortEnemyAi
