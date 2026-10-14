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
  // Not every time: about half the shots.
  if ((mgr.Random()->Next() % 100) >= 55) {
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
  a.dodgeCd = 1.6f;
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
        const CDamageInfo info(CWeaponMode(kWT_AI), MeleeDamage(self), 0.f, 12.f);
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
    mgr.ApplyDamage(player.GetUniqueId(), act->GetUniqueId(), player.GetUniqueId(), info,
                    CMaterialFilter::MakeIncludeExclude(CMaterialList(kMT_Solid), CMaterialList()), fwd);
    PortFx::MeleeHit(dist > 0.6f ? at - d * (0.35f / dist) : eye + fwd * 0.8f, fwd, beam);
    ++hits;
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
  return float(PortDebug::EnemySpeed()) / 100.f;
}

void Think(CPatterned& self, float dt, CStateManager& mgr) {
  ++sCounter;
  if ((sCounter & 255) == 0) {
    Prune();
  }
  if (!PortDebug::SmarterEnemies() || !self.IsAlive() || mgr.GetPlayer() == nullptr ||
      IsBoss(self.GetCharacterType()) || self.BodyCtrl() == nullptr || dt <= 0.f) {
    return;
  }
  Ai& a = StateOf(self);
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
    Charge(self, a, dt, mgr);
    return;
  }
  Dodge(self, a, dt, mgr);
}

bool SeedlingTouch(CSeedling& self, CStateManager& mgr) {
  if (!PortDebug::SmarterEnemies() || !self.IsAlive()) {
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
  if (!PortDebug::SmarterEnemies()) {
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
  const CDamageInfo info(CWeaponMode(kWT_PoisonWater), 9.f * dt, 0.f, 0.f, true);
  player.SetNoDamageLoopSfx(true);
  mgr.ApplyDamage(self.GetUniqueId(), player.GetUniqueId(), self.GetUniqueId(), info,
                  CMaterialFilter::MakeIncludeExclude(CMaterialList(kMT_Solid), CMaterialList()), CVector3f::Zero());
  player.SetNoDamageLoopSfx(false);
  mgr.PlayerState()->StaticInterference().AddSource(self.GetUniqueId(), 0.35f, 0.2f);
  return true;
}

float SeedlingCoolOffScale() { return PortDebug::SmarterEnemies() ? 0.45f : 1.f; }

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
  sMeleeT = -1.f;
}
} // namespace PortEnemyAi
