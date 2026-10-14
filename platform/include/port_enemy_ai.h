#pragma once

// Enemy behaviour (port_debug.h SmarterEnemies, EnemySpeed, CannonMelee):
//  - every enemy but the bosses moves, animates and attacks EnemySpeed% as fast;
//  - the mobile ones sidestep shots coming at them (space pirates with their own dodge roll) and,
//    nearly dead, run from Samus for a while;
//  - a space pirate hurt by Samus sends its nearest partner charging at her to strike with its
//    rifle's blade, for twice its shot's damage; a hurt pirate gets a health pack thrown by a partner;
//  - a Seedling that touches Samus latches onto her visor and hurts her until she morphs;
//  - Samus strikes with the arm cannon on LB (V on the keyboard), as hard as a charged shot; it
//    breaks through shields the beam can't, freezes with the Ice Beam, shatters what is frozen and
//    knocks grenades back;
//  - pirates take cover and step out to shoot, enemies flank Samus from the sides, pirates throw
//    grenades, and a pirate whose partner dies is enraged (faster, always closing in to strike);
//  - Hard+ (port_debug.h HardPlus): all of it, enemies at least 175% as fast, with half as much
//    health again, and their damage to Samus doubled.
#include "Kyoto/Math/CTransform4f.hpp"
#include "Kyoto/Math/CVector3f.hpp"

class CFinalInput;
class CPatterned;
class CPlayer;
class CSeedling;
class CStateManager;

namespace PortEnemyAi {
// The factor an enemy's think time runs at (1 for bosses and with the setting at 100%).
float SpeedScale(const CPatterned& self);
// Once a game tick (CStateManager::Update): the grenades in flight.
void Update(CStateManager& mgr, float dt);
// What the enemies' damage to Samus is multiplied by (Hard+: 2).
float PlayerDamageScale();
// At the end of CPatterned::Think, after its state machine (it may take over the body's commands).
void Think(CPatterned& self, float dt, CStateManager& mgr);

// A Seedling touched by something: true when it latched on (or is on), instead of bursting.
bool SeedlingTouch(CSeedling& self, CStateManager& mgr);
// A Seedling's think: true while it is latched (it then skips its own).
bool SeedlingThink(CSeedling& self, float dt, CStateManager& mgr);
// How much of its attack cool-off a Seedling waits (it fires its spines more often).
float SeedlingCoolOffScale();

// The player's input (CPlayer::ProcessInput): the cannon strike.
void PlayerInput(CPlayer& player, const CFinalInput& input, CStateManager& mgr);
// The cannon strike's motion: an offset added to the gun's position (gun space: x right,
// y forward, z up) and a turn applied to the gun's transform.
CVector3f MeleeGunOffset();
CTransform4f MeleeGunTurn();

// Forgets everything tracked (a world change).
void Reset();
} // namespace PortEnemyAi
