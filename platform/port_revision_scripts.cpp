#include "port_revision_scripts.h"

#include "port_skip_cutscenes.h"

namespace PortRevisionScripts {
namespace {

using Version = PortDisc::Version;

const uint32_t kGeothermalCore = 0xC0498676;
const uint32_t kMainQuarry = 0x643D038F;
const uint32_t kOreProcessing = 0x97D2B2F6;
const uint32_t kRuinedCourtyard = 0x1921876D;
const uint32_t kEliteResearch = 0x8A97BB54;
const uint32_t kArboretum = 0x18AB6106;

// Op codes and conn state/message codes (port_skip_cutscenes_ops.cpp).
enum : uint8_t { kRemConn = 3, kAddConn = 4, kDelete = 6 };
enum : uint32_t {
  kActive = 0,
  kArrived = 1,
  kInactive = 5,
  kDead = 0xE,
  kActivate = 1,
  kDeactivate = 4,
  kSetToZero = 0xD,
};

struct Writer {
  std::vector< uint8_t > ops;
  void u32(uint32_t v) {
    for (int shift = 24; shift >= 0; shift -= 8)
      ops.push_back(uint8_t(v >> shift));
  }
  void conn(uint8_t op, uint32_t sender, uint32_t state, uint32_t msg, uint32_t target) {
    ops.push_back(op);
    u32(sender);
    u32(state);
    u32(msg);
    u32(target);
  }
  void remove(uint32_t id) {
    ops.push_back(kDelete);
    u32(id);
  }
};

// PAL's Power Bomb rock and door blocker in front of Plasma Processing (the
// whole set: generator, debris, effect, rumble, memory relay, scan, blocker).
std::vector< uint8_t > GeothermalPal() {
  Writer w;
  for (uint32_t id = 0x001403AA; id <= 0x001403B5; ++id)
    w.remove(id);
  return std::move(w.ops);
}

// USA 1.01/1.02's invisible platform over the runic gate, and the message
// that turns it off.
std::vector< uint8_t > ArboretumUsa() {
  Writer w;
  w.conn(kRemConn, 0x00130140, kInactive, kDeactivate, 0x001302AA);
  w.remove(0x001302AA);
  return std::move(w.ops);
}

// The 1.02 locks are an actor, an inventory activator and a 0.02 s timer;
// dropping the three leaves the 1.00 room (Grapple Beam here).
std::vector< uint8_t > GeothermalUsa102() {
  Writer w;
  w.remove(0x0014039E);
  w.remove(0x0014039F);
  w.remove(0x001403A0);
  return std::move(w.ops);
}

// Power Bomb lock.
std::vector< uint8_t > MainQuarryUsa102() {
  Writer w;
  w.remove(0x000205D3);
  w.remove(0x000205D4);
  w.remove(0x000205D5);
  return std::move(w.ops);
}

// Power Bomb lock.
std::vector< uint8_t > OreProcessingUsa102() {
  Writer w;
  w.remove(0x0006036A);
  w.remove(0x0006036B);
  w.remove(0x0006036C);
  return std::move(w.ops);
}

// Thermal Visor lock: the relay F01D9 stops feeding the lock's objects, which
// go away, and is wired to F01C8 as in 1.00 (Dead -> Activate, and F01C8's
// Active -> Deactivate back).
std::vector< uint8_t > RuinedCourtyardUsa102() {
  Writer w;
  w.conn(kRemConn, 0x000F01D9, kDead, kSetToZero, 0x000F0315);
  w.conn(kRemConn, 0x000F01D9, kDead, kSetToZero, 0x000F0312);
  w.conn(kAddConn, 0x000F01D9, kDead, kActivate, 0x000F01C8);
  w.conn(kAddConn, 0x000F01C8, kActive, kDeactivate, 0x000F01D9);
  for (uint32_t id = 0x000F0312; id <= 0x000F0315; ++id)
    w.remove(id);
  return std::move(w.ops);
}

// Warrior exit lock: the relay D0517 that held the exit goes, and the pirate
// 40D01A4's death resets the exit's objects directly again. (They land after
// its other conns, not at their 1.00 positions.)
std::vector< uint8_t > EliteResearchUsa102() {
  Writer w;
  w.conn(kRemConn, 0x000D0340, kArrived, kSetToZero, 0x000D0517);
  w.remove(0x000D0517);
  w.conn(kAddConn, 0x040D01A4, kDead, kSetToZero, 0x040D0407);
  w.conn(kAddConn, 0x040D01A4, kDead, kSetToZero, 0x000D01F6);
  return std::move(w.ops);
}

} // namespace

std::vector< uint8_t > OpsFor(Version version, uint32_t mreaId) {
  if (version == Version::Pal && mreaId == kGeothermalCore)
    return GeothermalPal();
  if ((version == Version::Usa101 || version == Version::Usa102) && mreaId == kArboretum)
    return ArboretumUsa();
  if (version == Version::Usa102) {
    switch (mreaId) {
    case kGeothermalCore:
      return GeothermalUsa102();
    case kMainQuarry:
      return MainQuarryUsa102();
    case kOreProcessing:
      return OreProcessingUsa102();
    case kRuinedCourtyard:
      return RuinedCourtyardUsa102();
    case kEliteResearch:
      return EliteResearchUsa102();
    }
  }
  return {};
}

Result Apply(Version version, uint32_t mreaId, const uint8_t* scly, size_t size,
             std::vector< uint8_t >& out, int* misses) {
  out.clear();
  if (misses != nullptr)
    *misses = 0;
  const std::vector< uint8_t > ops = OpsFor(version, mreaId);
  if (ops.empty())
    return Result::NoPatch;
  std::vector< uint8_t > patched;
  const int missed = PortSkipCutscenes::ApplyOps(scly, size, ops.data(), ops.size(), patched);
  if (missed != 0) {
    if (misses != nullptr)
      *misses = missed;
    return Result::Missed;
  }
  out.swap(patched);
  return Result::Patched;
}

const char* Describe(Version version, uint32_t mreaId) {
  if (version == Version::Pal && mreaId == kGeothermalCore)
    return "PAL Geothermal Core Power Bomb rock";
  if (version == Version::Usa101 && mreaId == kArboretum)
    return "USA 1.01 Arboretum gate platform";
  if (version == Version::Usa102) {
    switch (mreaId) {
    case kArboretum:
      return "USA 1.02 Arboretum gate platform";
    case kGeothermalCore:
      return "USA 1.02 Geothermal Core Grapple lock";
    case kMainQuarry:
      return "USA 1.02 Main Quarry Power Bomb lock";
    case kOreProcessing:
      return "USA 1.02 Ore Processing Power Bomb lock";
    case kRuinedCourtyard:
      return "USA 1.02 Ruined Courtyard Thermal lock";
    case kEliteResearch:
      return "USA 1.02 Elite Research Warrior lock";
    }
  }
  return "revision script";
}

} // namespace PortRevisionScripts
