// Revision script patches (PortRevisionScripts) on synthetic rooms: the
// version/room gate, the content of every op stream, the all-or-nothing miss
// path, the five USA 1.02 restores and the PAL / USA 1.01 normalizations.
// No disc data is used; the all-disc replay is tools/ (see docs).
#include "port_revision_scripts.h"
#include "port_skip_cutscenes.h"

#include <cstdio>
#include <map>
#include <vector>

namespace {

using PortDisc::Version;
using PortRevisionScripts::Result;
using Conn = PortSkipCutscenes::ScriptObject::Connection;

bool sPassed = true;
void Check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "[revision-scripts] FAILED: %s\n", message);
    sPassed = false;
  }
}

const uint32_t kGeothermal = 0xC0498676, kQuarry = 0x643D038F, kOre = 0x97D2B2F6,
               kCourtyard = 0x1921876D, kElite = 0x8A97BB54, kArboretum = 0x18AB6106;

struct Obj {
  std::vector< Conn > conns;
};
using Room = std::map< uint32_t, Obj >;

void Put32(std::vector< uint8_t >& v, uint32_t x) {
  for (int s = 24; s >= 0; s -= 8)
    v.push_back(uint8_t(x >> s));
}

// A one-layer SCLY section holding the objects (type 5, empty name).
std::vector< uint8_t > Build(const Room& room) {
  std::vector< uint8_t > layer = {0};
  Put32(layer, uint32_t(room.size()));
  for (const auto& [id, obj] : room) {
    std::vector< uint8_t > body;
    Put32(body, id);
    Put32(body, uint32_t(obj.conns.size()));
    for (const Conn& c : obj.conns) {
      Put32(body, c.state);
      Put32(body, c.message);
      Put32(body, c.target);
    }
    Put32(body, 1);
    body.push_back(0);
    layer.push_back(5);
    Put32(layer, uint32_t(body.size()));
    layer.insert(layer.end(), body.begin(), body.end());
  }
  std::vector< uint8_t > scly = {'S', 'C', 'L', 'Y', 0, 0, 0, 1};
  Put32(scly, 1);
  Put32(scly, uint32_t(layer.size()));
  scly.insert(scly.end(), layer.begin(), layer.end());
  return scly;
}

Room Parse(const std::vector< uint8_t >& scly) {
  std::vector< PortSkipCutscenes::ScriptObject > objs;
  Room room;
  Check(PortSkipCutscenes::ScanObjects(scly.data(), scly.size(), objs), "result parses");
  for (const auto& o : objs)
    room[o.id].conns = o.connections;
  return room;
}

bool Has(const Room& r, uint32_t id) { return r.count(id) != 0; }
bool HasConn(const Room& r, uint32_t sender, uint32_t state, uint32_t msg, uint32_t target) {
  auto it = r.find(sender);
  if (it == r.end())
    return false;
  for (const Conn& c : it->second.conns)
    if (c.state == state && c.message == msg && c.target == target)
      return true;
  return false;
}

Room Ids(std::initializer_list< uint32_t > ids) {
  Room r;
  for (uint32_t id : ids)
    r[id];
  return r;
}

Result Run(Version v, uint32_t mrea, const Room& in, Room& out, int* misses = nullptr) {
  const std::vector< uint8_t > scly = Build(in);
  std::vector< uint8_t > bytes;
  const Result res = PortRevisionScripts::Apply(v, mrea, scly.data(), scly.size(), bytes, misses);
  if (res == Result::Patched)
    out = Parse(bytes);
  else
    Check(bytes.empty(), "out is empty unless patched");
  return res;
}

const Version kAll[] = {Version::Unknown, Version::Usa100, Version::Usa101, Version::Usa102,
                        Version::Pal};
const uint32_t kRooms[] = {kGeothermal, kQuarry, kOre, kCourtyard, kElite, kArboretum};

} // namespace

int main() {
  // Gate: exactly these (version, room) pairs have ops.
  for (Version v : kAll) {
    for (uint32_t room : kRooms) {
      bool expect = false;
      if (v == Version::Usa102)
        expect = true; // all five locks + Arboretum
      else if (v == Version::Usa101)
        expect = room == kArboretum;
      else if (v == Version::Pal)
        expect = room == kGeothermal;
      Check(PortRevisionScripts::OpsFor(v, room).empty() != expect, "version/room gate");
    }
    Check(PortRevisionScripts::OpsFor(v, 0x12345678).empty(), "unknown room has no ops");
  }
  Check(PortRevisionScripts::OpsFor(Version::Usa100, kGeothermal).empty(), "1.00 untouched");
  // Stream sizes: DELETE is 5 bytes, a conn op 17.
  Check(PortRevisionScripts::OpsFor(Version::Pal, kGeothermal).size() == 12 * 5, "PAL: 12 deletes");
  Check(PortRevisionScripts::OpsFor(Version::Usa101, kArboretum).size() == 17 + 5, "Arboretum ops");
  Check(PortRevisionScripts::OpsFor(Version::Usa102, kArboretum) ==
            PortRevisionScripts::OpsFor(Version::Usa101, kArboretum),
        "Arboretum 1.01 == 1.02");
  Check(PortRevisionScripts::OpsFor(Version::Usa102, kGeothermal).size() == 3 * 5, "1.02 Geothermal");
  Check(PortRevisionScripts::OpsFor(Version::Usa102, kQuarry).size() == 3 * 5, "1.02 Quarry");
  Check(PortRevisionScripts::OpsFor(Version::Usa102, kOre).size() == 3 * 5, "1.02 Ore");
  Check(PortRevisionScripts::OpsFor(Version::Usa102, kCourtyard).size() == 4 * 17 + 4 * 5,
        "1.02 Courtyard");
  Check(PortRevisionScripts::OpsFor(Version::Usa102, kElite).size() == 3 * 17 + 5, "1.02 Elite");

  Room out;
  int misses = -1;

  // Old PAL Geothermal: 12 ids 1403AA..1403B5 go, the neighbours stay.
  {
    Room in = Ids({0x1403A9, 0x1403B6});
    for (uint32_t id = 0x1403AA; id <= 0x1403B5; ++id)
      in[id];
    Check(Run(Version::Pal, kGeothermal, in, out, &misses) == Result::Patched && misses == 0,
          "PAL Geothermal applies");
    Check(out.size() == 2 && Has(out, 0x1403A9) && Has(out, 0x1403B6), "PAL Geothermal drops the set");
    // The 1.02 ids collide with different PAL objects: PAL must not delete them.
    Room pal = Ids({0x14039E, 0x14039F, 0x1403A0});
    for (uint32_t id = 0x1403AA; id <= 0x1403B5; ++id)
      pal[id];
    Check(Run(Version::Pal, kGeothermal, pal, out) == Result::Patched && out.size() == 3 &&
              Has(out, 0x14039E) && Has(out, 0x14039F) && Has(out, 0x1403A0),
          "PAL keeps the 1.02 lock ids");
    // Deletes of absent ids are silent, so a 1.00 USA-like room still "applies"
    // and USA discs never reach it (gate): check the gate instead.
    Check(Run(Version::Usa100, kGeothermal, in, out) == Result::NoPatch, "1.00 Geothermal no patch");
    Check(Run(Version::Usa101, kGeothermal, in, out) == Result::NoPatch, "1.01 Geothermal no patch");
  }

  // Old Arboretum (1.01 and 1.02): the platform and the conn that disables it.
  for (Version v : {Version::Usa101, Version::Usa102}) {
    Room in = Ids({0x1302AA, 0x1302AB});
    in[0x130140].conns = {{5, 4, 0x1302AA}, {5, 4, 0x1302AB}};
    Check(Run(v, kArboretum, in, out) == Result::Patched, "Arboretum applies");
    Check(!Has(out, 0x1302AA) && Has(out, 0x1302AB), "Arboretum drops only the platform");
    Check(!HasConn(out, 0x130140, 5, 4, 0x1302AA) && HasConn(out, 0x130140, 5, 4, 0x1302AB),
          "Arboretum drops only its message");
    Room bad = Ids({0x1302AA}); // the conn is missing: fail-safe
    Check(Run(v, kArboretum, bad, out, &misses) == Result::Missed && misses == 1,
          "Arboretum without the conn misses");
    Check(Run(Version::Usa100, kArboretum, in, out) == Result::NoPatch, "1.00 Arboretum untouched");
    Check(Run(Version::Pal, kArboretum, in, out) == Result::NoPatch, "PAL Arboretum untouched");
  }

  // The five 1.02 restores.
  Check(Run(Version::Usa102, kGeothermal, Ids({0x14039D, 0x14039E, 0x14039F, 0x1403A0, 0x1403A1}),
            out) == Result::Patched &&
            out.size() == 2 && Has(out, 0x14039D) && Has(out, 0x1403A1),
        "1.02 Geothermal lock");
  Check(Run(Version::Usa102, kQuarry, Ids({0x205D2, 0x205D3, 0x205D4, 0x205D5, 0x205D6}), out) ==
                Result::Patched &&
            out.size() == 2 && Has(out, 0x205D2) && Has(out, 0x205D6),
        "1.02 Quarry lock");
  Check(Run(Version::Usa102, kOre, Ids({0x60369, 0x6036A, 0x6036B, 0x6036C, 0x6036D}), out) ==
                Result::Patched &&
            out.size() == 2 && Has(out, 0x60369) && Has(out, 0x6036D),
        "1.02 Ore lock");

  Room court = Ids({0xF01C8, 0xF0311, 0xF0312, 0xF0313, 0xF0314, 0xF0315, 0xF0316});
  court[0xF01D9].conns = {{0xE, 0xD, 0xF0315}, {0xE, 0xD, 0xF0312}, {0xE, 0xD, 0xF0316}};
  Check(Run(Version::Usa102, kCourtyard, court, out) == Result::Patched, "1.02 Courtyard applies");
  Check(Has(out, 0xF01C8) && Has(out, 0xF0311) && Has(out, 0xF0316) && !Has(out, 0xF0312) &&
            !Has(out, 0xF0313) && !Has(out, 0xF0314) && !Has(out, 0xF0315),
        "Courtyard lock objects gone");
  Check(HasConn(out, 0xF01D9, 0xE, 0xD, 0xF0316) && !HasConn(out, 0xF01D9, 0xE, 0xD, 0xF0315) &&
            !HasConn(out, 0xF01D9, 0xE, 0xD, 0xF0312) && out[0xF01D9].conns.size() == 2,
        "Courtyard relay conns");
  Check(HasConn(out, 0xF01D9, 0xE, 1, 0xF01C8) && HasConn(out, 0xF01C8, 0, 4, 0xF01D9),
        "Courtyard 1.00 wiring added");
  Check(out[0xF01D9].conns.back().target == 0xF01C8, "added conn is appended");

  Room elite = Ids({0xD01F6, 0x40D0407});
  elite[0xD0340].conns = {{1, 0xD, 0xD0517}, {1, 0xD, 0xD0100}};
  elite[0xD0517];
  elite[0x40D01A4].conns = {{0xE, 4, 0x1}};
  Check(Run(Version::Usa102, kElite, elite, out) == Result::Patched, "1.02 Elite applies");
  Check(!Has(out, 0xD0517) && !HasConn(out, 0xD0340, 1, 0xD, 0xD0517) &&
            HasConn(out, 0xD0340, 1, 0xD, 0xD0100),
        "Elite relay and its conn go, the other conn stays");
  Check(HasConn(out, 0x40D01A4, 0xE, 0xD, 0x40D0407) && HasConn(out, 0x40D01A4, 0xE, 0xD, 0xD01F6) &&
            out[0x40D01A4].conns.size() == 3 && out[0x40D01A4].conns[0].target == 1,
        "Elite exit conns appended after the pirate's own");

  // Fail-safe: a missing REMCONN means nothing changes (all or nothing).
  {
    Room noRem = court;
    noRem[0xF01D9].conns = {{0xE, 0xD, 0xF0315}}; // the 0xF0312 conn is absent
    const std::vector< uint8_t > scly = Build(noRem);
    std::vector< uint8_t > bytes = {1, 2, 3};
    Check(PortRevisionScripts::Apply(Version::Usa102, kCourtyard, scly.data(), scly.size(), bytes,
                                     &misses) == Result::Missed && misses == 1 && bytes.empty(),
          "Courtyard missing remconn misses once, out empty");
    Room noRem2 = elite; // 1.00 / 1.01 / PAL layout: the relay conn isn't there
    noRem2[0xD0340].conns = {{1, 0xD, 0xD0100}};
    Check(Run(Version::Usa102, kElite, noRem2, out, &misses) == Result::Missed && misses == 1,
          "Elite missing remconn misses");
  }
  // A malformed script is a miss, never a crash.
  {
    const uint8_t junk[4] = {1, 2, 3, 4};
    std::vector< uint8_t > bytes;
    Check(PortRevisionScripts::Apply(Version::Usa102, kQuarry, junk, 4, bytes, &misses) ==
                  Result::Missed && bytes.empty(),
          "junk script misses");
  }

  Check(PortRevisionScripts::Describe(Version::Usa102, kElite)[0] != 0, "describe");

  if (sPassed)
    std::puts("[revision-scripts] OK");
  return sPassed ? 0 : 1;
}
