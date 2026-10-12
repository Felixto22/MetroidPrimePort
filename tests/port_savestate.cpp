#include "port_savestate.h"

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
int sLine = 0;
void Check(bool condition) {
  if (!condition) {
    std::fprintf(stderr, "savestate regression failed at line %d\n", sLine);
    std::abort();
  }
}
#define CHECK(cond)                                                                                \
  do {                                                                                             \
    sLine = __LINE__;                                                                              \
    Check(cond);                                                                                   \
  } while (0)

using PortSaveState::Header;
} // namespace

int main() {
  Header in;
  in.worldId = 0x83F6FF6F;
  in.areaId = 17;
  in.position[0] = -120.5f;
  in.position[1] = 33.25f;
  in.position[2] = 9.f;
  in.forward[0] = 0.f;
  in.forward[1] = -1.f;
  in.forward[2] = 0.f;
  in.morphed = true;
  in.playTime = 3723.5;
  in.savedAt = 1790000000;
  in.world = "Chozo Ruins";
  in.room = "Main Plaza";
  std::vector< uint8_t > blob(4096, 0);
  for (size_t i = 0; i < 940; ++i)
    blob[i] = static_cast< uint8_t >(i * 7);

  const std::string data = PortSaveState::Encode(in, blob);

  // Round trip.
  {
    Header out;
    std::vector< uint8_t > outBlob;
    CHECK(PortSaveState::Decode(data, out, outBlob));
    CHECK(out.worldId == in.worldId);
    CHECK(out.areaId == in.areaId);
    for (int i = 0; i < 3; ++i) {
      CHECK(out.position[i] == in.position[i]);
      CHECK(out.forward[i] == in.forward[i]);
    }
    CHECK(out.morphed);
    CHECK(out.playTime == in.playTime);
    CHECK(out.savedAt == in.savedAt);
    CHECK(out.world == in.world);
    CHECK(out.room == in.room);
    CHECK(outBlob == blob);
  }

  // Every truncation fails and leaves the outputs alone.
  for (size_t len = 0; len < data.size(); ++len) {
    Header out;
    out.worldId = 1;
    std::vector< uint8_t > outBlob{9};
    CHECK(!PortSaveState::Decode(data.substr(0, len), out, outBlob));
    CHECK(out.worldId == 1);
    CHECK(outBlob.size() == 1);
  }

  // Trailing junk, a wrong magic, a wrong version and an empty blob fail.
  {
    Header out;
    std::vector< uint8_t > outBlob;
    CHECK(!PortSaveState::Decode(data + "x", out, outBlob));
    std::string bad = data;
    bad[0] = 'X';
    CHECK(!PortSaveState::Decode(bad, out, outBlob));
    bad = data;
    bad[4] = 2;
    CHECK(!PortSaveState::Decode(bad, out, outBlob));
    CHECK(!PortSaveState::Decode(PortSaveState::Encode(in, {}), out, outBlob));
  }

  // A NaN position is refused (it would teleport Samus into nowhere).
  {
    Header nan = in;
    nan.position[1] = std::numeric_limits< float >::quiet_NaN();
    Header out;
    std::vector< uint8_t > outBlob;
    CHECK(!PortSaveState::Decode(PortSaveState::Encode(nan, blob), out, outBlob));
  }

  // Long labels are cut, not rejected.
  {
    Header longLabel = in;
    longLabel.room.assign(1000, 'r');
    Header out;
    std::vector< uint8_t > outBlob;
    CHECK(PortSaveState::Decode(PortSaveState::Encode(longLabel, blob), out, outBlob));
    CHECK(out.room.size() == 256);
  }

  // Quick resume wrapper.
  {
    PortSaveState::ResumeMeta meta;
    meta.fileIdx = 1;
    meta.cardSerial = 0x1122334455667788ull;
    meta.identity = "ap:seed42";
    const std::string state = "state-bytes";
    const std::string file = PortSaveState::EncodeResume(meta, state);
    PortSaveState::ResumeMeta back;
    std::string backState;
    CHECK(PortSaveState::DecodeResume(file, back, backState));
    CHECK(back.fileIdx == 1 && back.cardSerial == meta.cardSerial && back.identity == meta.identity);
    CHECK(backState == state);
    CHECK(!PortSaveState::DecodeResume(file.substr(0, file.size() - 1), back, backState));
    CHECK(!PortSaveState::DecodeResume(file + "x", back, backState));
    CHECK(!PortSaveState::DecodeResume("MPSS" + file.substr(4), back, backState));
    CHECK(!PortSaveState::DecodeResume("", back, backState));
    PortSaveState::ResumeMeta badSlot = meta;
    badSlot.fileIdx = 7;
    CHECK(!PortSaveState::DecodeResume(PortSaveState::EncodeResume(badSlot, state), back, backState));

    CHECK(PortSaveState::ResumeMatches(meta, meta));
    PortSaveState::ResumeMeta other = meta;
    other.identity = "ap:seed43";
    CHECK(!PortSaveState::ResumeMatches(meta, other));
    other = meta;
    other.cardSerial = 1;
    CHECK(!PortSaveState::ResumeMatches(meta, other));
    other = meta;
    other.identity.clear();
    CHECK(!PortSaveState::ResumeMatches(meta, other));
  }

  // Quick resume skip rules.
  {
    PortSaveState::ResumeContext ctx;
    CHECK(PortSaveState::ResumeSkipReason(ctx) != nullptr);  // not in game
    ctx.inGame = true;
    ctx.initialized = true;
    CHECK(PortSaveState::ResumeSkipReason(ctx) == nullptr);
    bool PortSaveState::ResumeContext::*flags[] = {
        &PortSaveState::ResumeContext::quitting, &PortSaveState::ResumeContext::dead,
        &PortSaveState::ResumeContext::cinematic, &PortSaveState::ResumeContext::transition,
        &PortSaveState::ResumeContext::saveUi};
    for (auto flag : flags) {
      PortSaveState::ResumeContext c = ctx;
      c.*flag = true;
      CHECK(PortSaveState::ResumeSkipReason(c) != nullptr);
    }
    ctx.initialized = false;
    CHECK(PortSaveState::ResumeSkipReason(ctx) != nullptr);
  }

  // Snapshot selection at quit/background.
  {
    using PortSaveState::ResumeWriteChoice;
    CHECK(PortSaveState::ChooseResumeWrite(true, false) == ResumeWriteChoice::kRefreshThenWrite);
    CHECK(PortSaveState::ChooseResumeWrite(true, true) == ResumeWriteChoice::kRefreshThenWrite);
    CHECK(PortSaveState::ChooseResumeWrite(false, true) == ResumeWriteChoice::kWriteSnapshot);
    CHECK(PortSaveState::ChooseResumeWrite(false, false) == ResumeWriteChoice::kNothing);
  }
  // Snapshot refresh schedule.
  {
    using PortSaveState::ResumeRefreshDue;
    const double stable = PortSaveState::kResumeStableSeconds;
    const double period = PortSaveState::kResumeRefreshSeconds;
    // Never while unsafe.
    CHECK(!ResumeRefreshDue(false, true, true, 1000.0, 1000.0));
    CHECK(!ResumeRefreshDue(false, false, false, 1000.0, 0.0));
    // First snapshot, or a new room, waits for the room to be stable.
    CHECK(!ResumeRefreshDue(true, false, false, stable - 0.1, 0.0));
    CHECK(ResumeRefreshDue(true, false, false, stable, 0.0));
    CHECK(!ResumeRefreshDue(true, true, false, stable - 0.1, 5.0));
    CHECK(ResumeRefreshDue(true, true, false, stable, 5.0));
    // The same room refreshes on the period only.
    CHECK(!ResumeRefreshDue(true, true, true, 100.0, period - 0.1));
    CHECK(ResumeRefreshDue(true, true, true, 100.0, period));
  }

  std::puts("savestate tests passed");
  return 0;
}
