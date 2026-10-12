#pragma once

#include <cstdint>
#include <string>
#include <vector>

class CStateManager;
class CInputStream;

// Save states: save anywhere, load back to the same spot. A state is the
// retail save data (the same CGameState stream a memory card save holds:
// items, health, ammo, map, scans, script layers and relays, in-game time)
// plus where Samus stood, how she faced and whether she was in morph ball.
// Loading rebuilds the world from that save, as the debug world warp does,
// then puts Samus back. Enemies, doors and puzzles come back the way a
// memory card load leaves them, not mid-fight.
namespace PortSaveState {

constexpr int kSlotCount = 8;  // slots 1..8
constexpr int kUndoSlot = 0;   // written before every load

struct Info {
  bool exists = false;
  std::string world;       // front-end name
  std::string room;        // room name, or "Room N"
  double playTime = 0.0;   // in-game seconds
  int64_t savedAt = 0;     // Unix seconds
  bool morphed = false;
};

// The fixed header ahead of the game-state blob. Pure, for the unit test.
struct Header {
  uint32_t worldId = 0;
  int32_t areaId = 0;
  float position[3] = {0.f, 0.f, 0.f};
  float forward[3] = {0.f, 1.f, 0.f};
  bool morphed = false;
  double playTime = 0.0;
  int64_t savedAt = 0;
  std::string world;
  std::string room;
};
std::string Encode(const Header& header, const std::vector< uint8_t >& blob);
bool Decode(const std::string& data, Header& header, std::vector< uint8_t >& blob);

// Quick resume: one extra state (resume.mpss, outside the slots) written when
// the app goes to the background or quits, with which memory card file it
// came from and which game it belongs to. The next launch offers it at the
// title screen. Pure parts here, for the unit test; glue in port_resume.cpp.
struct ResumeMeta {
  uint32_t fileIdx = 0;     // memory card file (0-2)
  uint64_t cardSerial = 0;  // the card it was read from
  std::string identity;     // randomizer seed / Archipelago game, "" for retail
};
std::string EncodeResume(const ResumeMeta& meta, const std::string& stateFile);
bool DecodeResume(const std::string& data, ResumeMeta& meta, std::string& stateFile);
// Whether a resume made in `stored` may be offered for the game in `current`.
bool ResumeMatches(const ResumeMeta& stored, const ResumeMeta& current);

// What the writer knows about the running game.
struct ResumeContext {
  bool inGame = false;       // a world is loaded with Samus in it
  bool initialized = false;  // the state manager finished starting up
  bool quitting = false;
  bool dead = false;
  bool cinematic = false;    // cutscene or cinematic camera
  bool transition = false;   // elevator ride / world transition
  bool saveUi = false;       // memory card save screen open
};
// Why a resume must not be written now, or nullptr when it may.
const char* ResumeSkipReason(const ResumeContext& ctx);

// The resume file is built from an in-memory snapshot kept on the game thread
// (a "last safe point", refreshed on room entry and every kResumeRefreshSeconds
// while nothing below applies), so the write at background or quit needs no
// game state and works from an SDL event watch. What to write then:
enum class ResumeWriteChoice {
  kNothing,          // no snapshot yet (title screen): keep the old file
  kRefreshThenWrite, // safe right now: take a fresh snapshot, write it
  kWriteSnapshot,    // dead, cutscene, elevator...: write the last safe one
};
ResumeWriteChoice ChooseResumeWrite(bool safeNow, bool haveSnapshot);
constexpr double kResumeRefreshSeconds = 30.0;
constexpr double kResumeStableSeconds = 2.0;
// Whether the game thread should refresh the snapshot this frame. A new room
// is taken once it has been stable for kResumeStableSeconds.
bool ResumeRefreshDue(bool safeNow, bool haveSnapshot, bool sameRoomAsSnapshot, double secondsInRoom,
                      double secondsSinceSnapshot);

// Port glue (port_savestate.cpp). Game thread unless noted.

// The resume file, as the title screen prompt shows it.
struct ResumeInfo {
  bool exists = false;
  uint32_t fileIdx = 0;
  uint64_t cardSerial = 0;
  std::string world;
  std::string room;
  double playTime = 0.0;
  int64_t savedAt = 0;
};
// Quit or background, between ticks: refreshes the snapshot when it is safe,
// then writes resume.mpss (synchronously, atomically) from the last snapshot.
// Needs the quick_resume setting (or `force`). True when written; with no
// snapshot the previous file is kept.
bool ResumeSave(bool force = false);
// Per frame, from CStateManager's tick: keeps the snapshot current.
void ResumeTick(CStateManager& mgr);
// Registers the SDL event watch that writes on WILL/DID_ENTER_BACKGROUND (SDL
// never queues those events, it only calls watchers, on the thread that pumps).
void ResumeInstallEventWatch();
// True around the main loop's event pump: a background event arriving there
// may take a fresh snapshot first (the game is between ticks).
void SetEventPumpSafe(bool safe);
// Writes the last snapshot. Any thread. False when there is none, nothing new
// to write, or the write failed.
bool ResumeFlush();
// The resume file, when it is readable and belongs to this game (same
// randomizer or Archipelago identity).
bool ResumeOffer(ResumeInfo& info);
// Yes at the title screen, once the card serial is known: queues the state
// for the game CFrontEndUI is about to start, and gives the file to start it
// with. The file is deleted once Samus is placed. False if the file is gone
// or from another game or card.
bool ResumeAccept(uint64_t cardSerial, uint32_t& fileIdx);
// No: deletes the file.
void ResumeDecline();
// Console: load the resume state from inside a game; delete it; one line.
bool RequestResumeLoad();
void ResumeClear();
std::string ResumeStatus();

// The folder states live in (created on first use).
std::string Folder();
Info SlotInfo(int slot);
// The slot F5/F9 use.
int SelectedSlot();
void SetSelectedSlot(int slot);

// Save or load from the overlay, a hotkey or the console. They run at the
// next game tick; the pause and map screens hold them until you unpause.
// A request with no game running is refused at once.
bool RequestSave(int slot); // false (and LastMessage says why) if refused now
bool RequestLoad(int slot);
// Reads the mods folder again (and installs a finished Remastered import):
// the game is rebuilt where Samus stands, as by a save and a load, with every
// PAK reopened. In the front end CFrontEndUI does it at its next tick; with
// no game running otherwise, the next one to start does it.
bool RequestModReload();
// How many times the mod files were read again, to tell when a request ran.
int ModReloads();
// What the last request did, for the overlay and the console.
std::string LastMessage();

// CStateManager::Update: does a queued save, places Samus after a load, and
// returns true when a load has started (the caller must return at once: the
// world is going away).
bool Tick(CStateManager& mgr);
// CMainFlow, before the next game loads: swaps in the loaded game state.
void InstallPending();
// CMainFlow: whether the front end (title screen, file select) is the flow.
void SetInFrontEnd(bool inFrontEnd);
// CFrontEndUI, each tick: true (once) when a reload was asked for in the
// front end. It closes its movies, calls ReloadModsNow and opens them again,
// since rebuilding the disc-file overlays breaks the reads of open files.
bool TakeFrontEndReload();
void ReloadModsNow();

} // namespace PortSaveState
