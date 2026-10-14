#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

// "What is this thread doing", for the port's hang watchdog (platform/port_watchdog.cpp).
// Two slots, the main thread and the render worker, each a string literal and the time it
// was set. Lock-free and cheap enough to call around anything that can block in a driver;
// readers see a name and a time that may belong to neighbouring updates, which is fine.
namespace aurora::phase {

enum Slot : int { Main = 0, Render = 1, SlotCount = 2 };

struct State {
  std::atomic<const char*> name{"startup"};
  std::atomic<int64_t> sinceNs{0};
};

inline State& state(Slot slot) noexcept {
  static State s_states[SlotCount];
  return s_states[slot];
}

inline int64_t now_ns() noexcept {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Time spent per phase name since reset_totals, for the slot's own thread to read (the stall
// log says where a slow frame went). Up to MaxTotals names; the rest are lumped as "other".
constexpr int MaxTotals = 48;
struct Total {
  const char* name;
  int64_t ns;
};
struct Totals {
  Total entries[MaxTotals];
  int count = 0;
  int64_t otherNs = 0;
};
inline Totals& totals(Slot slot) noexcept {
  static Totals s_totals[SlotCount];
  return s_totals[slot];
}
inline void reset_totals(Slot slot) noexcept {
  Totals& t = totals(slot);
  t.count = 0;
  t.otherNs = 0;
}
inline void add_total(Slot slot, const char* name, int64_t ns) noexcept {
  if (name == nullptr || ns <= 0) {
    return;
  }
  Totals& t = totals(slot);
  for (int i = 0; i < t.count; ++i) {
    if (t.entries[i].name == name) {
      t.entries[i].ns += ns;
      return;
    }
  }
  if (t.count < MaxTotals) {
    t.entries[t.count++] = {name, ns};
  } else {
    t.otherNs += ns;
  }
}

// `name` must outlive the process (a string literal, or an interned string).
inline void set(Slot slot, const char* name) noexcept {
  State& s = state(slot);
  const int64_t now = now_ns();
  add_total(slot, s.name.load(std::memory_order_relaxed), now - s.sinceNs.load(std::memory_order_relaxed));
  s.name.store(name, std::memory_order_relaxed);
  s.sinceNs.store(now, std::memory_order_relaxed);
}

// Sets a phase and puts the previous one back on exit.
class Scope {
public:
  Scope(Slot slot, const char* name) noexcept : m_slot(slot), m_previous(state(slot).name.load(std::memory_order_relaxed)) {
    set(slot, name);
  }
  ~Scope() { set(m_slot, m_previous); }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

private:
  Slot m_slot;
  const char* m_previous;
};

} // namespace aurora::phase
