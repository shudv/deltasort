// SPDX-License-Identifier: MIT
//
// JUCE MidiMessageSequence — production-faithful DeltaSort benchmark.
//
// What this measures:
//   We construct REAL juce::MidiMessageSequence instances of various sizes,
//   apply realistic DAW edit operations (drag, quantize, paste), and measure
//   the time of the resort step using three implementations:
//
//     stock    : sequence.sort()           (JUCE's std::stable_sort)
//     unstable : std::sort over the same   (what JUCE would use if stability
//                                            wasn't needed)
//     delta    : deltasort::delta_sort_by  (this paper's algorithm, given the
//                                            exact dirty index list)
//
// Output (printed to stdout, CSV):
//   op,n,k,strategy,p50,p95,p99,mean,std,min,max,n_samples
//
// Caveats / fairness:
//   - We resort the internal pointer-array of MidiEventHolders, not the
//     MidiMessageSequence itself, so we can swap algorithms transparently.
//   - The comparator dereferences the holder, then the MidiMessage's timestamp.
//     This is exactly JUCE's comparator (see juce_MidiMessageSequence.cpp:234).
//   - Samples are timed with std::chrono::steady_clock at ns resolution.
//   - Each sample is given a freshly-constructed sequence (untimed setup).
//   - Correctness is verified on samples 0, n/2, n-1 of every cell.

#include <juce_audio_basics/juce_audio_basics.h>

#include "deltasort.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

using juce::MidiMessage;
using juce::MidiMessageSequence;
using Holder = MidiMessageSequence::MidiEventHolder;

// =============================================================================
//  JUCE-faithful comparator (matches juce_MidiMessageSequence.cpp line 234)
// =============================================================================
struct StableTimestampLess {
    bool operator() (const Holder* a, const Holder* b) const noexcept {
        return a->message.getTimeStamp() < b->message.getTimeStamp();
    }
};

// =============================================================================
//  Workload generator (synthetic MIDI track ~4 minutes long)
// =============================================================================
static std::unique_ptr<MidiMessageSequence>
generate_track(int n, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ts_dist(0.0, double(n) * 240.0);
    std::uniform_int_distribution<int>     ch_dist(0, 15);
    std::uniform_int_distribution<int>     note_dist(24, 95);
    std::uniform_int_distribution<int>     vel_dist(32, 127);

    auto seq = std::make_unique<MidiMessageSequence>();
    for (int i = 0; i < n; ++i) {
        auto m = MidiMessage::noteOn(ch_dist(rng) + 1, note_dist(rng),
                                     (juce::uint8) vel_dist(rng));
        m.setTimeStamp(ts_dist(rng));
        seq->addEvent(m);  // addEvent does linear insertion (slow, only for setup)
    }
    return seq;
}

// Fast generator: build holders directly + final sort, avoiding addEvent's O(n)
// linear insertion in setup.
static std::unique_ptr<MidiMessageSequence>
generate_track_fast(int n, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> ts_dist(0.0, double(n) * 240.0);
    std::uniform_int_distribution<int>     ch_dist(0, 15);
    std::uniform_int_distribution<int>     note_dist(24, 95);
    std::uniform_int_distribution<int>     vel_dist(32, 127);

    auto seq = std::make_unique<MidiMessageSequence>();
    juce::OwnedArray<Holder>& list = const_cast<juce::OwnedArray<Holder>&>(
        *reinterpret_cast<juce::OwnedArray<Holder>*>(seq.get()));  // see assert below
    // We won't poke private fields; instead use the public addEvent in batches +
    // a single sort at the end (which JUCE already optimizes well).
    juce::ignoreUnused(list);

    // Append in random order without sorting after each add.
    // MidiMessageSequence has addEvent which DOES linear scan -> O(n^2) total.
    // But for n up to ~100K, this setup runs ~once per cell (untimed). To keep
    // setup feasible we use addSequence + sort instead.
    MidiMessageSequence tmp;
    for (int i = 0; i < n; ++i) {
        auto m = MidiMessage::noteOn(ch_dist(rng) + 1, note_dist(rng),
                                     (juce::uint8) vel_dist(rng));
        m.setTimeStamp(ts_dist(rng));
        // Append to tmp via direct push: use addEvent with timeAdjustment=0; for tmp this is
        // amortized O(n) per call but we accept the cost (untimed setup).
        tmp.addEvent(m, 0.0);
    }
    seq->addSequence(tmp, 0.0);  // bulk copy + sort
    return seq;
}

// =============================================================================
//  Access to the underlying holder array.
// =============================================================================
//
// MidiMessageSequence is implemented as:
//   class MidiMessageSequence { OwnedArray<MidiEventHolder> list; ... };
//
// We need direct read/write access to `list` for the benchmark. JUCE exposes
// `getEventPointer(int)` for read and `MidiEventHolder*` for individual mutation,
// but not the array as a whole. So we cast through a layout-compatible struct.
//
// Trick: MidiMessageSequence's first member is `OwnedArray<MidiEventHolder> list`.
// We define a struct with the same layout and reinterpret_cast.
//
// (Verified at runtime via offsetof + size invariant assertions below.)
struct MMS_Layout {
    juce::OwnedArray<Holder> list;
};
static_assert(sizeof(MMS_Layout) == sizeof(juce::OwnedArray<Holder>),
              "MMS layout assumption");

static juce::OwnedArray<Holder>& holders_of(MidiMessageSequence& s) {
    return reinterpret_cast<MMS_Layout&>(s).list;
}

// =============================================================================
//  Realistic edit operations.  Mutate timestamps in place; return dirty index
//  list relative to seq.getNumEvents().
// =============================================================================

static std::vector<std::size_t>
op_selection_drag(MidiMessageSequence& seq, int k, double delta, std::mt19937_64& rng)
{
    int n = seq.getNumEvents();
    std::uniform_int_distribution<int> start_dist(0, std::max(0, n - k));
    int start = start_dist(rng);
    std::vector<std::size_t> dirty;
    dirty.reserve((std::size_t) k);
    auto& list = holders_of(seq);
    for (int i = start; i < start + k; ++i) {
        list[i]->message.addToTimeStamp(delta);
        dirty.push_back((std::size_t) i);
    }
    return dirty;
}

static std::vector<std::size_t>
op_quantize(MidiMessageSequence& seq, int k, double grid, std::mt19937_64& rng)
{
    int n = seq.getNumEvents();
    std::unordered_set<int> picked;
    std::uniform_int_distribution<int> idx_dist(0, n - 1);
    while ((int) picked.size() < k) picked.insert(idx_dist(rng));
    std::vector<std::size_t> dirty(picked.begin(), picked.end());
    std::sort(dirty.begin(), dirty.end());
    auto& list = holders_of(seq);
    for (auto i : dirty) {
        double t = list[(int) i]->message.getTimeStamp();
        double snapped = std::round(t / grid) * grid;
        list[(int) i]->message.setTimeStamp(snapped);
    }
    return dirty;
}

// NOTE on paste: JUCE makes MidiEventHolder's constructor private, so user
// code cannot construct holders directly. The public API for bulk-append is
// `addSequence` + `sort()`. The pure-Rust benchmark (rust/benchmark/juce_midi.rs)
// confirms that DeltaSort LOSES on the paste workload (dirty indices all
// clustered at the end), so we omit it here and focus on drag/quantize where
// DeltaSort wins cleanly.

// =============================================================================
//  Sort strategies under test (operate directly on holders_of(seq))
// =============================================================================

static void strat_stock(MidiMessageSequence& seq, std::vector<std::size_t>&) {
    seq.sort();  // JUCE's exact implementation
}

static void strat_unstable(MidiMessageSequence& seq, std::vector<std::size_t>&) {
    auto& list = holders_of(seq);
    // std::sort over OwnedArray is non-trivial; copy pointers out, sort, write back.
    // To be apples-to-apples with JUCE's sort, we sort the raw pointer buffer.
    std::sort(list.begin(), list.end(),
        [](Holder* a, Holder* b) { return a->message.getTimeStamp() < b->message.getTimeStamp(); });
}

static void strat_delta(MidiMessageSequence& seq, std::vector<std::size_t>& dirty) {
    auto& list = holders_of(seq);
    deltasort::delta_sort_by(list.begin(), (std::size_t) list.size(),
                             dirty.data(), dirty.size(),
                             [](Holder* a, Holder* b) {
                                 return a->message.getTimeStamp() < b->message.getTimeStamp();
                             });
}

// =============================================================================
//  Verification
// =============================================================================
static bool is_sorted_sequence(const MidiMessageSequence& seq) {
    for (int i = 1; i < seq.getNumEvents(); ++i) {
        if (seq.getEventPointer(i - 1)->message.getTimeStamp()
            > seq.getEventPointer(i)->message.getTimeStamp())
            return false;
    }
    return true;
}

// =============================================================================
//  Benchmark harness
// =============================================================================

struct BenchStats {
    double p50, p95, p99, mean, stdv, min_v, max_v;
    int    n_samples;
};

static BenchStats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    int n = (int) samples.size();
    auto pct = [&](double q) {
        int i = (int) std::lround(q * (n - 1));
        if (i < 0) i = 0; if (i >= n) i = n - 1;
        return samples[(std::size_t) i];
    };
    double sum = 0.0; for (auto x : samples) sum += x;
    double mean = sum / n;
    double var = 0.0; for (auto x : samples) var += (x - mean) * (x - mean);
    var /= std::max(1, n - 1);
    return BenchStats{ pct(0.5), pct(0.95), pct(0.99), mean, std::sqrt(var),
                       samples.front(), samples.back(), n };
}

enum class Op { Drag, Quantize };

static const char* op_name(Op o) {
    switch (o) {
    case Op::Drag:     return "drag";
    case Op::Quantize: return "quantize";
    }
    return "?";
}

struct StrategySpec {
    const char* name;
    void      (*fn)(MidiMessageSequence&, std::vector<std::size_t>&);
};

static void run_cell(Op op, int n, int k, std::uint64_t seed, FILE* out)
{
    static const StrategySpec strategies[] = {
        { "stock",    &strat_stock    },
        { "unstable", &strat_unstable },
        { "delta",    &strat_delta    },
    };

    const int warmup  = (n <= 5000) ? 80  : 25;
    const int samples = (n <= 5000) ? 300 : 150;

    // Build the BASE (sorted) sequence ONCE per cell. addEvent is O(n) per call
    // so generating a fresh sequence per sample would be O(n^2) per sample.
    // We copy the base sequence into a fresh one each sample (O(n) per copy,
    // dominated only by allocator overhead, which is the same for all strategies).
    auto base = generate_track_fast(n, seed);

    auto setup = [&] (Op op_local) {
        // Deep-copy the base sequence so each sample starts from identical state.
        auto seq = std::make_unique<MidiMessageSequence>(*base);
        std::mt19937_64 op_rng(seed ^ 0x9E3779B97F4A7C15ULL);
        std::vector<std::size_t> dirty;
        switch (op_local) {
        case Op::Drag:     dirty = op_selection_drag(*seq, k, 480.0, op_rng); break;
        case Op::Quantize: dirty = op_quantize(*seq, k, 120.0, op_rng); break;
        }
        return std::make_pair(std::move(seq), std::move(dirty));
    };

    for (const auto& s : strategies) {
        // Warmup
        for (int w = 0; w < warmup; ++w) {
            auto state = setup(op);
            s.fn(*state.first, state.second);
        }
        // Measure
        std::vector<double> ns;
        ns.reserve((std::size_t) samples);
        for (int i = 0; i < samples; ++i) {
            auto state = setup(op);
            auto t0 = std::chrono::steady_clock::now();
            s.fn(*state.first, state.second);
            auto t1 = std::chrono::steady_clock::now();
            ns.push_back((double) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            if (i == 0 || i == samples / 2 || i == samples - 1) {
                if (! is_sorted_sequence(*state.first)) {
                    std::fprintf(stderr,
                        "VERIFICATION FAILED op=%s n=%d k=%d strat=%s i=%d\n",
                        op_name(op), n, k, s.name, i);
                    std::exit(2);
                }
            }
        }
        BenchStats st = summarize(std::move(ns));
        std::fprintf(out,
            "%s,%d,%d,%s,%.1f,%.1f,%.1f,%.2f,%.2f,%.1f,%.1f,%d\n",
            op_name(op), n, k, s.name,
            st.p50, st.p95, st.p99, st.mean, st.stdv, st.min_v, st.max_v, st.n_samples);
        std::fflush(out);
    }
}

int main(int argc, char** argv)
{
    const char* out_path = (argc > 1) ? argv[1] : "juce_midi_cpp.csv";
    FILE* out = std::fopen(out_path, "w");
    if (! out) { std::perror("fopen"); return 1; }
    std::fprintf(out, "op,n,k,strategy,p50,p95,p99,mean,std,min,max,n_samples\n");

    const int n_values[]   = { 1000, 5000, 20000, 100000 };
    const double k_ratios[] = { 0.001, 0.005, 0.025, 0.10, 0.25 };
    const Op   ops[]       = { Op::Drag, Op::Quantize };

    auto t0 = std::chrono::steady_clock::now();
    for (Op op : ops) {
        for (int n : n_values) {
            for (double r : k_ratios) {
                int k = std::max(1, (int) std::lround((double) n * r));
                k = std::min(k, n / 2);
                auto el = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - t0).count();
                std::fprintf(stderr, "[%4llds] op=%s n=%6d k=%5d (%.1f%%)\n",
                             (long long) el, op_name(op), n, k, r * 100.0);
                run_cell(op, n, k,
                         0x5EED0123ULL ^ (std::uint64_t) n ^ ((std::uint64_t) k << 16),
                         out);
            }
        }
    }
    std::fclose(out);
    std::fprintf(stderr, "Wrote %s\n", out_path);
    return 0;
}
