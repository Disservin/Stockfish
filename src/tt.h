/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2025 The Stockfish developers (see AUTHORS file)

  Stockfish is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Stockfish is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef TT_H_INCLUDED
#define TT_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>

#include "memory.h"
#include "types.h"

namespace Stockfish {

class ThreadPool;
struct Cluster;

template<typename T>
class AtomicRelaxed {
   private:
    std::atomic<T> value;

   public:
    AtomicRelaxed() = default;
    AtomicRelaxed(T init_value) :
        value(init_value) {}

    AtomicRelaxed(const AtomicRelaxed&)            = delete;
    AtomicRelaxed& operator=(const AtomicRelaxed&) = delete;

    T load() const { return value.load(std::memory_order_relaxed); }

    void store(T new_value) { value.store(new_value, std::memory_order_relaxed); }

    operator T() const { return load(); }

    AtomicRelaxed& operator=(T new_value) {
        store(new_value);
        return *this;
    }
};

struct TTData8 {
    uint8_t  depth8;
    uint8_t  genBound8;
    uint16_t key;
    int16_t  value16;
    int16_t  eval16;


    bool is_occupied() const;

    // The returned age is a multiple of TranspositionTable::GENERATION_DELTA
    uint8_t relative_age(const uint8_t generation8) const;

    // Convert internal bitfields to external types
    // TTData read() const {
    //     return TTData{Move(move16),           Value(value16),
    //                   Value(eval16),          Depth(depth8 + DEPTH_ENTRY_OFFSET),
    //                   Bound(genBound8 & 0x3), bool(genBound8 & 0x4)};
    // }

    uint64_t packed() const {
        uint64_t packed_data;
        std::memcpy(&packed_data, this, sizeof(packed_data));
        return packed_data;
    }

    static TTData8 unpack(uint64_t packed_data) {
        TTData8 data;
        std::memcpy(&data, &packed_data, sizeof(data));
        return data;
    }
};


// There is only one global hash table for the engine and all its threads. For chess in particular, we even allow racy
// updates between threads to and from the TT, as taking the time to synchronize access would cost thinking time and
// thus elo. As a hash table, collisions are possible and may cause chess playing issues (bizarre blunders, faulty mate
// reports, etc). Fixing these also loses elo; however such risk decreases quickly with larger TT size.
//
// `probe` is the primary method: given a board position, we lookup its entry in the table, and return a tuple of:
//   1) whether the entry already has this position
//   2) a copy of the prior data (if any) (may be inconsistent due to read races)
//   3) a writer object to this entry
// The copied data and the writer are separated to maintain clear boundaries between local vs global objects.


// A copy of the data already in the entry (possibly collided). `probe` may be racy, resulting in inconsistent data.
struct TTData {
    Move  move;
    Value value, eval;
    Depth depth;
    Bound bound;
    bool  is_pv;

    TTData() = delete;

    // clang-format off
    TTData(Move m, Value v, Value ev, Depth d, Bound b, bool pv) :
        move(m),
        value(v),
        eval(ev),
        depth(d),
        bound(b),
        is_pv(pv) {};
    // clang-format on

    TTData(const uint16_t move_, const TTData8& data) :
        move(Move(move_)),
        value(Value(data.value16)),
        eval(Value(data.eval16)),
        depth(Depth(data.depth8 + DEPTH_ENTRY_OFFSET)),
        bound(Bound(data.genBound8 & 0x3)),
        is_pv(bool(data.genBound8 & 0x4)) {}
};


struct TTWriter {
   public:
    void write(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8);

   private:
    friend class TranspositionTable;
    AtomicRelaxed<uint16_t>* data_2_atomic;
    AtomicRelaxed<uint64_t>* data_atomic;

    TTWriter(AtomicRelaxed<uint16_t>* move_ptr, AtomicRelaxed<uint64_t>* data_ptr);
};


class TranspositionTable {

   public:
    ~TranspositionTable() { aligned_large_pages_free(table); }

    void resize(size_t mbSize, ThreadPool& threads);  // Set TT size
    void clear(ThreadPool& threads);                  // Re-initialize memory, multithreaded
    int  hashfull(int maxAge = 0)
      const;  // Approximate what fraction of entries (permille) have been written to during this root search

    void
    new_search();  // This must be called at the beginning of each root search to track entry aging
    uint8_t generation() const;  // The current age, used when writing new data to the TT
    std::tuple<bool, TTData, TTWriter>
    probe(const Key key) const;  // The main method, whose retvals separate local vs global objects
    void prefetch(const Key key) const;

   private:
    size_t   clusterCount;
    Cluster* table = nullptr;

    uint8_t generation8 = 0;  // Size must be not bigger than TTEntry::genBound8
};

}  // namespace Stockfish

#endif  // #ifndef TT_H_INCLUDED
