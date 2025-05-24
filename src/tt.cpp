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

#include "tt.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memory.h"
#include "misc.h"
#include "syzygy/tbprobe.h"
#include "thread.h"

namespace Stockfish {

struct TTData8 {
    uint8_t  depth8;
    uint8_t  genBound8;
    uint16_t move16;
    int16_t  value16;
    int16_t  eval16;


    bool is_occupied() const;

    // The returned age is a multiple of TranspositionTable::GENERATION_DELTA
    uint8_t relative_age(const uint8_t generation8) const;

    // Convert internal bitfields to external types
    TTData read() const {
        return TTData{Move(move16),           Value(value16),
                      Value(eval16),          Depth(depth8 + DEPTH_ENTRY_OFFSET),
                      Bound(genBound8 & 0x3), bool(genBound8 & 0x4)};
    }
};

// `genBound8` is where most of the details are. We use the following constants to manipulate 5 leading generation bits
// and 3 trailing miscellaneous bits.

// These bits are reserved for other things.
static constexpr unsigned GENERATION_BITS = 3;
// increment for generation field
static constexpr int GENERATION_DELTA = (1 << GENERATION_BITS);
// cycle length
static constexpr int GENERATION_CYCLE = 255 + GENERATION_DELTA;
// mask to pull out generation number
static constexpr int GENERATION_MASK = (0xFF << GENERATION_BITS) & 0xFF;

// DEPTH_ENTRY_OFFSET exists because 1) we use `bool(depth8)` as the occupancy check, but
// 2) we need to store negative depths for QS. (`depth8` is the only field with "spare bits":
// we sacrifice the ability to store depths greater than 1<<8 less the offset, as asserted in `save`.)
bool TTData8::is_occupied() const { return bool(depth8); }

uint8_t TTData8::relative_age(const uint8_t generation8) const {
    // Due to our packed storage format for generation and its cyclic
    // nature we add GENERATION_CYCLE (256 is the modulus, plus what
    // is needed to keep the unrelated lowest n bits from affecting
    // the result) to calculate the entry age correctly even after
    // generation8 overflows into the next cycle.
    return (GENERATION_CYCLE + generation8 - genBound8) & GENERATION_MASK;
}


static_assert(sizeof(TTData8) == 8, "TTData8 must be exactly 8 bytes");

struct TTEntry {
    TTEntry(std::atomic<uint16_t>* key_ptr, std::atomic<uint64_t>* data_ptr) :
        key_atomic(key_ptr),
        data_atomic(data_ptr) {}

    TTData read() const {
        // uint16_t key         = key_atomic->load(std::memory_order_relaxed);
        uint64_t packed_data = data_atomic->load(std::memory_order_relaxed);

        TTData8 data;
        std::memcpy(&data, &packed_data, sizeof(data));

        return TTData{Move(data.move16),           Value(data.value16),
                      Value(data.eval16),          Depth(data.depth8 + DEPTH_ENTRY_OFFSET),
                      Bound(data.genBound8 & 0x3), bool(data.genBound8 & 0x4)};
    }

    TTData8 read_raw() const {
        uint64_t packed_data = data_atomic->load(std::memory_order_relaxed);
        TTData8  data;
        std::memcpy(&data, &packed_data, sizeof(data));
        return data;
    }

   private:
    friend class TranspositionTable;

    std::atomic<uint16_t>* key_atomic;
    std::atomic<uint64_t>* data_atomic;
};


TTWriter::TTWriter(std::atomic<uint16_t>* key_ptr, std::atomic<uint64_t>* data_ptr) :
    key_atomic(key_ptr),
    data_atomic(data_ptr) {}

// Populates the TTEntry with a new node's data, possibly
// overwriting an old position.
void TTWriter::write(
  Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t generation8) {
    uint16_t current_key    = key_atomic->load(std::memory_order_relaxed);
    uint64_t current_packed = data_atomic->load(std::memory_order_relaxed);

    TTData8 current_data;
    std::memcpy(&current_data, &current_packed, sizeof(current_data));

    bool update = false;

    // Preserve the old ttmove if we don't have a new one
    if (m || uint16_t(k) != current_key)
    {
        update              = true;
        current_data.move16 = m.raw();
    }

    // Overwrite less valuable entries (cheapest checks first)
    if (b == BOUND_EXACT || uint16_t(k) != current_key
        || d - DEPTH_ENTRY_OFFSET + 2 * pv > current_data.depth8 - 4
        || current_data.relative_age(generation8))
    {
        assert(d > DEPTH_ENTRY_OFFSET);
        assert(d < 256 + DEPTH_ENTRY_OFFSET);

        current_data.depth8    = uint8_t(d - DEPTH_ENTRY_OFFSET);
        current_data.genBound8 = uint8_t(generation8 | uint8_t(pv) << 2 | b);
        current_data.value16   = int16_t(v);
        current_data.eval16    = int16_t(ev);

        uint64_t packed_new_data;
        std::memcpy(&packed_new_data, &current_data, sizeof(packed_new_data));

        key_atomic->store(uint16_t(k), std::memory_order_relaxed);
        data_atomic->store(packed_new_data, std::memory_order_relaxed);

        return;
    }

    if (update)
    {
        uint64_t packed_new_data;
        std::memcpy(&packed_new_data, &current_data, sizeof(packed_new_data));

        data_atomic->store(packed_new_data, std::memory_order_relaxed);
    }
}

// A TranspositionTable is an array of Cluster, of size clusterCount. Each cluster consists of ClusterSize number
// of TTEntry. Each non-empty TTEntry contains information on exactly one position. The size of a Cluster should
// divide the size of a cache line for best performance, as the cacheline is prefetched when possible.

static constexpr int ClusterSize = 3;

struct Cluster {
    std::atomic<uint16_t> keys[ClusterSize];
    std::atomic<uint64_t> data[ClusterSize];
};

static_assert(sizeof(Cluster) == 32, "Suboptimal Cluster size");


// Sets the size of the transposition table,
// measured in megabytes. Transposition table consists
// of clusters and each cluster consists of ClusterSize number of TTEntry.
void TranspositionTable::resize(size_t mbSize, ThreadPool& threads) {
    aligned_large_pages_free(table);

    clusterCount = mbSize * 1024 * 1024 / sizeof(Cluster);

    table = static_cast<Cluster*>(aligned_large_pages_alloc(clusterCount * sizeof(Cluster)));

    if (!table)
    {
        std::cerr << "Failed to allocate " << mbSize << "MB for transposition table." << std::endl;
        exit(EXIT_FAILURE);
    }

    clear(threads);
}


// Initializes the entire transposition table to zero,
// in a multi-threaded way.
void TranspositionTable::clear(ThreadPool& threads) {
    generation8              = 0;
    const size_t threadCount = threads.num_threads();

    for (size_t i = 0; i < threadCount; ++i)
    {
        threads.run_on_thread(i, [this, i, threadCount]() {
            // Each thread will zero its part of the hash table
            const size_t stride = clusterCount / threadCount;
            const size_t start  = stride * i;
            const size_t len    = i + 1 != threadCount ? stride : clusterCount - start;

            for (size_t j = start; j < start + len; ++j)
            {
                for (int k = 0; k < ClusterSize; ++k)
                {
                    table[j].keys[k].store(0, std::memory_order_relaxed);
                    table[j].data[k].store(0, std::memory_order_relaxed);
                }
            }
        });
    }

    for (size_t i = 0; i < threadCount; ++i)
        threads.wait_on_thread(i);
}


// Returns an approximation of the hashtable
// occupation during a search. The hash is x permill full, as per UCI protocol.
// Only counts entries which match the current generation.
int TranspositionTable::hashfull(int maxAge) const {
    int maxAgeInternal = maxAge << GENERATION_BITS;
    int cnt            = 0;
    for (int i = 0; i < 1000; ++i)
    {
        for (int j = 0; j < ClusterSize; ++j)
        {
            TTData8 entry = (TTEntry(&table[i].keys[j], &table[i].data[j]).read_raw());

            cnt += entry.is_occupied() && entry.relative_age(generation8) <= maxAgeInternal;
        }
    }

    return cnt / ClusterSize;
}


void TranspositionTable::new_search() {
    // increment by delta to keep lower bits as is
    generation8 += GENERATION_DELTA;
}


uint8_t TranspositionTable::generation() const { return generation8; }


// Looks up the current position in the transposition
// table. It returns true if the position is found.
// Otherwise, it returns false and a pointer to an empty or least valuable TTEntry
// to be replaced later. The replace value of an entry is calculated as its depth
// minus 8 times its relative age. TTEntry t1 is considered more valuable than
// TTEntry t2 if its replace value is greater than that of t2.
std::tuple<bool, TTData, TTWriter> TranspositionTable::probe(const Key key) const {
    Cluster*       cluster = &table[mul_hi64(key, clusterCount)];
    const uint16_t key16   = uint16_t(key);  // Use the low 16 bits as key inside the cluster

    for (int i = 0; i < ClusterSize; ++i)
    {
        if (cluster->keys[i].load(std::memory_order_relaxed) == key16)
        {
            // This gap is the main place for read races.
            // After `read()` completes that copy is final, but may be self-inconsistent.

            TTData8 data = TTEntry(&cluster->keys[i], &cluster->data[i]).read_raw();
            return {data.is_occupied(), data.read(),
                    TTWriter(&cluster->keys[i], &cluster->data[i])};
        }
    }

    // Find an entry to be replaced according to the replacement strategy
    int replace_idx = 0;
    for (int i = 1; i < ClusterSize; ++i)
    {

        TTData8 current =
          TTEntry(&cluster->keys[replace_idx], &cluster->data[replace_idx]).read_raw();
        TTData8 candidate = TTEntry(&cluster->keys[i], &cluster->data[i]).read_raw();

        if (current.depth8 - current.relative_age(generation8)
            > candidate.depth8 - candidate.relative_age(generation8))
        {
            replace_idx = i;
        }
    }

    return {false,
            TTData{Move::none(), VALUE_NONE, VALUE_NONE, DEPTH_ENTRY_OFFSET, BOUND_NONE, false},
            TTWriter(&cluster->keys[replace_idx], &cluster->data[replace_idx])};
}


void TranspositionTable::prefetch(const Key key) const {
    Cluster* cluster = &table[mul_hi64(key, clusterCount)];
    Stockfish::prefetch(&cluster->keys[0]);
    Stockfish::prefetch(&cluster->data[0]);
}

}  // namespace Stockfish