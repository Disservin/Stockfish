/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2024 The Stockfish developers (see AUTHORS file)

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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string_view>
#include <type_traits>

#include "../incbin/incbin.h"


#include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "../uci.h"
#include "nnue_architecture.h"
#include "nnue_accumulator.h"
#include "nnue_common.h"
#include "evaluate_nnue.h"

namespace {
// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUEBig, EvalFileDefaultNameBig);
INCBIN(EmbeddedNNUESmall, EvalFileDefaultNameSmall);
#else
const unsigned char        gEmbeddedNNUEBigData[1]   = {0x0};
const unsigned char* const gEmbeddedNNUEBigEnd       = &gEmbeddedNNUEBigData[1];
const unsigned int         gEmbeddedNNUEBigSize      = 1;
const unsigned char        gEmbeddedNNUESmallData[1] = {0x0};
const unsigned char* const gEmbeddedNNUESmallEnd     = &gEmbeddedNNUESmallData[1];
const unsigned int         gEmbeddedNNUESmallSize    = 1;
#endif
}

namespace Stockfish::Eval::NNUE {

namespace Detail {

// Initialize the evaluation function parameters
template<typename T>
void initialize(AlignedPtr<T>& pointer) {

    pointer.reset(reinterpret_cast<T*>(std_aligned_alloc(alignof(T), sizeof(T))));
    std::memset(pointer.get(), 0, sizeof(T));
}

template<typename T>
void initialize(LargePagePtr<T>& pointer) {

    static_assert(alignof(T) <= 4096,
                  "aligned_large_pages_alloc() may fail for such a big alignment requirement of T");
    pointer.reset(reinterpret_cast<T*>(aligned_large_pages_alloc(sizeof(T))));
    std::memset(pointer.get(), 0, sizeof(T));
}

// Read evaluation function parameters
template<typename T>
bool read_parameters(std::istream& stream, T& reference) {

    std::uint32_t header;
    header = read_little_endian<std::uint32_t>(stream);
    if (!stream || header != T::get_hash_value())
        return false;
    return reference.read_parameters(stream);
}

// Write evaluation function parameters
template<typename T>
bool write_parameters(std::ostream& stream, const T& reference) {

    write_little_endian<std::uint32_t>(stream, T::get_hash_value());
    return reference.write_parameters(stream);
}

}  // namespace Detail


template<NetSize NetSize, typename Arch, typename Transformer>
class Network {
   public:
    void load(const EvalFile& user_eval_file, const std::string& path) {
        std::ifstream stream(path, std::ios::binary);
        auto          description = load(stream);

        if (description.has_value())
        {
            evalFile.current        = user_eval_file;
            evalFile.netDescription = description.value();
        }
    }

    void loadInternal() {
        // C++ way to prepare a buffer for a memory stream
        class MemoryBuffer: public std::basic_streambuf<char> {
           public:
            MemoryBuffer(char* p, size_t n) {
                setg(p, p, p + n);
                setp(p, p + n);
            }
        };

        const auto embeddedData = NetSize == Small ? gEmbeddedNNUESmallData : gEmbeddedNNUEBigData;
        const auto embeddedSize = NetSize == Small ? gEmbeddedNNUESmallSize : gEmbeddedNNUEBigSize;

        MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(embeddedData)),
                            size_t(embeddedSize));
        (void) gEmbeddedNNUEBigEnd;  // Silence warning on unused variable
        (void) gEmbeddedNNUESmallEnd;

        std::istream stream(&buffer);
        auto         description = load(stream);

        if (description.has_value())
        {
            // evalFile.current        = user_eval_file;
            evalFile.netDescription = description.value();
        }
    }

    std::optional<std::string> load(std::istream& stream) {
        initialize();
        std::string description;

        return read_parameters(stream, description) ? std::make_optional(description)
                                                    : std::nullopt;
    }

    bool save(std::ostream& stream, const std::string& name, const std::string& netDescription) {
        if (name.empty() || name == "None")
            return false;

        return write_parameters(stream, NetSize, netDescription);
    }

    Value evaluate(const Position& pos, bool adjusted, int* complexity, bool psqtOnly) {

        // We manually align the arrays on the stack because with gcc < 9.3
        // overaligning stack variables with alignas() doesn't work correctly.

        constexpr uint64_t alignment = CacheLineSize;
        constexpr int      delta     = 24;

#if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
        TransformedFeatureType transformedFeaturesUnaligned
          [FeatureTransformer < NetSize == Small ? TransformedFeatureDimensionsSmall
                                                 : TransformedFeatureDimensionsBig,
           nullptr > ::BufferSize + alignment / sizeof(TransformedFeatureType)];

        auto* transformedFeatures = align_ptr_up<alignment>(&transformedFeaturesUnaligned[0]);
#else

        alignas(alignment)
          TransformedFeatureType transformedFeatures[FeatureTransformer < NetSize == Small
                                                       ? TransformedFeatureDimensionsSmall
                                                       : TransformedFeatureDimensionsBig,
                                                     nullptr > ::BufferSize];
#endif

        ASSERT_ALIGNED(transformedFeatures, alignment);

        const int  bucket = (pos.count<ALL_PIECES>() - 1) / 4;
        const auto psqt = featureTransformer->transform(pos, transformedFeatures, bucket, psqtOnly);
        const auto positional = !psqtOnly ? (network[bucket]->propagate(transformedFeatures)) : 0;

        if (complexity)
            *complexity = !psqtOnly ? std::abs(psqt - positional) / OutputScale : 0;

        // Give more value to positional evaluation when adjusted flag is set
        if (adjusted)
            return static_cast<Value>(((1024 - delta) * psqt + (1024 + delta) * positional)
                                      / (1024 * OutputScale));
        else
            return static_cast<Value>((psqt + positional) / OutputScale);
    }


   private:
    void initialize() {
        Detail::initialize(featureTransformer);
        for (std::size_t i = 0; i < LayerStacks; ++i)
            Detail::initialize(network[i]);
    }

    // Read network header
    static bool read_header(std::istream& stream, std::uint32_t* hashValue, std::string* desc) {
        std::uint32_t version, size;

        version    = read_little_endian<std::uint32_t>(stream);
        *hashValue = read_little_endian<std::uint32_t>(stream);
        size       = read_little_endian<std::uint32_t>(stream);
        if (!stream || version != Version)
            return false;
        desc->resize(size);
        stream.read(&(*desc)[0], size);
        return !stream.fail();
    }

    // Write network header
    static bool
    write_header(std::ostream& stream, std::uint32_t hashValue, const std::string& desc) {
        write_little_endian<std::uint32_t>(stream, Version);
        write_little_endian<std::uint32_t>(stream, hashValue);
        write_little_endian<std::uint32_t>(stream, std::uint32_t(desc.size()));
        stream.write(&desc[0], desc.size());
        return !stream.fail();
    }

    bool read_parameters(std::istream& stream, std::string& netDescription) {
        std::uint32_t hashValue;
        if (!read_header(stream, &hashValue, &netDescription))
            return false;
        if (hashValue != HashValue[NetSize])
            return false;
        if (!Detail::read_parameters(stream, *featureTransformer))
            return false;
        for (std::size_t i = 0; i < LayerStacks; ++i)
        {
            if (!Detail::read_parameters(stream, *(network[i])))
                return false;
        }
        return stream && stream.peek() == std::ios::traits_type::eof();
    }

    bool write_parameters(std::ostream& stream, const std::string& netDescription) {

        if (!write_header(stream, HashValue[NetSize], netDescription))
            return false;
        if (!Detail::write_parameters(stream, *featureTransformer))
            return false;
        for (std::size_t i = 0; i < LayerStacks; ++i)
        {
            if (!Detail::write_parameters(stream, *(network[i])))
                return false;
        }
        return bool(stream);
    }


    // Input feature converter
    LargePagePtr<Transformer> featureTransformer;

    // Evaluation function
    AlignedPtr<Arch> network[LayerStacks];

    EvalFile evalFile;
};

using NetworkBig =
  Network<NetSize::Big,
          NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>,
          FeatureTransformer<TransformedFeatureDimensionsBig, &StateInfo::accumulatorBig>>;

}  // namespace Stockfish


#endif