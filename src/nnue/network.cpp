/*
  Stockfish, a UCI chess playing engine derived from Glaurung 2.1
  Copyright (C) 2004-2026 The Stockfish developers (see AUTHORS file)

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

#include "network.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>
#include <sstream>

#define INCBIN_SILENCE_BITCODE_WARNING
#include "../incbin/incbin.h"

#include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_misc.h"
#include "nnz_helper.h"

// Macro to embed the default efficiently updatable neural network (NNUE) file
// data in the engine binary (using incbin.h, by Dale Weiler).
// This macro invocation will declare the following three variables
//     const unsigned char        gEmbeddedNNUEData[];  // a pointer to the embedded data
//     const unsigned char *const gEmbeddedNNUEEnd;     // a marker to the end
//     const unsigned int         gEmbeddedNNUESize;    // the size of the embedded file
// Note that this does not work in Microsoft Visual Studio.
#if !defined(UNIVERSAL_BINARY) && !defined(_MSC_VER) && !defined(NNUE_EMBEDDING_OFF)
INCBIN(EmbeddedNNUE, EvalFileDefaultName);
#elif defined(UNIVERSAL_BINARY_MACOS_X86_SLICE)
// Determined at runtime, see universal/nnue_embed.cpp
extern const unsigned char* const gEmbeddedNNUEData;
extern const unsigned int         gEmbeddedNNUESize;
#elif defined(UNIVERSAL_BINARY)
extern const unsigned char gEmbeddedNNUEData[];
extern const unsigned int  gEmbeddedNNUESize;
#else
const unsigned char gEmbeddedNNUEData[1] = {0x0};
const unsigned int  gEmbeddedNNUESize    = 1;
#endif

namespace Stockfish::Eval::NNUE {

namespace {

struct EvaluationBuffer {
    static constexpr u64 alignment = CacheLineSize;

    alignas(alignment) TransformedFeatureType transformedFeatures[FeatureTransformer::BufferSize];

    TransformedFeatureType* data() {
        ASSERT_ALIGNED(transformedFeatures, alignment);
        return transformedFeatures;
    }
};

}  // namespace


namespace Detail {

// Read evaluation function parameters
template<typename T>
bool read_parameters(std::istream& stream, T& reference) {

    u32 header;
    header = read_little_endian<u32>(stream);
    if (!stream || header != T::get_hash_value())
        return false;
    return reference.read_parameters(stream);
}

// Write evaluation function parameters
template<typename T>
bool write_parameters(std::ostream& stream, const T& reference) {

    write_little_endian<u32>(stream, T::get_hash_value());
    return reference.write_parameters(stream);
}

}  // namespace Detail

void Network::load(const std::string& rootDirectory, std::string evalfilePath) {
    const std::string requestedPath = resolve_evalfile_path(std::move(evalfilePath));

#if defined(DEFAULT_NNUE_DIRECTORY)
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory,
                                     stringify(DEFAULT_NNUE_DIRECTORY)};
#else
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory};
#endif

    for (const auto& directory : dirs)
    {
        if (std::string(evalFile.current) == requestedPath)
            break;

        std::optional<LoadedNetwork> loaded;

        if (directory != "<internal>")
            loaded = load_user_net(directory, requestedPath);
        else if (requestedPath == std::string(evalFile.defaultName))
            loaded = load_internal();

        if (loaded)
        {
            apply_loaded_network(std::move(*loaded));
            break;
        }
    }
}


bool Network::save(const std::optional<std::string>& filename) const {
    const auto actualFilename = resolve_save_target(filename);
    if (!actualFilename)
    {
        sync_cout
          << "Failed to export a net. A non-embedded net can only be saved if the filename is specified"
          << sync_endl;
        return false;
    }

    std::ofstream stream(*actualFilename, std::ios_base::binary);
    const bool    saved = save(stream, evalFile.netDescription);

    sync_cout << (saved ? "Network saved successfully to " + *actualFilename
                        : "Failed to export a net")
              << sync_endl;
    return saved;
}


NetworkOutput Network::evaluate(const Position&    pos,
                                AccumulatorStack&  accumulatorStack,
                                AccumulatorCaches& cache) const {
    EvaluationBuffer buffer;

    const int bucket = get_bucket(pos);
    return evaluate_bucket(pos, accumulatorStack, cache, bucket, buffer.data());
}


NetworkVerificationResult Network::verify(std::string evalfilePath) const {
    const std::string requestedPath = resolve_evalfile_path(std::move(evalfilePath));
    const std::string loadedPath    = std::string(evalFile.current);

    if (loadedPath == requestedPath)
    {
        std::ostringstream info;
        const usize size = (sizeof(featureTransformer) + sizeof(NetworkArchitecture) * LayerStacks);
        const std::string sizeMiB = std::to_string(size / (1024 * 1024));

        const auto transformedFeatureDimensions = network[0].TransformedFeatureDimensions;
        const auto fc0Outputs                   = network[0].FC_0_OUTPUTS;
        const auto fc1Outputs                   = network[0].FC_1_OUTPUTS;

        const std::string architecture = std::to_string(featureTransformer.InputDimensions) + ", "
                                       + std::to_string(transformedFeatureDimensions) + ", "
                                       + std::to_string(fc0Outputs) + ", "
                                       + std::to_string(fc1Outputs) + ", 1";

        info << "NNUE evaluation using " << loadedPath                //
             << " (" << sizeMiB << "MiB, (" << architecture << "))";  //

        return {true, requestedPath, loadedPath, info.str(), {}};
    }

    std::ostringstream error;

    error
      << "ERROR: Network evaluation parameters compatible with the engine must be available.\n"
      << "ERROR: The network file " << requestedPath << " was not loaded successfully.\n"
      << "ERROR: The UCI option EvalFile might need to specify the full path, including the directory name, to the network file.\n"
      << "ERROR: The default net can be downloaded from: https://tests.stockfishchess.org/api/nn/"
      << evalFile.defaultName.c_str() << '\n'
      << "ERROR: The engine will be terminated now.\n";

    return {false, requestedPath, loadedPath, {}, error.str()};
}


NnueEvalTrace Network::trace_evaluate(const Position&    pos,
                                      AccumulatorStack&  accumulatorStack,
                                      AccumulatorCaches& cache) const {
    EvaluationBuffer buffer;

    NnueEvalTrace t{};
    t.correctBucket = get_bucket(pos);
    for (IndexType bucket = 0; bucket < LayerStacks; ++bucket)
    {
        std::tie(t.psqt[bucket], t.positional[bucket]) =
          evaluate_bucket(pos, accumulatorStack, cache, bucket, buffer.data());
    }

    return t;
}


std::optional<LoadedNetwork> Network::load_user_net(const std::string& dir,
                                                    const std::string& evalfilePath) {
    std::ifstream stream(dir + evalfilePath, std::ios::binary);
    auto          description = load(stream);

    if (!description)
        return std::nullopt;

    return LoadedNetwork{evalfilePath, std::move(*description)};
}


std::optional<LoadedNetwork> Network::load_internal() {
    // C++ way to prepare a buffer for a memory stream
    class MemoryBuffer: public std::basic_streambuf<char> {
       public:
        MemoryBuffer(char* p, usize n) {
            setg(p, p, p + n);
            setp(p, p + n);
        }
    };

#ifdef UNIVERSAL_BINARY_MACOS_X86_SLICE
    if (gEmbeddedNNUEData == nullptr)  // failed embedded load
        return std::nullopt;
#endif

    MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(gEmbeddedNNUEData)),
                        usize(gEmbeddedNNUESize));

    std::istream stream(&buffer);
    auto         description = load(stream);

    if (!description)
        return std::nullopt;

    return LoadedNetwork{std::string(evalFile.defaultName), std::move(*description)};
}


void Network::apply_loaded_network(LoadedNetwork&& loaded) {
    evalFile.current        = std::move(loaded.path);
    evalFile.netDescription = std::move(loaded.description);
    initialized             = true;
}


bool Network::save(std::ostream& stream, const std::string& netDescription) const {
    return write_parameters(stream, netDescription);
}


NetworkOutput Network::evaluate_bucket(const Position&         pos,
                                       AccumulatorStack&       accumulatorStack,
                                       AccumulatorCaches&      cache,
                                       IndexType               bucket,
                                       TransformedFeatureType* transformedFeatures) const {
    NNZInfo<L1> nnzInfo;

    const auto psqt       = featureTransformer.transform(pos, accumulatorStack, cache,
                                                         transformedFeatures, bucket, nnzInfo);
    const auto positional = network[bucket].propagate(transformedFeatures, nnzInfo);
    return {static_cast<Value>(psqt / OutputScale), static_cast<Value>(positional / OutputScale)};
}


std::optional<std::string> Network::load(std::istream& stream) { return read_parameters(stream); }


std::string Network::resolve_evalfile_path(std::string evalfilePath) const {
    return evalfilePath.empty() ? std::string(evalFile.defaultName) : std::move(evalfilePath);
}


std::optional<std::string>
Network::resolve_save_target(const std::optional<std::string>& filename) const {
    if (filename.has_value())
        return filename;

    if (std::string(evalFile.current) != std::string(evalFile.defaultName))
        return std::nullopt;

    return std::string(evalFile.defaultName);
}


usize Network::get_content_hash() const {
    if (!initialized)
        return 0;

    usize h = 0;
    hash_combine(h, featureTransformer);
    for (auto&& layerstack : network)
        hash_combine(h, layerstack);
    hash_combine(h, evalFile);
    return h;
}

// Read network header
std::optional<NetworkHeader> Network::read_header(std::istream& stream) const {
    const u32 version   = read_little_endian<u32>(stream);
    const u32 hashValue = read_little_endian<u32>(stream);
    const u32 size      = read_little_endian<u32>(stream);

    if (!stream || version != Version)
        return std::nullopt;

    NetworkHeader header{hashValue, std::string(size, '\0')};
    stream.read(header.description.data(), size);

    if (stream.fail())
        return std::nullopt;

    return header;
}


// Write network header
bool Network::write_header(std::ostream& stream, u32 hashValue, const std::string& desc) const {
    write_little_endian<u32>(stream, Version);
    write_little_endian<u32>(stream, hashValue);
    write_little_endian<u32>(stream, u32(desc.size()));
    stream.write(&desc[0], desc.size());
    return !stream.fail();
}


std::optional<std::string> Network::read_parameters(std::istream& stream) {
    const auto header = read_header(stream);
    if (!header)
        return std::nullopt;

    if (header->hashValue != Network::hash)
        return std::nullopt;

    if (!Detail::read_parameters(stream, featureTransformer))
        return std::nullopt;

    for (usize i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::read_parameters(stream, network[i]))
            return std::nullopt;
    }

    if (!(stream && stream.peek() == std::ios::traits_type::eof()))
        return std::nullopt;

    return header->description;
}


bool Network::write_parameters(std::ostream& stream, const std::string& netDescription) const {
    if (!write_header(stream, Network::hash, netDescription))
        return false;
    if (!Detail::write_parameters(stream, featureTransformer))
        return false;
    for (usize i = 0; i < LayerStacks; ++i)
    {
        if (!Detail::write_parameters(stream, network[i]))
            return false;
    }
    return bool(stream);
}

}  // namespace Stockfish::Eval::NNUE
