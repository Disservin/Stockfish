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

#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include "../types.h"
#include "../misc.h"
#include "nnue_architecture.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace Stockfish {
class Position;
}

namespace Stockfish::Eval::NNUE {

class AccumulatorStack;
struct AccumulatorCaches;

using NetworkOutput = std::tuple<Value, Value>;

struct NetworkHeader {
    u32         hashValue;
    std::string description;
};

struct LoadedNetwork {
    std::string path;
    std::string description;
};

struct NetworkVerificationResult {
    bool        ok;
    std::string requestedPath;
    std::string loadedPath;
    std::string infoMessage;
    std::string errorMessage;
};

// The network must be a trivial type, i.e. the memory must be in-line.
// This is required to allow sharing the network via shared memory, as
// there is no way to run destructors.
class Network {
   public:
    Network(EvalFile file) :
        evalFile(file) {}

    Network(const Network& other) = default;
    Network(Network&& other)      = default;

    Network& operator=(const Network& other) = default;
    Network& operator=(Network&& other)      = default;

    void load(const std::string& rootDirectory, std::string evalfilePath);
    bool save(const std::optional<std::string>& filename) const;

    usize get_content_hash() const;

    NetworkOutput evaluate(const Position&    pos,
                           AccumulatorStack&  accumulatorStack,
                           AccumulatorCaches& cache) const;


    NetworkVerificationResult verify(std::string evalfilePath) const;
    NnueEvalTrace             trace_evaluate(const Position&    pos,
                                             AccumulatorStack&  accumulatorStack,
                                             AccumulatorCaches& cache) const;

   private:
    std::optional<LoadedNetwork> load_user_net(const std::string&, const std::string&);
    std::optional<LoadedNetwork> load_internal();
    void                         apply_loaded_network(LoadedNetwork&& loaded);

    bool                       save(std::ostream&, const std::string&) const;
    std::optional<std::string> load(std::istream&);
    std::string                resolve_evalfile_path(std::string evalfilePath) const;
    std::optional<std::string>
    resolve_save_target(const std::optional<std::string>& filename) const;

    std::optional<NetworkHeader> read_header(std::istream&) const;
    bool                         write_header(std::ostream&, u32, const std::string&) const;

    std::optional<std::string> read_parameters(std::istream&);
    bool                       write_parameters(std::ostream&, const std::string&) const;
    NetworkOutput              evaluate_bucket(const Position&         pos,
                                               AccumulatorStack&       accumulatorStack,
                                               AccumulatorCaches&      cache,
                                               IndexType               bucket,
                                               TransformedFeatureType* transformedFeatures) const;

    int get_bucket(const Position& pos) const { return (pos.count<ALL_PIECES>() - 1) / 4; }

    // Input feature converter
    FeatureTransformer featureTransformer;

    // Evaluation function
    NetworkArchitecture network[LayerStacks];

    EvalFile evalFile;

    bool initialized = false;

    // Hash value of evaluation function structure
    static constexpr u32 hash =
      FeatureTransformer::get_hash_value() ^ NetworkArchitecture::get_hash_value();

    friend struct AccumulatorCaches;
};


}  // namespace Stockfish

template<>
struct std::hash<Stockfish::Eval::NNUE::Network> {
    Stockfish::usize operator()(const Stockfish::Eval::NNUE::Network& network) const noexcept {
        return network.get_content_hash();
    }
};

#endif
