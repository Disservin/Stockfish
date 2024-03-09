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

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_feature_transformer.h"
#include "nnue_misc.h"

namespace Stockfish::Eval::NNUE {

template<NetSize NetSize, typename Arch, typename Transformer>
class Network {
   public:
    Network(EvalFile file) :
        evalFile(file) {}

    void load(const std::string& rootDirectory, std::string user_eval_file_path);

    void loadUserNet(const std::string& dir, const std::string& user_eval_file_path);
    void loadInternal();
    bool save_eval(const std::optional<std::string>& filename);


    Value evaluate(const Position& pos,
                   bool            adjusted   = false,
                   int*            complexity = nullptr,
                   bool            psqtOnly   = false);
    void  verify(std::string user_eval_file);

    void          hint_common_access(const Position& pos, bool psqtOnl);
    NnueEvalTrace trace_evaluate(const Position& pos);

   private:
    void initialize();

    bool save(std::ostream& stream, const std::string& name, const std::string& netDescription);
    std::optional<std::string> load(std::istream& stream);

    bool read_header(std::istream& stream, std::uint32_t* hashValue, std::string* desc);
    bool write_header(std::ostream& stream, std::uint32_t hashValue, const std::string& desc);

    bool read_parameters(std::istream& stream, std::string& netDescription);
    bool write_parameters(std::ostream& stream, const std::string& netDescription);

    // Input feature converter
    LargePagePtr<Transformer> featureTransformer;

    // Evaluation function
    AlignedPtr<Arch> network[LayerStacks];

    EvalFile evalFile;
};

using SmallFeatureTransformer =
  FeatureTransformer<TransformedFeatureDimensionsSmall, &StateInfo::accumulatorSmall>;
using SmallNetworkArchitecture =
  NetworkArchitecture<TransformedFeatureDimensionsSmall, L2Small, L3Small>;

using BigFeatureTransformer =
  FeatureTransformer<TransformedFeatureDimensionsBig, &StateInfo::accumulatorBig>;
using BigNetworkArchitecture = NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>;

using NetworkBig = Network<NetSize::Big, BigNetworkArchitecture, BigFeatureTransformer>;

using NetworkSmall = Network<NetSize::Small, SmallNetworkArchitecture, SmallFeatureTransformer>;


struct Networks {
    Networks(NetworkBig&& nB, NetworkSmall&& nS) :
        networkBig(std::move(nB)),
        networkSmall(std::move(nS)) {}

    NetworkBig   networkBig;
    NetworkSmall networkSmall;
};

}  // namespace Stockfish

#endif