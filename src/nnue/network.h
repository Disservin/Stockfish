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


// #include "../evaluate.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
// #include "../uci.h"
#include "nnue_architecture.h"
#include "nnue_accumulator.h"
#include "nnue_common.h"
#include "evaluate_nnue.h"


namespace Stockfish::Eval {
#define EvalFileDefaultNameBig "nn-1ceb1ade0001.nnue"
#define EvalFileDefaultNameSmall "nn-baff1ede1f90.nnue"
}


namespace Stockfish::Eval::NNUE {

struct EvalFile {
    // UCI option name
    std::string optionName;
    // Default net name, will use one of the macros above
    std::string defaultName;
    // Selected net name, either via uci option or default
    std::string current;
    // Net description extracted from the net file
    std::string netDescription;
};


struct NnueEvalTrace {
    static_assert(LayerStacks == PSQTBuckets);

    Value       psqt[LayerStacks];
    Value       positional[LayerStacks];
    std::size_t correctBucket;
};


template<NetSize NetSize, typename Arch, typename Transformer>
class Network {
   public:
    Network(EvalFile file) :
        evalFile(file) {}

    void load(const std::string& rootDirectory, std::string user_eval_file_path);

    void loadUserNet(const std::string& dir, const std::string& user_eval_file_path);
    void loadInternal();
    bool save_eval(const std::optional<std::string>& filename);


    Value evaluate(const Position& pos, bool adjusted, int* complexity, bool psqtOnly);
    void  verify(std::string user_eval_file);

    void          hint_common_access(const Position& pos, bool psqtOnl);
    NnueEvalTrace trace_evaluate(const Position& pos);

   private:
    bool save(std::ostream& stream, const std::string& name, const std::string& netDescription);
    std::optional<std::string> load(std::istream& stream);
    void                       initialize();
    // Read network header
    static bool read_header(std::istream& stream, std::uint32_t* hashValue, std::string* desc);

    // Write network header
    static bool
         write_header(std::ostream& stream, std::uint32_t hashValue, const std::string& desc);
    bool read_parameters(std::istream& stream, std::string& netDescription);

    bool write_parameters(std::ostream& stream, const std::string& netDescription);

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

using NetworkSmall =
  Network<NetSize::Small,
          NetworkArchitecture<TransformedFeatureDimensionsSmall, L2Small, L3Small>,
          FeatureTransformer<TransformedFeatureDimensionsSmall, &StateInfo::accumulatorSmall>>;


struct Networks {
    Networks(NetworkBig&& nB, NetworkSmall&& nS) :
        networkBig(nB),
        networkSmall(nS) {}

    NetworkBig&   networkBig;
    NetworkSmall& networkSmall;
};


}  // namespace Stockfish


#endif