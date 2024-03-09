#include "network.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>

#include "../evaluate.h"
#include "../incbin/incbin.h"
#include "../misc.h"
#include "../position.h"
#include "../types.h"
#include "nnue_architecture.h"
#include "nnue_common.h"
#include "nnue_misc.h"

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

// Hash value of evaluation function structure
constexpr std::uint32_t HashValueMap[2] = {
  BigFeatureTransformer::get_hash_value() ^ BigNetworkArchitecture::get_hash_value(),
  SmallFeatureTransformer::get_hash_value() ^ SmallNetworkArchitecture::get_hash_value()};

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
void Network<NetSize, Arch, Transformer>::load(const std::string& rootDirectory,
                                               std::string        evalfilePath) {
#if defined(DEFAULT_NNUE_DIRECTORY)
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory,
                                     stringify(DEFAULT_NNUE_DIRECTORY)};
#else
    std::vector<std::string> dirs = {"<internal>", "", rootDirectory};
#endif

    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    for (const std::string& directory : dirs)
    {
        if (evalFile.current != evalfilePath)
        {
            if (directory != "<internal>")
            {
                loadUserNet(directory, evalfilePath);
            }

            if (directory == "<internal>" && evalfilePath == evalFile.defaultName)
            {
                loadInternal();
            }
        }
    }
}


template<NetSize NetSize, typename Arch, typename Transformer>
bool Network<NetSize, Arch, Transformer>::save(const std::optional<std::string>& filename) const {
    std::string actualFilename;
    std::string msg;

    if (filename.has_value())
        actualFilename = filename.value();
    else
    {
        if (evalFile.current != (evalFile.defaultName))
        {
            msg = "Failed to export a net. "
                  "A non-embedded net can only be saved if the filename is specified";

            sync_cout << msg << sync_endl;
            return false;
        }

        actualFilename = evalFile.defaultName;
    }

    std::ofstream stream(actualFilename, std::ios_base::binary);
    bool          saved = save(stream, evalFile.current, evalFile.netDescription);

    msg = saved ? "Network saved successfully to " + actualFilename : "Failed to export a net";

    sync_cout << msg << sync_endl;
    return saved;
}


template<NetSize NetSize, typename Arch, typename Transformer>
Value Network<NetSize, Arch, Transformer>::evaluate(const Position& pos,
                                                    bool            adjusted,
                                                    int*            complexity,
                                                    bool            psqtOnly) const {
    // We manually align the arrays on the stack because with gcc < 9.3
    // overaligning stack variables with alignas() doesn't work correctly.

    constexpr uint64_t alignment = CacheLineSize;
    constexpr int      delta     = 24;

#if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
    TransformedFeatureType transformedFeaturesUnaligned
      [FeatureTransformer<Arch::TransformedFeatureDimensions, nullptr>::BufferSize
       + alignment / sizeof(TransformedFeatureType)];

    auto* transformedFeatures = align_ptr_up<alignment>(&transformedFeaturesUnaligned[0]);
#else
    alignas(alignment) TransformedFeatureType transformedFeatures
      [FeatureTransformer<Arch::TransformedFeatureDimensions, nullptr>::BufferSize];
#endif

    ASSERT_ALIGNED(transformedFeatures, alignment);

    const int  bucket = (pos.count<ALL_PIECES>() - 1) / 4;
    const auto psqt   = featureTransformer->transform(pos, transformedFeatures, bucket, psqtOnly);
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


template<NetSize NetSize, typename Arch, typename Transformer>
void Network<NetSize, Arch, Transformer>::verify(std::string evalfilePath) const {
    if (evalfilePath.empty())
        evalfilePath = evalFile.defaultName;

    if (evalFile.current != evalfilePath)
    {
        std::string msg1 =
          "Network evaluation parameters compatible with the engine must be available.";
        std::string msg2 = "The network file " + evalfilePath + " was not loaded successfully.";
        std::string msg3 = "The UCI option EvalFile might need to specify the full path, "
                           "including the directory name, to the network file.";
        std::string msg4 = "The default net can be downloaded from: "
                           "https://tests.stockfishchess.org/api/nn/"
                         + evalFile.defaultName;
        std::string msg5 = "The engine will be terminated now.";

        sync_cout << "info string ERROR: " << msg1 << sync_endl;
        sync_cout << "info string ERROR: " << msg2 << sync_endl;
        sync_cout << "info string ERROR: " << msg3 << sync_endl;
        sync_cout << "info string ERROR: " << msg4 << sync_endl;
        sync_cout << "info string ERROR: " << msg5 << sync_endl;
        exit(EXIT_FAILURE);
    }

    sync_cout << "info string NNUE evaluation using " << evalfilePath << sync_endl;
}


template<NetSize NetSize, typename Arch, typename Transformer>
void Network<NetSize, Arch, Transformer>::hint_common_access(const Position& pos,
                                                             bool            psqtOnl) const {
    featureTransformer->hint_common_access(pos, psqtOnl);
}


template<NetSize NetSize, typename Arch, typename Transformer>
NnueEvalTrace Network<NetSize, Arch, Transformer>::trace_evaluate(const Position& pos) const {
    // We manually align the arrays on the stack because with gcc < 9.3
    // overaligning stack variables with alignas() doesn't work correctly.
    constexpr uint64_t alignment = CacheLineSize;

#if defined(ALIGNAS_ON_STACK_VARIABLES_BROKEN)
    TransformedFeatureType transformedFeaturesUnaligned
      [FeatureTransformer<Arch::TransformedFeatureDimensions, nullptr>::BufferSize
       + alignment / sizeof(TransformedFeatureType)];

    auto* transformedFeatures = align_ptr_up<alignment>(&transformedFeaturesUnaligned[0]);
#else
    alignas(alignment) TransformedFeatureType transformedFeatures
      [FeatureTransformer<Arch::TransformedFeatureDimensions, nullptr>::BufferSize];
#endif

    ASSERT_ALIGNED(transformedFeatures, alignment);

    NnueEvalTrace t{};
    t.correctBucket = (pos.count<ALL_PIECES>() - 1) / 4;
    for (IndexType bucket = 0; bucket < LayerStacks; ++bucket)
    {
        const auto materialist =
          featureTransformer->transform(pos, transformedFeatures, bucket, false);
        const auto positional = network[bucket]->propagate(transformedFeatures);

        t.psqt[bucket]       = static_cast<Value>(materialist / OutputScale);
        t.positional[bucket] = static_cast<Value>(positional / OutputScale);
    }

    return t;
}


template<NetSize NetSize, typename Arch, typename Transformer>
void Network<NetSize, Arch, Transformer>::loadUserNet(const std::string& dir,
                                                      const std::string& evalfilePath) {
    std::ifstream stream(dir + evalfilePath, std::ios::binary);
    auto          description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalfilePath;
        evalFile.netDescription = description.value();
    }
}


template<NetSize NetSize, typename Arch, typename Transformer>
void Network<NetSize, Arch, Transformer>::loadInternal() {
    // C++ way to prepare a buffer for a memory stream
    class MemoryBuffer: public std::basic_streambuf<char> {
       public:
        MemoryBuffer(char* p, size_t n) {
            setg(p, p, p + n);
            setp(p, p + n);
        }
    };

    const auto embeddedData =
      NetSize == NetSize::Small ? gEmbeddedNNUESmallData : gEmbeddedNNUEBigData;
    const auto embeddedSize =
      NetSize == NetSize::Small ? gEmbeddedNNUESmallSize : gEmbeddedNNUEBigSize;

    MemoryBuffer buffer(const_cast<char*>(reinterpret_cast<const char*>(embeddedData)),
                        size_t(embeddedSize));
    (void) gEmbeddedNNUEBigEnd;  // Silence warning on unused variable
    (void) gEmbeddedNNUESmallEnd;

    std::istream stream(&buffer);
    auto         description = load(stream);

    if (description.has_value())
    {
        evalFile.current        = evalFile.defaultName;
        evalFile.netDescription = description.value();
    }
}


template<NetSize NetSize, typename Arch, typename Transformer>
void Network<NetSize, Arch, Transformer>::initialize() {
    Detail::initialize(featureTransformer);
    for (std::size_t i = 0; i < LayerStacks; ++i)
        Detail::initialize(network[i]);
}


template<NetSize NetSize, typename Arch, typename Transformer>
bool Network<NetSize, Arch, Transformer>::save(std::ostream&      stream,
                                               const std::string& name,
                                               const std::string& netDescription) const {
    if (name.empty() || name == "None")
        return false;

    return write_parameters(stream, netDescription);
}


template<NetSize NetSize, typename Arch, typename Transformer>
std::optional<std::string> Network<NetSize, Arch, Transformer>::load(std::istream& stream) {
    initialize();
    std::string description;

    return read_parameters(stream, description) ? std::make_optional(description) : std::nullopt;
}


// Read network header
template<NetSize NetSize, typename Arch, typename Transformer>
bool Network<NetSize, Arch, Transformer>::read_header(std::istream&  stream,
                                                      std::uint32_t* hashValue,
                                                      std::string*   desc) const {
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
template<NetSize NetSize, typename Arch, typename Transformer>
bool Network<NetSize, Arch, Transformer>::write_header(std::ostream&      stream,
                                                       std::uint32_t      hashValue,
                                                       const std::string& desc) const {
    write_little_endian<std::uint32_t>(stream, Version);
    write_little_endian<std::uint32_t>(stream, hashValue);
    write_little_endian<std::uint32_t>(stream, std::uint32_t(desc.size()));
    stream.write(&desc[0], desc.size());
    return !stream.fail();
}


template<NetSize NetSize, typename Arch, typename Transformer>
bool Network<NetSize, Arch, Transformer>::read_parameters(std::istream& stream,
                                                          std::string&  netDescription) const {
    std::uint32_t hashValue;
    if (!read_header(stream, &hashValue, &netDescription))
        return false;
    if (hashValue != HashValueMap[static_cast<std::size_t>(NetSize)])
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


template<NetSize NetSize, typename Arch, typename Transformer>
bool Network<NetSize, Arch, Transformer>::write_parameters(
  std::ostream& stream, const std::string& netDescription) const {
    if (!write_header(stream, HashValueMap[static_cast<std::size_t>(NetSize)], netDescription))
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

// Explicit template instantiation

template class Network<
  NetSize::Big,
  NetworkArchitecture<TransformedFeatureDimensionsBig, L2Big, L3Big>,
  FeatureTransformer<TransformedFeatureDimensionsBig, &StateInfo::accumulatorBig>>;

template class Network<
  NetSize::Small,
  NetworkArchitecture<TransformedFeatureDimensionsSmall, L2Small, L3Small>,
  FeatureTransformer<TransformedFeatureDimensionsSmall, &StateInfo::accumulatorSmall>>;

}  // namespace Stockfish::Eval::NNUE