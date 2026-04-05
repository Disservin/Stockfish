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

#include "datagen.h"

#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <optional>
#include <random>
#include <csignal>
#include <sstream>
#include <string>
#include <vector>

#include "engine.h"
#include "misc.h"
#include "movegen.h"
#include "position.h"
#include "score.h"
#include "search.h"
#include "sfbinpack.h"
#include "types.h"
#include "uci.h"
#include "ucioption.h"

namespace Stockfish {

namespace {

constexpr int    MaxGamePlies           = 512;
constexpr int    OpeningMultiPVLines    = 4;
constexpr int    AdjudicationAbsScore   = 2000;
constexpr int    DrawAdjudicationScore  = 10;
constexpr int    AdjudicationMoveCount  = 4;
constexpr auto   HardSearchNodeLimit    = uint64_t(40000);
constexpr auto   OpeningSearchNodeLimit = uint64_t(500);
constexpr auto   GameSearchNodeLimit    = uint64_t(5000);
constexpr size_t RandomPlies[]          = {9, 10};

volatile std::sig_atomic_t DatagenStopRequested = 0;

void on_datagen_sigint(int) { DatagenStopRequested = 1; }

class ScopedSigintHandler {
   public:
    ScopedSigintHandler() { previous = std::signal(SIGINT, on_datagen_sigint); }

    ~ScopedSigintHandler() { std::signal(SIGINT, previous); }

   private:
    using Handler = void (*)(int);
    Handler previous;
};

struct DatagenPosition {
    std::string fen;
    std::string move;
    int         score;
    uint16_t    ply;
};

struct DatagenArgs {
    uint64_t    games = 0;
    uint64_t    hash  = 0;
    uint64_t    seed  = 0;
    std::string output;
};

struct DatagenGame {
    std::string                  startFen;
    std::vector<std::string>     moves;
    std::vector<DatagenPosition> entries;
    int                          finalResult = 0;
    uint16_t                     finalPly    = 0;
};

bool parse_datagen_args(std::istream& is, DatagenArgs& args, std::string& error) {
    std::string token;
    bool        haveGames  = false;
    bool        haveHash   = false;
    bool        haveSeed   = false;
    bool        haveOutput = false;

    while (is >> token)
    {
        if (token == "games")
        {
            if (!(is >> args.games))
            {
                error = "Missing value after 'games'.";
                return false;
            }
            haveGames = true;
        }
        else if (token == "hash")
        {
            if (!(is >> args.hash))
            {
                error = "Missing value after 'hash'.";
                return false;
            }
            haveHash = true;
        }
        else if (token == "seed")
        {
            if (!(is >> args.seed))
            {
                error = "Missing value after 'seed'.";
                return false;
            }
            haveSeed = true;
        }
        else if (token == "output")
        {
            if (!(is >> args.output))
            {
                error = "Missing value after 'output'.";
                return false;
            }
            haveOutput = true;
        }
        else
        {
            error = "Unknown datagen argument: " + token;
            return false;
        }
    }

    if (!haveGames || !haveHash || !haveSeed || !haveOutput)
    {
        error = "Usage: datagen games <count> hash <mb> seed <seed> output <data.binpack>";
        return false;
    }

    if (!args.games)
    {
        error = "games must be greater than 0.";
        return false;
    }

    if (!args.hash)
    {
        error = "hash must be greater than 0.";
        return false;
    }

    if (args.hash > uint64_t(std::numeric_limits<int>::max()))
    {
        error = "hash is too large.";
        return false;
    }

    return true;
}

Search::LimitsType nodes_limit() {
    Search::LimitsType limits;
    limits.startTime = now();
    return limits;
}

std::optional<Move> choose_datagen_move(Engine& engine, const Position& pos, std::mt19937_64& rng) {
    const auto legalMoves = MoveList<LEGAL>(pos);
    if (legalMoves.size() == 0)
        return std::nullopt;

    auto analysis = engine.analyze(
      nodes_limit(),
      Engine::AnalysisConfig{OpeningMultiPVLines, OpeningSearchNodeLimit, HardSearchNodeLimit});

    std::vector<Move> candidates;
    candidates.reserve(analysis.rootMoves.size());

    for (const auto& rootMove : analysis.rootMoves)
        if (!is_decisive(rootMove.score) && !rootMove.pv.empty())
            candidates.push_back(rootMove.pv[0]);

    if (candidates.empty())
        return std::nullopt;

    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng)];
}

bool generate_random_opening(Engine&                   engine,
                             std::mt19937_64&          rng,
                             std::string&              startFen,
                             std::vector<std::string>& moves,
                             std::string&              error) {
    startFen = StartFEN;

    auto err = engine.set_position(startFen, {});
    if (err.has_value())
    {
        error = "Invalid book FEN: " + startFen + ". " + std::string(err->what());
        return false;
    }

    std::uniform_int_distribution<size_t> pliesDist(0, std::size(RandomPlies) - 1);
    const size_t                          plies      = RandomPlies[pliesDist(rng)];
    const bool                            isChess960 = engine.get_options()["UCI_Chess960"];

    std::deque<StateInfo> states(1);
    Position              pos;
    err = pos.set(startFen, isChess960, &states.back());
    if (err.has_value())
    {
        error = "Invalid book FEN: " + startFen + ". " + std::string(err->what());
        return false;
    }

    moves.clear();
    moves.reserve(plies);

    for (size_t i = 0; i < plies; ++i)
    {
        auto move = choose_datagen_move(engine, pos, rng);
        if (!move.has_value())
            return false;

        const auto uciMove = UCIEngine::move(*move, pos.is_chess960());
        states.emplace_back();
        pos.do_move(*move, states.back(), nullptr);
        moves.push_back(uciMove);

        err = engine.set_position(startFen, moves);
        if (err.has_value())
        {
            error = "Failed to apply opening moves: " + std::string(err->what());
            return false;
        }
    }

    return MoveList<LEGAL>(pos).size() != 0;
}

bool is_terminal_position(const Position& pos) {
    return pos.is_draw(0) || MoveList<LEGAL>(pos).size() == 0;
}

int terminal_result(const Position& pos) {
    if (pos.is_draw(0))
        return 0;

    if (MoveList<LEGAL>(pos).size() != 0)
        return 0;

    return pos.checkers() ? -1 : 0;
}

bool play_datagen_game(Engine&                  engine,
                       const std::string&       startFen,
                       std::vector<std::string> moves,
                       DatagenGame&             game,
                       std::string&             error) {
    game.startFen = startFen;
    game.moves    = std::move(moves);
    game.entries.clear();
    game.finalResult             = 0;
    game.finalPly                = 0;
    const bool isChess960        = engine.get_options()["UCI_Chess960"];
    int        adjudicationCount = 0;
    int        drawCount         = 0;

    std::deque<StateInfo> states(1);
    Position              pos;
    auto                  err = pos.set(startFen, isChess960, &states.back());
    if (err.has_value())
    {
        error = "Failed to initialize self-play position: " + std::string(err->what());
        return false;
    }

    for (const auto& move : game.moves)
    {
        const auto parsed = UCIEngine::to_move(pos, move);
        if (parsed == Move::none())
        {
            error = "Illegal opening move while building self-play history: " + move;
            return false;
        }

        states.emplace_back();
        pos.do_move(parsed, states.back(), nullptr);
    }

    for (int plyCount = 0; plyCount < MaxGamePlies; ++plyCount)
    {
        if (is_terminal_position(pos))
            break;

        err = engine.set_position(game.startFen, game.moves);
        if (err.has_value())
        {
            error = "Failed to sync engine for self-play: " + std::string(err->what());
            return false;
        }

        auto analysis = engine.analyze(
          nodes_limit(), Engine::AnalysisConfig{1, GameSearchNodeLimit, HardSearchNodeLimit});
        if (analysis.rootMoves.empty() || analysis.rootMoves[0].pv.empty())
            break;

        const Move  bestMove = analysis.rootMoves[0].pv[0];
        const Value rawScore = analysis.rootMoves[0].score;
        const int   entryScore =
          (pos.capture_stage(bestMove) || pos.checkers()) ? VALUE_NONE : rawScore;

        game.entries.push_back({
          pos.fen(),
          UCIEngine::move(bestMove, pos.is_chess960()),
          entryScore,
          static_cast<uint16_t>(pos.game_ply()),
        });

        adjudicationCount = std::abs(rawScore) > AdjudicationAbsScore ? adjudicationCount + 1 : 0;
        drawCount         = std::abs(rawScore) < DrawAdjudicationScore ? drawCount + 1 : 0;

        if (drawCount >= AdjudicationMoveCount)
        {
            game.finalResult = 0;
            game.finalPly    = static_cast<uint16_t>(pos.game_ply());
            return true;
        }

        if (adjudicationCount > AdjudicationMoveCount)
        {
            game.finalResult = rawScore > 0 ? 1 : -1;
            game.finalPly    = static_cast<uint16_t>(pos.game_ply());
            return true;
        }

        states.emplace_back();
        pos.do_move(bestMove, states.back(), nullptr);
        game.moves.push_back(UCIEngine::move(bestMove, pos.is_chess960()));
    }

    if (!is_terminal_position(pos))
    {
        game.entries.clear();
        return true;
    }

    game.finalResult = terminal_result(pos);
    game.finalPly    = static_cast<uint16_t>(pos.game_ply());
    return true;
}

void write_datagen_game(sfbinpack_writer_handle*            writer,
                        const std::vector<DatagenPosition>& entries,
                        int                                 finalResult,
                        uint16_t                            finalPly,
                        std::string&                        error) {
    for (const auto& entry : entries)
    {
        const int sideResult = ((finalPly - entry.ply) % 2 == 0) ? finalResult : -finalResult;

        SfbinpackEntry ffiEntry{
          entry.fen.c_str(),
          entry.move.c_str(),
          static_cast<short>(entry.score),
          entry.ply,
          static_cast<short>(sideResult),
        };

        const auto status = sfbinpack_writer_write_entry(writer, &ffiEntry);
        if (status != SFBINPACK_STATUS_OK)
        {
            const char* lastError = sfbinpack_last_error_message();
            error =
              std::string("binpack write failed: ") + (lastError ? lastError : "unknown error");
            return;
        }
    }
}

uint64_t output_size(const std::string& path) {
    std::error_code ec;
    const auto      size = std::filesystem::file_size(path, ec);
    return ec ? 0 : size;
}

std::string human_bytes(uint64_t bytes) {
    static constexpr const char* Units[] = {"B", "KiB", "MiB", "GiB", "TiB"};

    double value = bytes;
    int    unit  = 0;
    while (value >= 1024.0 && unit < 4)
    {
        value /= 1024.0;
        ++unit;
    }

    std::ostringstream os;
    os << std::fixed << std::setprecision(unit == 0 ? 0 : 1) << value << ' ' << Units[unit];
    return os.str();
}

void print_progress_line(uint64_t           generated,
                         uint64_t           totalGames,
                         uint64_t           positions,
                         TimePoint          elapsed,
                         const std::string& outputPath) {
    const uint64_t posPerSec   = 1000 * positions / elapsed;
    const uint64_t bytes       = output_size(outputPath);
    const uint64_t bytesPerMin = 60000 * bytes / elapsed;

    sync_cout_start();
    std::cout << "\rinfo string datagen games " << generated << "/" << totalGames << " positions "
              << positions << " speed " << posPerSec << " pos/s write " << human_bytes(bytesPerMin)
              << "/min" << std::string(16, ' ') << std::flush;
    sync_cout_end();
}

}  // namespace

void run_datagen(Engine& engine, std::istream& args) {
    DatagenArgs datagenArgs;
    std::string error;

    if (!parse_datagen_args(args, datagenArgs, error))
    {
        sync_cout << "info string datagen error: " << error << sync_endl;
        return;
    }

    std::istringstream hashOption("name Hash value " + std::to_string(datagenArgs.hash));
    engine.get_options().setoption(hashOption);

    DatagenStopRequested = 0;
    ScopedSigintHandler sigintHandler;

    std::mt19937_64          rng(datagenArgs.seed);
    sfbinpack_writer_handle* writer = sfbinpack_writer_new(datagenArgs.output.c_str());
    const TimePoint          start  = now();

    if (!writer)
    {
        const char* lastError = sfbinpack_last_error_message();
        sync_cout << "info string datagen error: Failed to create binpack writer: "
                  << (lastError ? lastError : "unknown error") << sync_endl;
        return;
    }

    uint64_t generated = 0;
    uint64_t positions = 0;
    while (generated < datagenArgs.games && !DatagenStopRequested)
    {
        std::string              startFen;
        std::vector<std::string> openingMoves;
        DatagenGame              game;

        error.clear();
        if (!generate_random_opening(engine, rng, startFen, openingMoves, error))
        {
            if (!error.empty())
            {
                sync_cout << "info string datagen error: " << error << sync_endl;
                sfbinpack_writer_free(writer);
                return;
            }

            continue;
        }

        error.clear();
        if (!play_datagen_game(engine, startFen, openingMoves, game, error))
        {
            sync_cout << "info string datagen error: " << error << sync_endl;
            sfbinpack_writer_free(writer);
            return;
        }

        if (game.entries.empty())
            continue;

        error.clear();
        write_datagen_game(writer, game.entries, game.finalResult, game.finalPly, error);
        if (!error.empty())
        {
            sync_cout << "info string datagen error: " << error << sync_endl;
            sfbinpack_writer_free(writer);
            return;
        }

        ++generated;
        positions += game.entries.size();

        const TimePoint elapsed = std::max<TimePoint>(1, now() - start);
        print_progress_line(generated, datagenArgs.games, positions, elapsed, datagenArgs.output);
    }

    sync_cout << "" << sync_endl;

    if (DatagenStopRequested)
        sync_cout << "info string datagen interrupt received, finalizing output" << sync_endl;

    const auto finishStatus = sfbinpack_writer_finish(writer);
    if (finishStatus != SFBINPACK_STATUS_OK)
    {
        const char* lastError = sfbinpack_last_error_message();
        sync_cout << "info string datagen error: Failed to finalize binpack writer: "
                  << (lastError ? lastError : "unknown error") << sync_endl;
        sfbinpack_writer_free(writer);
        return;
    }

    sfbinpack_writer_free(writer);
}

}  // namespace Stockfish
