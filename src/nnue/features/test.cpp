#include <iostream>
#include <string>
#include <algorithm>


#define private public

#include "../../bitboard.h"
#include "../../position.h"
#include "../../types.h"
#include "../../uci.h"
#include "../network.h"
#include "../nnue_misc.h"

using namespace Stockfish;
//including this for now for debug purposes
int main(int argc, char* argv[]) {
    Bitboards::init();
    Position::init();
    
    UCIEngine engin(argc, argv);
    std::vector<std::string> moves;
    engin.engine.set_position("bnr3k1/4qppp/3b1n2/1p6/2pP4/4PN2/1BQNBPPP/5RK1 w - - 2 17", moves);
    engin.engine.networks->big.featureTransformer->update_accumulator_scratch<WHITE>(engin.engine.pos);
    engin.engine.states->emplace_back();
    engin.engine.pos.do_move(engin.to_move(engin.engine.pos, "d2c4"), engin.engine.states->back());
    engin.engine.networks->big.featureTransformer->update_accumulator<WHITE>(engin.engine.pos);
    //bigft.update_accumulator_scratch<WHITE>(pos);
    /*
    std::string fen1 = std::string(argv[1]);
    std::string fen2 = std::string(argv[2]);
    StateListPtr states;
    states = StateListPtr(new std::deque<StateInfo>(1));
    Eval::NNUE::Features::Simplified_Threats::IndexList white1;
    Eval::NNUE::Features::Simplified_Threats::IndexList white2;
    Eval::NNUE::Features::Simplified_Threats::IndexList black1;
    Eval::NNUE::Features::Simplified_Threats::IndexList black2;
    Eval::NNUE::Features::Simplified_Threats::IndexList white3;
    Eval::NNUE::Features::Simplified_Threats::IndexList black3;
    Eval::NNUE::Features::Simplified_Threats::IndexList white4;
    Eval::NNUE::Features::Simplified_Threats::IndexList black4;
    pos.set(fen1, false, &states->back());
    test.append_active_threats<WHITE>(pos, white1);
    test.append_active_threats<BLACK>(pos, black1);
    std::cout << "Position 1 " << white1.size() << " white perspective features:\n";
    for (auto feature : white1) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black1.size() << " black perspective features:\n";
    for (auto feature : black1) {
        std::cout << feature << ", ";
    }
    pos.set(fen2, false, &states->back());
    test.append_active_threats<WHITE>(pos, white2);
    test.append_active_threats<BLACK>(pos, black2);
    std::cout << "\nPosition 2 " << white2.size() << " white perspective features:\n";
    for (auto feature : white2) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black2.size() << " black perspective features:\n";
    for (auto feature : black2) {
        std::cout << feature << ", ";
    }
    Eval::NNUE::write_difference(white1, white2, white3, white4);
    Eval::NNUE::write_difference(black1, black2, black3, black4);
    std::cout << "\n" << white3.size() << " removed white perspective features:\n";
    for (auto feature : white3) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black3.size() << " removed black perspective features:\n";
    for (auto feature : black3) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << white4.size() << " added white perspective features:\n";
    for (auto feature : white4) {
        std::cout << feature << ", ";
    }
    std::cout << "\n" << black4.size() << " added black perspective features:\n";
    for (auto feature : black4) {
        std::cout << feature << ", ";
    }
    */
    return 0;
}


/*
clang++ -pthread -static -o test.exe test.cpp simplified_threats.cpp ../network.cpp ../nnue_misc.cpp ../../benchmark.cpp ../../bitboard.cpp 
*/