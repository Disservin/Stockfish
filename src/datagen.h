#ifndef DATAGEN_H_INCLUDED
#define DATAGEN_H_INCLUDED

#include <iosfwd>

namespace Stockfish {

class Engine;

void run_datagen(Engine& engine, std::istream& args);

}  // namespace Stockfish

#endif  // #ifndef DATAGEN_H_INCLUDED
