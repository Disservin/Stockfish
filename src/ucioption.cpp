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

#include "ucioption.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <iostream>
#include <sstream>
#include <utility>

#include "misc.h"

namespace Stockfish {

template<typename... Ts>
struct overload: Ts... {
    using Ts::operator()...;
};

template<typename... Ts>
overload(Ts...) -> overload<Ts...>;

size_t Option::insert_order = 0;

bool CaseInsensitiveLess::operator()(const std::string& s1, const std::string& s2) const {

    return std::lexicographical_compare(
      s1.begin(), s1.end(), s2.begin(), s2.end(),
      [](char c1, char c2) { return std::tolower(c1) < std::tolower(c2); });
}

void OptionsMap::add_info_listener(InfoListener&& message_func) { info = std::move(message_func); }

void OptionsMap::setoption(std::istringstream& is) {
    std::string token, name, value;

    is >> token;  // Consume the "name" token

    // Read the option name (can contain spaces)
    while (is >> token && token != "value")
        name += (name.empty() ? "" : " ") + token;

    // Read the option value (can contain spaces)
    while (is >> token)
        value += (value.empty() ? "" : " ") + token;

    if (options_map.count(name))
        options_map[name] = value;
    else
        sync_cout << "No such option: " << name << sync_endl;
}

Option OptionsMap::operator[](const std::string& name) const {
    auto it = options_map.find(name);
    return it != options_map.end() ? it->second : Option(this);
}

Option& OptionsMap::operator[](const std::string& name) {
    if (!options_map.count(name))
        options_map[name] = Option(this);
    return options_map[name];
}

std::size_t OptionsMap::count(const std::string& name) const { return options_map.count(name); }

Option::Option(OnChange f) :
    on_change(std::move(f)) {}

Option::Option(const OptionsMap* map) :
    parent(map) {}

Option::Option(const OptionValue& v, OnChange f) :
    value(v),
    on_change(std::move(f)) {}

Option::operator int() const {

    if (const auto* check = std::get_if<CheckOption>(&value))
        return check->value;

    if (const auto* spin = std::get_if<SpinOption>(&value))
        return spin->value;

    assert(false);

    return 0;
}

Option::operator std::string() const {

    if (const auto* str = std::get_if<StringOption>(&value))
        return str->value;

    if (const auto* combo = std::get_if<ComboOption>(&value))
        return combo->value;

    assert(false);

    return "";
}

bool Option::operator==(const char* s) const {

    const auto* combo = std::get_if<ComboOption>(&value);
    assert(combo != nullptr);

    return !CaseInsensitiveLess()(combo->value, s) && !CaseInsensitiveLess()(s, combo->value);
}

bool Option::operator!=(const char* s) const { return !(*this == s); }

std::string Option::get_type_string() const {
    const auto pattern = overload{
      [](const CheckOption&) -> std::string { return "check"; },
      [](const SpinOption&) -> std::string { return "spin"; },
      [](const ComboOption&) -> std::string { return "combo"; },
      [](const StringOption&) -> std::string { return "string"; },
      [](const ButtonOption&) -> std::string { return "button"; }  //
    };

    return std::visit(pattern, value);
}


// Inits options and assigns idx in the correct printing order

void Option::operator<<(const Option& o) {

    auto p = this->parent;
    *this  = o;

    this->parent = p;
    idx          = insert_order++;
}

void Option::operator<<(const OptionValue& o) {

    auto p      = this->parent;
    this->value = o;

    this->parent = p;
    idx          = insert_order++;
}

// Updates currentValue and triggers on_change() action. It's up to
// the GUI to check for option's limits, but we could receive the new value
// from the user by console window, so let's check the bounds anyway.
Option& Option::operator=(const std::string& v) {

    const auto pattern = overload{
      [](ButtonOption&) {
          // do nothing
      },
      [&v](CheckOption& opt) {
          if (v != "true" && v != "false")
              return;

          opt.value = (v == "true");
      },
      [&v](SpinOption& opt) {
          int new_value = std::stoi(v);

          if (new_value < opt.min || new_value > opt.max)
              return;

          opt.value = new_value;
      },
      [&v](ComboOption& opt) {
          OptionsMap         comboMap;  // To have case insensitive compare
          std::string        token;
          std::istringstream ss(opt.defaultValue);

          while (ss >> token)
              comboMap[token] << Option();
          if (!comboMap.count(v) || v == "var")
              return;

          opt.value = v;
      },
      [&v](StringOption& opt) { opt.value = v == "<empty>" ? "" : v; }  //
    };

    std::visit(pattern, value);

    if (on_change)
    {
        const auto ret = on_change(*this);
        if (ret && parent != nullptr && parent->info != nullptr)
            parent->info(ret);
    }

    return *this;
}

std::ostream& operator<<(std::ostream& os, const OptionsMap& om) {

    for (size_t idx = 0; idx < om.options_map.size(); ++idx)
    {
        for (const auto& [name, option] : om.options_map)
        {
            if (option.idx != idx)
                continue;

            os << "\noption name " << name << " type " << option.get_type_string();

            const auto pattern = overload{
              [&os](const CheckOption& opt) {
                  os << " default " << (opt.value ? "true" : "false");
              },
              [&os](const SpinOption& opt) {
                  os << " default " << opt.value << " min " << opt.min << " max " << opt.max;
              },
              [&os](const ComboOption& opt) {  //
                  os << " default " << opt.value;
              },
              [&os](const StringOption& opt) {
                  os << " default " << (opt.value.empty() ? "<empty>" : opt.value);
              },
              [](const ButtonOption&) {
                  // do nothing
              }  //
            };

            std::visit(pattern, option.value);

            break;
        }
    }

    return os;
}
}
