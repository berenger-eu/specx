// Copyright (c) 2018 the complexes++ development team and contributors
// (see the file AUTHORS for the full list of names)
//
// This file is part of complexes++.
//
// complexes++ is free software: you can redistribute it and/or modify
// it under the terms of the Lesser GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// complexes++ is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with complexes++.  If not, see <https://www.gnu.org/licenses/>

///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////

#ifndef SPDEBUGASSERT_HPP
#define SPDEBUGASSERT_HPP


////////////////////////////////////////////////////

#ifdef NDEBUG
#define SP_DEBUG_ASSERT(condition, ...) ((void)0)
#else

#include <iostream>

template<typename StreamClass, typename ... Params>
inline void SP_DEBUG_ASSERT_HELPER([[maybe_unused]] StreamClass& stream,
                                   [[maybe_unused]] Params&&... params){
    if(sizeof... (Params)){
        using expander = int[];
        (void)expander{0, (void(stream << ',' << std::forward<Params>(params)), 0)...};
    }
}

#define SP_DEBUG_ASSERT(condition, ...)                                           \
  if (!(condition)) {                                                          \
    std::cerr << "An assert has failed : " << #condition << "\n";         \
    std::cerr << "\t In file : " << __FILE__ << "\n";                      \
    std::cerr << "\t At line : " << __LINE__ << "\n";                      \
    std::cerr << "\t Log : ";                                        \
    SP_DEBUG_ASSERT_HELPER(std::cerr, __VA_ARGS__);                                        \
    std::cerr << std::endl;                                               \
    throw std::runtime_error("Bad Assert Exit");                               \
  }
#endif

////////////////////////////////////////////////////

#endif
