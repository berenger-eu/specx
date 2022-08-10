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
// This file is a trimmed version of src/util/util.h file from the complexes++
// project. 
// https://github.com/bio-phys/complexespp
//
// [12 Aug 2020] It only includes the DEBUG_ASSERT macro which has been renamed to
// SP_DEBUG_ASSERT.
///////////////////////////////////////////////////////////////////////////


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

#define SP_DEBUG_ASSERT(condition, ...)                                           \
  if (!(condition)) {                                                          \
    fmt::print(std::cerr, "An assert has failed : {} \n", #condition);         \
    fmt::print(std::cerr, "\t In file : {}\n", __FILE__);                      \
    fmt::print(std::cerr, "\t At line : {}\n", __LINE__);                      \
    fmt::print(std::cerr, "\t Log : ");                                        \
    fmt::print(std::cerr, __VA_ARGS__);                                        \
    fmt::print(std::cerr, "\n");                                               \
    throw std::runtime_error("Bad Assert Exit");                               \
  }
#endif

////////////////////////////////////////////////////

#endif
