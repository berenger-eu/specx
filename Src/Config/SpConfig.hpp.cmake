///////////////////////////////////////////////////////////////////////////
// SPETABARU - Berenger Bramas MPCDF - 2016
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPCONFIG_H
#define SPCONFIG_H

// Define all macros (ADD-NEW-HERE)
// #cmakedefine SPETABARU_USE_X
// @SPETABARU_X@

#cmakedefine SPETABARU_COMPILE_WITH_CUDA

namespace SpConfig {
    #ifdef SPETABARU_COMPILE_WITH_CUDA
        inline constexpr bool CompileWithCuda = true;
    #else
        inline constexpr bool CompileWithCuda = false;
    #endif
}

#endif
