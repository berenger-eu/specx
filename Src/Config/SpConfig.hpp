///////////////////////////////////////////////////////////////////////////
// SPECX - Berenger Bramas MPCDF - 2016
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPCONFIG_H
#define SPCONFIG_H

// Define all macros (ADD-NEW-HERE)
/* #undef SPECX_USE_X */
// 

#define SPECX_USE_DEBUG_PRINT
/* #undef SPECX_COMPILE_WITH_CUDA */
/* #undef SPECX_COMPILE_WITH_MPI */
/* #undef SPECX_COMPILE_WITH_HIP */

namespace SpConfig {
    #ifdef SPECX_COMPILE_WITH_CUDA
        inline constexpr bool CompileWithCuda = true;
        inline constexpr int SpMaxNbCudas = 16;
    #else
        inline constexpr bool CompileWithCuda = false;    
    #endif
    #ifdef SPECX_COMPILE_WITH_MPI
        inline constexpr bool CompileWithMPI = true;
    #else
        inline constexpr bool CompileWithMPI = false;    
    #endif
    #ifdef SPECX_COMPILE_WITH_HIP
        inline constexpr bool CompileWithHip = true;
        inline constexpr int SpMaxNbHips = 16;
    #else
        inline constexpr bool CompileWithHip = false;
    #endif
}

#endif
