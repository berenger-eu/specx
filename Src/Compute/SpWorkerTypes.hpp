///////////////////////////////////////////////////////////////////////////
// Specx - Berenger Bramas MPCDF - 2017
// Under LGPL Licence, please you must read the LICENCE file.
///////////////////////////////////////////////////////////////////////////
#ifndef SPWORKERTYPES_HPP
#define SPWORKERTYPES_HPP

#include "Config/SpConfig.hpp"

namespace SpWorkerTypes{

enum class Type {
    CPU_WORKER,
#ifdef SPECX_COMPILE_WITH_CUDA
    CUDA_WORKER,
#endif
#ifdef SPECX_COMPILE_WITH_HIP
    HIP_WORKER,
#endif
    NB_WORKER_TYPES
};

}

#endif
