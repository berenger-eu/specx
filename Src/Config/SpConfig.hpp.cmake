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

/*#ifdef SPETABARU_COMPILE_WITH_CUDA

	#ifndef SPHOST
		#define SPHOST __host__
	#endif

	#ifndef SPDEVICE
		#define SPDEVICE __device__
	#endif

	#ifndef SPGLOBAL
		#define SPGLOBAL __GLOBAL__
	#endif


#else*/

		#ifndef SPHOST
			#define SPHOST
		#endif
		
		#ifndef SPDEVICE
			#define SPDEVICE
		#endif
		
		#ifndef SPGLOBAL
			#define SPGLOBAL
		#endif

//#endif

namespace SpConfig {
    #ifdef SPETABARU_COMPILE_WITH_CUDA
        inline constexpr bool CompileWithCuda = true;
        inline constexpr int SpMaxNbGpus = 16;
    #else
        inline constexpr bool CompileWithCuda = false;    
    #endif
}

#endif
