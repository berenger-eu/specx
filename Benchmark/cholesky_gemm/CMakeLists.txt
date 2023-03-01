###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(BENCHMARK_CHOLESKY_GEMM_SPECX CXX)

ADD_DEFINITIONS(${SPECX_CXX_FLAGS})

if($ENV{VERBOSE})
    MESSAGE(STATUS "Benchmark CHOLESKY_GEMM -- SPECX_CXX_FLAGS   : ${BENCHMARK_CP_SPECX_CXX_FLAGS}")
endif()

find_package(BLAS)
find_package(LAPACK)

if(BLAS_FOUND AND LAPACK_FOUND)
	if($ENV{VERBOSE})
	    MESSAGE(STATUS "Benchmark CHOLESKY_GEMM -- BLAS_LIBRARIES   : ${BLAS_LIBRARIES}")
	    MESSAGE(STATUS "Benchmark CHOLESKY_GEMM -- BLAS_LINKER_FLAGS   : ${BLAS_LINKER_FLAGS}")
	    MESSAGE(STATUS "Benchmark CHOLESKY_GEMM -- LAPACK_LIBRARIES   : ${LAPACK_LIBRARIES}")
	    MESSAGE(STATUS "Benchmark CHOLESKY_GEMM -- LAPACK_LINKER_FLAGS   : ${LAPACK_LINKER_FLAGS}")
	endif()

	# Find all code files
	file(	
	    GLOB_RECURSE
	    source_tests_files
	    ./*.cpp
	)


	# Adding the project sources dir as an include dir
	INCLUDE_DIRECTORIES(
	     ${SPECX_BINARY_DIR}/Src    
	     ${SPECX_SOURCE_DIR}/Src    
	)

    LINK_DIRECTORIES(
    )


	# Add execs - 1 cpp = 1 exec
	foreach(exec ${source_tests_files}) 
	    get_filename_component(
		    execname ${exec}
		    NAME_WE
	    )

	    if($ENV{VERBOSE})
		    MESSAGE(STATUS "Benchmark -- Add ${execname}")
	    endif()

	    if(SPECX_COMPILE_WITH_CUDA)
		    set_source_files_properties(${exec} PROPERTIES LANGUAGE CUDA)
		    set(SPECX_LIBRARIES "${SPECX_LIBRARIES};cublas")
	    endif()
	    
	    add_executable(${execname} ${exec})
	    
	    # TODO set_target_properties(${execname} PROPERTIES COMPILE_FLAGS "-llapack -lblas")

	    target_link_libraries(
		    ${execname}
		    specx
		    ${SPECX_LIBRARIES}
		    ${BLAS_LIBRARIES}
		    ${LAPACK_LIBRARIES}
		)
	endforeach(exec)
	
else(BLAS_FOUND AND LAPACK_FOUND)
    MESSAGE(STATUS "Benchmark CHOLESKY_GEMM -- Disable -- BLAS_FOUND : ${BLAS_FOUND}")
    MESSAGE(STATUS "Benchmark CHOLESKY_GEMM -- Disable -- LAPACK_FOUND : ${LAPACK_FOUND}")
endif()
