###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(BENCHMARK_CP_SPECX_OMP CXX)

ADD_DEFINITIONS(${SPECX_CXX_FLAGS})

if($ENV{VERBOSE})
    MESSAGE(STATUS "Benchmark -- SPECX_CXX_FLAGS   : ${BENCHMARK_CP_SPECX_CXX_FLAGS}")
endif()


find_package(OpenMP)

if(OpenMP_FOUND)
    list(APPEND BENCHMARK_CP_SPECX_OMP_LIBRARIES OpenMP::OpenMP_CXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

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


    # Add execs - 1 cpp = 1 exec
    foreach(exec ${source_tests_files})
        if(${SPECX_COMPILE_WITH_MPI} OR NOT ${exec} MATCHES "-mpi\.")
            get_filename_component(
	            execname ${exec}
	            NAME_WE
            )

            if($ENV{VERBOSE})
                MESSAGE(STATUS "Benchmark -- Add ${execname}")
            endif()

            if(SPECX_COMPILE_WITH_CUDA)
                set_source_files_properties(${exec} PROPERTIES LANGUAGE CUDA)
            endif()
            add_executable(
		            ${execname}
		            ${exec}
	            )

            target_link_libraries(
                ${execname}
                specx
                ${SPECX_LIBRARIES}
                ${BENCHMARK_CP_SPECX_OMP_LIBRARIES}
                )
        endif()
    endforeach(exec)
else(OpenMP_FOUND)
    MESSAGE(WARNING "Openmp not found, disable particle simulation")
endif(OpenMP_FOUND)
