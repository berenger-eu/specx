###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(BENCHMARK_LG_SPECX CXX)

ADD_DEFINITIONS(${SPECX_CXX_FLAGS})

message(CHECK_START "check OpenMP support")
if (NOT APPLE)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

message(CHECK_PASS "found")

if($ENV{VERBOSE})
    MESSAGE(STATUS "Benchmark -- SPECX_CXX_FLAGS   : ${BENCHMARK_LG_SPECX_CXX_FLAGS}")
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
    endif()
    add_executable(
		    ${execname}
		    ${exec}
	    )

    target_link_libraries(
        ${execname}
        specx
        ${SPECX_LIBRARIES}
        )
endforeach(exec)
else()
 message(CHECK_FAIL "not supported on APPLE")
endif(NOT APPLE)
