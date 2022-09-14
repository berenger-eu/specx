###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(BENCHMARK_SPECX CXX)

OPTION( SPECX_BUILD_BENCHMARK  "Set to ON to build BENCHMARK" ON )

if($ENV{VERBOSE})
    MESSAGE(STATUS "Benchmark -- SPECX_BUILD_BENCHMARK = ${SPECX_BUILD_BENCHMARK}")
endif()

if( SPECX_BUILD_BENCHMARK )
    INCLUDE_DIRECTORIES(
         ${CMAKE_CURRENT_SOURCE_DIR}
         ${CMAKE_CURRENT_BINARY_DIR}
    )

    file(GLOB all_benchmark ./*)

    foreach(benchmark_name ${all_benchmark})
        if(IS_DIRECTORY ${benchmark_name})
          if($ENV{VERBOSE})
              message(STATUS "Benchmark -- ${benchmark_name}" )
          endif()
          add_subdirectory(${benchmark_name})
        endif()
    endforeach()
endif()
