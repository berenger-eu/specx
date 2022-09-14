###########################################################################
# SPECX - Berenger Bramas MPCDF - 2017
# Under MIT Licence, please you must read the LICENCE file.
###########################################################################
cmake_minimum_required (VERSION 3.20 FATAL_ERROR)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

#===========================================================================
# Project Declaration
#===========================================================================
project(SPECX CXX)

include(CheckLanguage)
check_language(CUDA)

OPTION(SPECX_COMPILE_WITH_CUDA "Set to on to compile tasks with CUDA callables" ${CMAKE_CUDA_COMPILER})
OPTION(SPECX_COMPILE_WITH_HIP "Set to on to compile tasks with HIP callables" ${CMAKE_HIP_COMPILER})

if(SPECX_COMPILE_WITH_CUDA AND SPECX_COMPILE_WITH_HIP)
    message(FATAL_ERROR "Impossible to use both CUDA and HIP together")
endif()


if(SPECX_COMPILE_WITH_CUDA)
	enable_language(CUDA)
		
	if(NOT CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING
          "Set CUDA target architecture" FORCE)
    endif()
endif()

if(SPECX_COMPILE_WITH_HIP)
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    # Find hip
    find_package(hip)
    # Set the GPU to compile for
    set(GPU_TARGETS "gfx900" CACHE STRING "GPU targets to compile for")
endif()

# check if compiling into source directories
string(COMPARE EQUAL "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" insource)
if(insource)
  message(FATAL_ERROR "${PROJECT_NAME} requires an out of source build.\
                      Remove CMakeCache.txt (alternatively you can remove all untracked files with\
                      git clean -i, git clean -d or git clean -df). Then, create a subirectory outside\
                      the specx source tree, cd into the directory you just created and then type\
                      cmake <path to specx repository>.\
                      Alternatively you can create a build directory at the root of the specx\
                      source tree but then it is recommended to name it 'build' as it is the only\
                      build directory name that is explicitly marked as to be ignored by Git.")
endif(insource)

# Adds the CMAKE_DEPENDENT_OPTION command
INCLUDE(CMakeDependentOption)
set(SPECX_LIBRARIES "")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


if(SPECX_COMPILE_WITH_CUDA)
    # TODO https://stackoverflow.com/questions/68950463/cmake-cuda-compiler-flag-is-false-despite-cuda-being-found
	#set(CMAKE_CUDA_FLAGS "-std=c++17")
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	SET(CMAKE_CUDA_EXTENSIONS OFF)
	include(FindCUDA)
	include_directories(${CUDA_INCLUDE_DIRS})
    list(APPEND SPECX_LIBRARIES ${CUDA_LIBRARIES})
endif()

# if(SPECX_COMPILE_WITH_HIP)
#     # https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html
#     list(APPEND SPECX_LIBRARIES hip::device hip::host)    
# endif()

OPTION(SPECX_COMPILE_WITH_MPI "Set to on to use MPI" OFF)
if(SPECX_COMPILE_WITH_MPI)
    find_package(MPI REQUIRED)
    include_directories(SYSTEM ${MPI_INCLUDE_PATH})
    list(APPEND SPECX_LIBRARIES ${MPI_CXX_LIBRARIES})
endif()

# need pthread
list(APPEND SPECX_LIBRARIES pthread)

# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo" FORCE)
endif()

if($ENV{VERBOSE})
    MESSAGE(STATUS "Main -- CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")
endif()

# Test if SPECX is included as a subproject
if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(SPECX_AS_SUBPROJECT FALSE)
else()
    set(SPECX_AS_SUBPROJECT TRUE)
endif()

if($ENV{VERBOSE})
    MESSAGE(STATUS "Main -- SPECX_AS_SUBPROJECT ${SPECX_AS_SUBPROJECT}")
endif()

# Build only the lib if subproject
if(NOT DEFINED SPECX_JUST_LIB)
    set(SPECX_JUST_LIB ${SPECX_AS_SUBPROJECT} CACHE BOOL
        "Set to on to build the lib for external use (without test and extra flags)" FORCE)
endif()

if(NOT DEFINED SPECX_DO_INSTALL)
    set(SPECX_DO_INSTALL ON CACHE BOOL
        "Set to on to install the headers file of SPECX" FORCE)
endif()

if($ENV{VERBOSE})
    MESSAGE(STATUS "Main -- SPECX_JUST_LIB ${SPECX_JUST_LIB}")
    MESSAGE(STATUS "Main -- SPECX_DO_INSTALL ${SPECX_DO_INSTALL}")
endif()

#===========================================================================
# Version Number
#===========================================================================

set(SPECX_MAJOR_VERSION 0)
set(SPECX_MINOR_VERSION 1)
set(SPECX_PATCH_VERSION 0)
set(SPECX_VERSION  "${SPECX_MAJOR_VERSION}.${SPECX_MINOR_VERSION}.${SPECX_PATCH_VERSION}" )

#===========================================================================
# Options
#===========================================================================

option(SPECX_USE_COVERAGE "Should be ON to enable code coverage" OFF)

if( SPECX_USE_COVERAGE )
    message(STATUS "Main -- Use code coverage")
    set(SPECX_CXX_FLAGS "${SPECX_CXX_FLAGS} -g -O0 --coverage -ftest-coverage") # -fprofile-arcs
    list(APPEND SPECX_LIBRARIES gcov)
    # Then one can use these commands:
    # lcov --directory CMakeFiles/ -c -o coverage-rapport.info
    # genhtml -o ./coverage -t "Coverage specx" coverage-rapport.info
endif()

# Extra-flags
SET( SPECX_EXTRA_CXX_FLAGS "-funroll-loops" CACHE STRING "These flags are used for all configurations"  )
set(SPECX_CXX_FLAGS  "-std=c++17 ${SPECX_CXX_FLAGS} ${SPECX_EXTRA_CXX_FLAGS}")

# Options
if(NOT SPECX_AS_SUBPROJECT)
    SET( SPECX_EXTRA_CXX_FLAGS_RELEASE "-O3" CACHE STRING "These flags are added to the usual ones for RELEASE"  )
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} ${SPECX_EXTRA_CXX_FLAGS_RELEASE}")

    SET( SPECX_EXTRA_CXX_FLAGS_DEBUG "-O0" CACHE STRING "These flags are added to the usual ones for DEBUG"  )
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} ${SPECX_EXTRA_CXX_FLAGS_DEBUG}")

    set(SPECX_CXX_FLAGS "${SPECX_CXX_FLAGS} -fpic -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wall -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wno-error")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    else()
        set(SPECX_CXX_FLAGS "${SPECX_CXX_FLAGS} -Wconversion -Wno-sign-conversion")
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
        # Builting functions are not correctly making variables/parameters used
        set(SPECX_CXX_FLAGS "${SPECX_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-unused-but-set-parameter")
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(SPECX_CXX_FLAGS  "${SPECX_CXX_FLAGS} -m64")
    endif()

    OPTION( SPECX_ATTACHE_SOURCE "Set to on to add -g flag" OFF )
    if( SPECX_ATTACHE_SOURCE )
        set(SPECX_CXX_FLAGS  "${SPECX_CXX_FLAGS} -g")
    endif(SPECX_ATTACHE_SOURCE)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "icpc.*$")
else()
  set(SPECX_CXX_FLAGS "${SPECX_CXX_FLAGS} -Wold-style-cast")
endif()

#===========================================================================
# Generate config
#===========================================================================
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/Src/Config/SpConfig.hpp.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/Src/Config/SpConfig.hpp  )

#===========================================================================
# Build lib
#===========================================================================
add_subdirectory(Src)


if(NOT SPECX_JUST_LIB)
    option(USE_ADVANCE_TESTING "Should be ON to enable more tests" OFF)

    #===========================================================================
    # Build unit tests
    #===========================================================================
    enable_testing()
    add_subdirectory(UTests)
    add_subdirectory(UTests/ShouldNotCompile)
    
    if(SPECX_COMPILE_WITH_CUDA)
		add_subdirectory(UTests/Cuda)
	endif()
	
    if(SPECX_COMPILE_WITH_HIP)
		add_subdirectory(UTests/Hip)
	endif()
	
    if(SPECX_COMPILE_WITH_MPI)
		add_subdirectory(UTests/MPI)
	endif()
	
    #===========================================================================
    # Build examples
    #===========================================================================
    add_subdirectory(Examples)
    #===========================================================================
    # Build benchmark
    #===========================================================================
    add_subdirectory(Benchmark)
endif()

#===========================================================================
# Export cmake info
#===========================================================================
configure_file(${SPECX_SOURCE_DIR}/CMakeModules/SpecxInfo.cmake.in
    ${SPECX_BINARY_DIR}/SpecxInfo.cmake
    @ONLY IMMEDIATE
)
if(SPECX_DO_INSTALL)
    install(FILES  ${SPECX_BINARY_DIR}/SpecxInfo.cmake  DESTINATION lib/  )
endif()

if(SPECX_AS_SUBPROJECT)
    set(SPECX_CXX_FLAGS  "${SPECX_CXX_FLAGS}" PARENT_SCOPE)
    set(SPECX_BINARY_DIR  "${SPECX_BINARY_DIR}" PARENT_SCOPE)
    set(SPECX_SOURCE_DIR  "${SPECX_SOURCE_DIR}" PARENT_SCOPE)
    set(SPECX_INCLUDE_DIR  "${SPECX_INCLUDE_DIR}" PARENT_SCOPE)
    set(SPECX_LIBRARIES  "${SPECX_LIBRARIES}" PARENT_SCOPE)
else()
    # nothing - we are not used by another lib here
endif()



