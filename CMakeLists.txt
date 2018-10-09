###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2017
# Under MIT Licence, please you must read the LICENCE file.
###########################################################################
cmake_minimum_required (VERSION 2.8.3 FATAL_ERROR)

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

#===========================================================================
# Project Declaration
#===========================================================================
project(SPETABARU CXX)

# check if compiling into source directories
string(COMPARE EQUAL "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}" insource)
if(insource)
  message(FATAL_ERROR "${PROJECT_NAME} requires an out of source build. Goto ./Build and tapes cmake ../")
endif(insource)

# Adds the CMAKE_DEPENDENT_OPTION command
INCLUDE(CMakeDependentOption)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# need pthread
set(SPETABARU_LIBRARIES pthread)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE) 
# the RPATH to be used when installing
SET(CMAKE_INSTALL_RPATH "/home/bbramas/Downloads/gcc7/lib64/")
# don't add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# Release by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo" FORCE)
endif()

if($ENV{VERBOSE})
    MESSAGE(STATUS "Main -- CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}")
endif()

# Test if SPETABARU is included as a subproject
if(${CMAKE_PROJECT_NAME} STREQUAL ${PROJECT_NAME})
    set(SPETABARU_AS_SUBPROJECT FALSE)
else()
    set(SPETABARU_AS_SUBPROJECT TRUE)
endif()

if($ENV{VERBOSE})
    MESSAGE(STATUS "Main -- SPETABARU_AS_SUBPROJECT ${SPETABARU_AS_SUBPROJECT}")
endif()

# Build only the lib if subproject
if(NOT DEFINED SPETABARU_JUST_LIB)
    set(SPETABARU_JUST_LIB ${SPETABARU_AS_SUBPROJECT} CACHE BOOL
        "Set to on to build the lib for external use (without test and extra flags)" FORCE)
endif()

if(NOT DEFINED SPETABARU_DO_INSTALL)
    set(SPETABARU_DO_INSTALL OFF CACHE BOOL
        "Set to on to install the headers file of SPETABARU" FORCE)
endif()

if($ENV{VERBOSE})
    MESSAGE(STATUS "Main -- SPETABARU_JUST_LIB ${SPETABARU_JUST_LIB}")
    MESSAGE(STATUS "Main -- SPETABARU_DO_INSTALL ${SPETABARU_DO_INSTALL}")
endif()

#===========================================================================
# Version Number
#===========================================================================

set(SPETABARU_MAJOR_VERSION 0)
set(SPETABARU_MINOR_VERSION 1)
set(SPETABARU_PATCH_VERSION 0)
set(SPETABARU_VERSION  "${SPETABARU_MAJOR_VERSION}.${SPETABARU_MINOR_VERSION}.${SPETABARU_PATCH_VERSION}" )

#===========================================================================
# Options
#===========================================================================

# Extra-flags
SET( SPETABARU_EXTRA_CXX_FLAGS "-funroll-loops" CACHE STRING "These flags are used for all configurations"  )
set(SPETABARU_CXX_FLAGS  "-std=c++17 ${SPETABARU_CXX_FLAGS} ${SPETABARU_EXTRA_CXX_FLAGS}")

# Options
if(NOT SPETABARU_AS_SUBPROJECT)
    SET( SPETABARU_EXTRA_CXX_FLAGS_RELEASE "-O3" CACHE STRING "These flags are added to the usual ones for RELEASE"  )
    set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} ${SPETABARU_EXTRA_CXX_FLAGS_RELEASE}")

    SET( SPETABARU_EXTRA_CXX_FLAGS_DEBUG "-O0" CACHE STRING "These flags are added to the usual ones for DEBUG"  )
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} ${SPETABARU_EXTRA_CXX_FLAGS_DEBUG}")

    set(SPETABARU_CXX_FLAGS "${SPETABARU_CXX_FLAGS} -fpic -Wextra -Wnon-virtual-dtor -Wshadow -Wpointer-arith -Wcast-qual -Wall -pedantic -Woverloaded-virtual -Wpointer-arith -Wcast-qual -Wno-error")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    else()
        set(SPETABARU_CXX_FLAGS "${SPETABARU_CXX_FLAGS} -Wconversion -Wno-sign-conversion")
    endif()

    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "ppc64le")
        # Builting functions are not correctly making variables/parameters used
        set(SPETABARU_CXX_FLAGS "${SPETABARU_CXX_FLAGS} -Wno-unused-but-set-variable -Wno-unused-but-set-parameter")
    endif()

    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(SPETABARU_CXX_FLAGS  "${SPETABARU_CXX_FLAGS} -m64")
    endif()

    OPTION( SPETABARU_ATTACHE_SOURCE "Set to on to add -g flag" OFF )
    if( SPETABARU_ATTACHE_SOURCE )
        set(SPETABARU_CXX_FLAGS  "${SPETABARU_CXX_FLAGS} -g")
    endif(SPETABARU_ATTACHE_SOURCE)
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "icpc.*$")
else()
  set(SPETABARU_CXX_FLAGS "${SPETABARU_CXX_FLAGS} -Wold-style-cast")
endif()

#===========================================================================
# Generate config
#===========================================================================
configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/Src/SPETABARUConfig.h.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/Src/SPETABARUConfig.h  )

#===========================================================================
# Build lib
#===========================================================================
add_subdirectory(Src)


if(NOT SPETABARU_JUST_LIB)
    #===========================================================================
    # Build unit tests
    #===========================================================================
    enable_testing()
    add_subdirectory(UTests)
    add_subdirectory(UTests/ShouldNotCompile)

    #===========================================================================
    # Build examples
    #===========================================================================
    add_subdirectory(Examples)
endif()

#===========================================================================
# Export cmake info
#===========================================================================
configure_file(${SPETABARU_SOURCE_DIR}/CMakeModules/SpetabaruInfo.cmake.in
    ${SPETABARU_BINARY_DIR}/SpetabaruInfo.cmake
    @ONLY IMMEDIATE
)
if(SPETABARU_DO_INSTALL)
    install(FILES  ${SPETABARU_BINARY_DIR}/SpetabaruInfo.cmake  DESTINATION lib/  )
endif()

if(SPETABARU_AS_SUBPROJECT)
    set(SPETABARU_CXX_FLAGS  "${SPETABARU_CXX_FLAGS}" PARENT_SCOPE)
    set(SPETABARU_BINARY_DIR  "${SPETABARU_BINARY_DIR}" PARENT_SCOPE)
    set(SPETABARU_SOURCE_DIR  "${SPETABARU_SOURCE_DIR}" PARENT_SCOPE)
    set(SPETABARU_INCLUDE_DIR  "${SPETABARU_INCLUDE_DIR}" PARENT_SCOPE)
else()
    # nothing - we are not used by another lib here
endif()



