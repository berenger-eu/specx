###########################################################################
# SPECX - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(EXAMPLES_SPECX CXX)

OPTION( SPECX_BUILD_EXAMPLES  "Set to ON to build examples" ON )

if($ENV{VERBOSE})
    MESSAGE(STATUS "Examples -- SPECX_BUILD_EXAMPLES = ${SPECX_BUILD_EXAMPLES}")
endif()

if( SPECX_BUILD_EXAMPLES )
    file(GLOB all_examples ./*)

    foreach(example_name ${all_examples})
        if(IS_DIRECTORY ${example_name})
          if($ENV{VERBOSE})
              message(STATUS "Examples -- ${example_name}" )
          endif()
          add_subdirectory(${example_name})
        endif()
    endforeach()
endif()
