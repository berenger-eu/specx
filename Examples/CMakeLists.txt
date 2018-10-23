###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(EXAMPLES_SPETABARU CXX)

OPTION( SPETABARU_BUILD_EXAMPLES  "Set to ON to build examples" ON )

if($ENV{VERBOSE})
    MESSAGE(STATUS "Examples -- SPETABARU_BUILD_EXAMPLES = ${SPETABARU_BUILD_EXAMPLES}")
endif()

if( SPETABARU_BUILD_EXAMPLES )
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
