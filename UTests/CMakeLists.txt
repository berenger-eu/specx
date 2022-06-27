###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(UTESTS_SPETABARU CXX)

ADD_DEFINITIONS(${SPETABARU_CXX_FLAGS})

OPTION( SPETABARU_BUILD_UTESTS  "Set to ON to build the unit tests" ON )

if(SPETABARU_BUILD_UTESTS)
    # Find all code files (no GLOB_RECURSE)
    file(
        GLOB	
	    source_tests_files
	    ./*.cpp
	    )


    # Adding the project sources dir as an include dir
    INCLUDE_DIRECTORIES(
         ${SPETABARU_BINARY_DIR}/Src    
         ${SPETABARU_SOURCE_DIR}/Src    
         ${UTESTS_SPETABARU_SOURCE_DIR}
    )


    # Add execs - 1 cpp = 1 exec
    foreach(exec ${source_tests_files}) 
	    get_filename_component(
		    execname ${exec}
		    NAME_WE
	    )
        if($ENV{VERBOSE})
            MESSAGE(STATUS "UTests -- Add ${execname}")
        endif()

        if(SPETABARU_COMPILE_WITH_CUDA)
            set_source_files_properties(${exec} PROPERTIES LANGUAGE CUDA)
        endif()
        add_executable(
			    ${execname}
			    ${exec}
		    )

        target_link_libraries(
            ${execname}
            spetabaru
            ${SPETABARU_LIBRARIES}
            )
        add_test(${execname} ${CMAKE_CURRENT_BINARY_DIR}/${execname})
    endforeach(exec)
endif()
