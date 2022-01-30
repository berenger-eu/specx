###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(PATTERNS_SPETABARU CXX)

ADD_DEFINITIONS(${SPETABARU_CXX_FLAGS})


if(SPETABARU_BUILD_EXAMPLES)
    OPTION( SPETABARU_BUILD_CHOLESKY_EXAMPLES  "Set to ON to build cholesky example" ON )

    if(SPETABARU_BUILD_CHOLESKY_EXAMPLES)
        if($ENV{VERBOSE})
            MESSAGE(STATUS "Examples -- SPETABARU_CXX_FLAGS   : ${SPETABARU_CXX_FLAGS}")
        endif()

        # Find all code files
        file(	
	        GLOB_RECURSE
	        source_tests_files
	        ./*.cpp
	        )


        # Adding the project sources dir as an include dir
        INCLUDE_DIRECTORIES(
             ${SPETABARU_BINARY_DIR}/Src    
             ${SPETABARU_SOURCE_DIR}/Src
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
                MESSAGE(STATUS "Patterns -- Add ${execname}")
            endif()

            add_executable(
			        ${execname}
			        ${exec}
		        )

            set_target_properties(
                ${execname}
                PROPERTIES COMPILE_FLAGS "-llapack -lblas"
            )
            target_link_libraries(
                ${execname}
                spetabaru
	            ${SPETABARU_LIBRARIES}
                "-llapack -lblas"
                ) 
        endforeach(exec)
    endif()
endif()
