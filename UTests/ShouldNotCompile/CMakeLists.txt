###########################################################################
# SPETABARU - Berenger Bramas MPCDF - 2016
# Under LGPL Licence, please you must read the LICENCE file.
###########################################################################
project(UTESTS_NOCOMPILE_SPETABARU CXX)

ADD_DEFINITIONS(${SPETABARU_CXX_FLAGS})

if(SPETABARU_BUILD_UTESTS)
    # Find all code files (no GLOB_RECURSE)
    file(
        GLOB
	    source_comptests_files
	    ./*.cpp
	    )


    # Adding the project sources dir as an include dir
    INCLUDE_DIRECTORIES(
         ${SPETABARU_BINARY_DIR}/Src    
         ${SPETABARU_SOURCE_DIR}/Src    
         ${UTESTS_NOCOMPILE_SPETABARU_SOURCE_DIR}
    )


    # Add execs - 1 cpp = 1 exec
    foreach(exec ${source_comptests_files}) 
        if($ENV{VERBOSE})
            MESSAGE(STATUS "UTests -- Add ${execname} (${exec})")
        endif() 
        
	    get_filename_component(
	        execname ${exec}
	        NAME_WE
        )
        
        file(STRINGS "${exec}" line_nbtests REGEX "@NBTESTS = ")
                
	    if(line_nbtests)
	        string(REGEX REPLACE "// @NBTESTS = ([0-9]+)" "\\1" nbtests_key "${line_nbtests}")
	        
	        if(NOT line_nbtests)
                MESSAGE(ERROR "UTests -- From ${execname} cannot extract number of tests from ${line_nbtests}, check the format")
            endif()
	        
            if($ENV{VERBOSE})
                MESSAGE(STATUS "UTests -- In ${execname} found line ${line_nbtests} --> ${nbtests_key} tests")
            endif()
	    
		    foreach(test_var RANGE 1 "${nbtests_key}")
		        if($ENV{VERBOSE})
                    MESSAGE(STATUS "UTests -- Add ${execname}-${test_var}")
                endif() 
                
		        add_executable(
			        ${execname}-${test_var}
			        ${exec}
		        )
	            set_target_properties(${execname}-${test_var} PROPERTIES
                               EXCLUDE_FROM_ALL TRUE
                               EXCLUDE_FROM_DEFAULT_BUILD TRUE)

                target_link_libraries(
                    ${execname}-${test_var}
                    spetabaru
                    ${SPETABARU_LIBRARIES}
                    )
                    
                target_compile_definitions(${execname}-${test_var} PRIVATE "TEST${test_var}")
                
                add_test(NAME ${execname}-${test_var}
                         COMMAND ${CMAKE_COMMAND} --build . --target ${execname}-${test_var} --config $<CONFIGURATION>
                         WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

                set_tests_properties(${execname}-${test_var} PROPERTIES WILL_FAIL TRUE)
		    endforeach()
		else()
            add_executable(
			        ${execname}
			        ${exec}
		        )
		    set_target_properties(${execname} PROPERTIES
                           EXCLUDE_FROM_ALL TRUE
                           EXCLUDE_FROM_DEFAULT_BUILD TRUE)

            target_link_libraries(
                ${execname}
                spetabaru
                ${SPETABARU_LIBRARIES}
                )
            add_test(NAME ${execname}
                     COMMAND ${CMAKE_COMMAND} --build . --target ${execname} --config $<CONFIGURATION>
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

            set_tests_properties(${execname} PROPERTIES WILL_FAIL TRUE)
	    endif()        
    endforeach(exec)
endif()
