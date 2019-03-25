cmake_minimum_required (VERSION 3.2)

project (archerycounter)

include_directories("${PROJECT_SOURCE_DIR}")

if (WIN32)
add_definitions(/std:c++latest)
else (WIN32)
add_definitions(-std=c++17)
endif(WIN32)

set(OpenCV_DIR "${PROJECT_SOURCE_DIR}" CACHE PATH "The root directory of OpenCV")

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# sources in the resolver_server directory
set (SOURCES 
	${PROJECT_SOURCE_DIR}/main.cpp
	)

add_executable(ArcheryCounter ${SOURCES})
target_link_libraries( ArcheryCounter ${OpenCV_LIBS} )