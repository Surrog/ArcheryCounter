cmake_minimum_required (VERSION 3.2)

project (archerycounter)

include_directories("${PROJECT_SOURCE_DIR}")

if (WIN32)
add_definitions(/std:c++latest)
else (WIN32)
add_definitions(-std=c++17)
endif(WIN32)

# sources in the resolver_server directory
set (SOURCES 
	${PROJECT_SOURCE_DIR}/main.cpp
	)

add_executable(ArcheryCounter ${SOURCES})
