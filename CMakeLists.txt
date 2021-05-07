cmake_minimum_required(VERSION 3.9)

project(
    Particle
    VERSION 0.1
    DESCRIPTION "Particle")

#if (MSVC)
#    add_compile_options(/O2)
#else()
#    add_compile_options(-xO3)
#endif()

add_subdirectory(solver)
add_subdirectory(extern)
#add_subdirectory(include)    # look in apps subdirectory for CMakeLists.txt to process
#set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT app)