cmake_minimum_required( VERSION 3.12 )

add_library( vtu11 INTERFACE )
target_compile_features( vtu11 INTERFACE cxx_std_11 )
target_include_directories( vtu11 INTERFACE vtu11 )

set_target_properties(vtu11 PROPERTIES FOLDER extern)