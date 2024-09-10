cmake_minimum_required(VERSION 3.27)

project(o4fm VERSION 0.1.0 LANGUAGES C)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(lib/o4fm_pack)

enable_testing()
add_test(NAME o4fm_pack_pack_test COMMAND $<TARGET_FILE:o4fm_pack_pack_test>)
add_test(NAME o4fm_pack_render_test COMMAND $<TARGET_FILE:o4fm_pack_render_test>)
