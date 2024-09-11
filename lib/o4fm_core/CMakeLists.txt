cmake_minimum_required(VERSION 3.27)

add_library(
  o4fm_core
  include/errno.h
  include/header.h
  include/pack.h
  include/render.h
  src/pack.c
  src/render.c
)
target_include_directories(o4fm_core PRIVATE include)
target_link_libraries(o4fm_core m)

add_executable(o4fm_core_pack_test test/pack_test.c)
target_include_directories(o4fm_core_pack_test PRIVATE include)
target_link_libraries(o4fm_core_pack_test o4fm_core)

add_executable(o4fm_core_render_test test/render_test.c)
target_include_directories(o4fm_core_render_test PRIVATE include)
target_link_libraries(o4fm_core_render_test o4fm_core)
