cmake_minimum_required(VERSION 3.27)

add_library(
  o4fm_pack
  include/errno.h
  include/header.h
  include/pack.h
  src/pack.c
  src/render.c
)
target_include_directories(o4fm_pack PRIVATE include)
target_link_libraries(o4fm_pack m)
