#pragma once

#include <stddef.h>
#include <stdint.h>

int8_t o4fm_render(char* source, size_t source_size, uint8_t mode, size_t sample_rate, char** p_output);
