#pragma once

#include <stddef.h>
#include <stdint.h>

#define O4FM_RENDER_SAMPLE_RATE 48000
#define O4FM_RENDER_SYMBOL_OFFSET_CENTER 5000

static int32_t o4fm_render_symbol_offsets[] = { \
  -2400, 2400, \
  -1200, 1200, \
  -300, 300, -600, 600, -900, 900, -1500, 1500, -1800, 1800, -2100, 2100 \
};

int8_t o4fm_render(char* source, size_t source_size, uint8_t mode, char** p_output);
