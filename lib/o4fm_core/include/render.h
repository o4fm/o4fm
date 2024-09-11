#pragma once

#include <stddef.h>
#include <stdint.h>

#define O4FM_RENDER_SAMPLE_RATE 48000

// output is 16-bit signed PCM
int8_t o4fm_render_pcm(const char* source, size_t source_size, uint8_t mode, size_t* p_output_size, int16_t** p_output);
