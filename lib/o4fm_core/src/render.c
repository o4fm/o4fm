#include "errno.h"
#include "header.h"
#include "render.h"

#include <limits.h>
#include <stdlib.h>

static int8_t o4fm_render_parse_baudrate(uint8_t mode, size_t* baudrate)
{
  O4FM_ERR_ASSERT(baudrate != NULL, O4FM_ERR_INVALID_ARG);

  switch (mode & 0xF0)
  {
    case O4FM_MODE_BAUDRATE_2400:
      *baudrate = 2400;
      break;
    case O4FM_MODE_BAUDRATE_4800:
      *baudrate = 4800;
      break;
    case O4FM_MODE_BAUDRATE_9600:
      *baudrate = 9600;
      break;
    default:
      return O4FM_ERR_INVALID_ARG;
  }
  return O4FM_ERR_OK;
}

static int8_t o4fm_render_parse_symbol_size(uint8_t mode, size_t* symbol_size)
{
  O4FM_ERR_ASSERT(symbol_size != NULL, O4FM_ERR_INVALID_ARG);

  switch (mode & 0x0F)
  {
    case O4FM_MODE_NRZ:
      *symbol_size = 1;
      break;
    case O4FM_MODE_PAM4:
      *symbol_size = 2;
      break;
    case O4FM_MODE_PAM16:
      *symbol_size = 8;
      break;
    default:
      return O4FM_ERR_INVALID_ARG;
  }
  return O4FM_ERR_OK;
}

static inline size_t o4fm_render_calculate_output_size(size_t source_size, size_t symbol_size, size_t baudrate)
{
  // output_size = 
  //   source_size (in bytes) * 8 (in bits)
  //     / symbol_size (in symbols)
  //     / baudrate (in symbols per second)
  //     * sample_rate (in samples per second)
  // rearrange the formula to avoid overflow or floating point
  return source_size * 8 * O4FM_RENDER_SAMPLE_RATE / symbol_size / baudrate;
}

int8_t o4fm_render_pcm(const char* source, size_t source_size, uint8_t mode, size_t* p_output_size, int16_t** p_output)
{
  O4FM_ERR_ASSERT(source != NULL, O4FM_ERR_INVALID_ARG);
  O4FM_ERR_ASSERT(p_output_size != NULL, O4FM_ERR_INVALID_ARG);
  O4FM_ERR_ASSERT(p_output != NULL, O4FM_ERR_INVALID_ARG);
  
  size_t baudrate = 2400;
  O4FM_ERR_RET(o4fm_render_parse_baudrate(mode, &baudrate));
  size_t symbol_size = 1;
  O4FM_ERR_RET(o4fm_render_parse_symbol_size(mode, &symbol_size));

  *p_output_size = o4fm_render_calculate_output_size(source_size, symbol_size, baudrate);
  *p_output = (int16_t*)malloc(*p_output_size * sizeof(int16_t));
  O4FM_ERR_ASSERT(*p_output != NULL, O4FM_ERR_OOM);
  
  size_t samples_per_symbol = O4FM_RENDER_SAMPLE_RATE / baudrate;
  size_t current_sample = 0;

  for (size_t i = 0; i < source_size; i++) {
    for (size_t bit = 0; bit < 8; bit += symbol_size) {
      int16_t symbol_value = 0;
      
      switch (mode & 0x0F) {
        case O4FM_MODE_NRZ:
          symbol_value = (source[i] & (1 << (7 - bit))) ? INT16_MAX : INT16_MIN;
          break;
        case O4FM_MODE_PAM4:
          symbol_value = ((source[i] >> (6 - bit)) & 0x03) * (INT16_MAX / 3) - INT16_MAX;
          break;
        case O4FM_MODE_PAM16:
          symbol_value = ((source[i] >> (4 - bit)) & 0x0F) * (INT16_MAX / 15) - INT16_MAX;
          break;
      }
      
      for (size_t j = 0; j < samples_per_symbol; j++) {
        (*p_output)[current_sample++] = symbol_value;
      }
    }
  }

  return O4FM_ERR_OK;
}
