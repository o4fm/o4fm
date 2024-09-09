#include "errno.h"
#include "header.h"
#include "render.h"

#include <stdlib.h>

static int8_t o4fm_render_parse_symbol_rate(uint8_t mode, size_t* symbol_rate)
{
  O4FM_ERR_ASSERT(symbol_rate != NULL, O4FM_ERR_INVALID_ARG);

  switch (mode & 0xF0)
  {
    case O4FM_MODE_SYMBOL_RATE_2400:
      *symbol_rate = 2400;
      break;
    case O4FM_MODE_SYMBOL_RATE_4800:
      *symbol_rate = 4800;
      break;
    case O4FM_MODE_SYMBOL_RATE_9600:
      *symbol_rate = 9600;
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
    case O4FM_MODE_FSK_2:
      *symbol_size = 1;
      break;
    case O4FM_MODE_FSK_4:
      *symbol_size = 2;
      break;
    case O4FM_MODE_FSK_16:
      *symbol_size = 4;
      break;
    default:
      return O4FM_ERR_INVALID_ARG;
  }
  return O4FM_ERR_OK;
}

static inline size_t o4fm_render_calculate_output_size(size_t source_size, size_t symbol_size, size_t symbol_rate)
{
  // output_size = 
  //   source_size (in bytes) * 8 (in bits)
  //     / symbol_size (in symbols)
  //     / symbol_rate (in symbols per second)
  //     * sample_rate (in samples per second)
  // rearrange the formula to avoid overflow or floating point
  return source_size * 8 * O4FM_RENDER_SAMPLE_RATE / symbol_size / symbol_rate;
}

int8_t o4fm_render(char* source, size_t source_size, uint8_t mode, char** p_output)
{
  O4FM_ERR_ASSERT(source != NULL, O4FM_ERR_INVALID_ARG);
  O4FM_ERR_ASSERT(p_output != NULL, O4FM_ERR_INVALID_ARG);
  
  size_t symbol_rate = 2400;
  O4FM_ERR_RET(o4fm_render_parse_symbol_rate(mode, &symbol_rate));
  size_t symbol_size = 1;
  O4FM_ERR_RET(o4fm_render_parse_symbol_size(mode, &symbol_size));

  size_t output_size = o4fm_render_calculate_output_size(source_size, symbol_size, symbol_rate);

  *p_output = (char*)malloc(output_size);
  O4FM_ERR_ASSERT(*p_output != NULL, O4FM_ERR_OOM);
  
  char* output = *p_output;
  
  return O4FM_ERR_OK;
}
