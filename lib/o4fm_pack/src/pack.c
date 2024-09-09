#include <stddef.h>
#include <stdlib.h>

#include "errno.h"
#include "pack.h"

int8_t o4fm_pack_header(o4fm_pack_header_t *header, char** p_output)
{
  O4FM_ERR_ASSERT(header != NULL, O4FM_ERR_INVALID_ARG);
  O4FM_ERR_ASSERT(p_output != NULL, O4FM_ERR_INVALID_ARG);
  
  *p_output = (char*)malloc(8);
  char* output = *p_output;

  O4FM_ERR_ASSERT(output != NULL, O4FM_ERR_OOM);

  output[0] = header->version;
  output[1] = header->mode;
  output[2] = header->flags;
  output[3] = header->fec_mode;
  output[4] = header->sender_id;
  output[5] = header->receiver_id;

  // in big endian
  output[6] = header->body_size >> 8;
  output[7] = header->body_size & 0xFF;

  return O4FM_ERR_OK;
}
