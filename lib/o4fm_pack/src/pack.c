#include <stddef.h>
#include <stdlib.h>

#include "errno.h"
#include "pack.h"

int8_t o4fm_pack_header(o4fm_pack_header_t *header, char** output)
{
  if (header == NULL || output == NULL)
    return O4FM_ERR_INVALID_ARG;
  
  *output = (char*)malloc(8);
  if (*output == NULL)
    return O4FM_ERR_OOM_ERROR;

  *output[0] = header->version;
  *output[1] = header->mode;
  *output[2] = header->flags;
  *output[3] = header->fec_mode;
  *output[4] = header->sender_id;
  *output[5] = header->receiver_id;

  // in big endian
  *output[6] = header->body_size >> 8;
  *output[7] = header->body_size & 0xFF;

  return O4FM_ERR_OK;
}
