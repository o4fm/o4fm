#include <stddef.h>

#include "errno.h"
#include "pack.h"

int8_t o4fm_pack_header(o4fm_pack_header_t *header, char* target)
{
  if (header == NULL || target == NULL)
    return O4FM_ERR_INVALID_ARG;

  target[0] = header->version;
  target[1] = header->mode;
  target[2] = header->flags;
  target[3] = header->fec_mode;
  target[4] = header->sender_id;
  target[5] = header->receiver_id;

  // in big endian
  target[6] = header->body_size >> 8;
  target[7] = header->body_size & 0xFF;

  return O4FM_ERR_OK;
}
