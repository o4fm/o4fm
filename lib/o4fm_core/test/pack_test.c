#include <stddef.h>
#include <stdlib.h>

#include "errno.h"
#include "header.h"
#include "pack.h"

int main()
{
  o4fm_core_header_t header = {
    .version = 0x01,
    .mode = O4FM_MODE_NRZ | O4FM_MODE_BAUDRATE_2400,
    .flags = 0x00,
    .fec_mode = O4FM_FEC_NONE,
    .sender_id = 0x00,
    .receiver_id = 0xFF,
    .call_sign = "BH4GTN",
    .body_size = 0x0,
  };

  char* output = NULL;
  O4FM_ERR_RET(o4fm_pack_header(&header, &output));
  O4FM_ERR_ASSERT(output != NULL, O4FM_ERR_TEST);
  free(output);

  return 0;
}
