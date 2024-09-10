#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "errno.h"
#include "header.h"
#include "render.h"

#define TEST_DATA "BH4GTN"
#define TEST_DATA_SIZE 6

int test_render_ok(uint8_t mode)
{
  size_t output_size = 0;
  int16_t* output = NULL;
  O4FM_ERR_RET(o4fm_render_pcm(TEST_DATA, TEST_DATA_SIZE, mode, &output_size, &output));
  O4FM_ERR_ASSERT(output != NULL, O4FM_ERR_TEST);

  free(output);

  return 0;
}

int main()
{
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_NRZ | O4FM_MODE_BAUDRATE_2400));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_PAM4 | O4FM_MODE_BAUDRATE_2400));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_PAM16 | O4FM_MODE_BAUDRATE_2400));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_NRZ | O4FM_MODE_BAUDRATE_4800));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_PAM4 | O4FM_MODE_BAUDRATE_4800));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_PAM16 | O4FM_MODE_BAUDRATE_4800));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_NRZ | O4FM_MODE_BAUDRATE_9600));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_PAM4 | O4FM_MODE_BAUDRATE_9600));
  O4FM_ERR_RET(test_render_ok(O4FM_MODE_PAM16 | O4FM_MODE_BAUDRATE_9600));

  return 0;
}
