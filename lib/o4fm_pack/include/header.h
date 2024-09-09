#pragma once

#include <stdint.h>

#define O4FM_MODE_FSK_2 0x00
#define O4FM_MODE_FSK_4 0x01
#define O4FM_MODE_FSK_16 0x02

#define O4FM_MODE_SYMBOL_RATE_2400 0x00
#define O4FM_MODE_SYMBOL_RATE_4800 0x10
#define O4FM_MODE_SYMBOL_RATE_9600 0x20

#define O4FM_FEC_NONE 0
#define O4FM_FEC_RS_255_239 1
#define O4FM_FEC_RS_255_223 2
#define O4FM_FEC_RS_255_191 3

typedef struct o4fm_pack_header
{
  uint8_t version;
  uint8_t mode;
  uint8_t flags;
  uint8_t fec_mode;
  uint8_t sender_id; // group_id(4) + node_id(4)
  uint8_t receiver_id; // group_id(4) + node_id(4), 0xFF for broadcast, 0xxF for multicast
  char call_sign[16]; // ASCII call sign
  uint16_t body_size; // FEC encoded payload size
} o4fm_pack_header_t;
