#pragma once

#include <stdint.h>

#include "header.h"

int8_t o4fm_pack_header(const o4fm_core_header_t *header, char** p_output);
int8_t o4fm_pack_payload(const char* payload, size_t payload_size, uint8_t fec_mode, char** p_output, size_t* p_output_size);
