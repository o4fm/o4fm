#pragma once

#include <stdint.h>
#include <stddef.h>

// Initialize Reed-Solomon encoder/decoder with specific parameters
int8_t o4fm_rsfec_init(uint8_t symsize, uint8_t gfpoly, uint8_t fcr, uint8_t prim, uint8_t nroots);

// Reed-Solomon Forward Error Correction encoding function
int8_t o4fm_rsfec_encode(const uint8_t *data, size_t data_len, uint8_t *encoded, size_t *encoded_len);

// Reed-Solomon Forward Error Correction decoding function
int8_t o4fm_rsfec_decode(const uint8_t *encoded, size_t encoded_len, uint8_t *decoded, size_t *decoded_len);

// Clean up Reed-Solomon encoder/decoder resources
void o4fm_rsfec_cleanup();
