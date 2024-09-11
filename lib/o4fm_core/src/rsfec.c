#include <stdlib.h>
#include <string.h>

#include "rsfec.h"
#include "errno.h"

// Reed-Solomon codec structure
typedef struct {
    int mm;          // Bits per symbol
    int nn;          // Symbols per block (= (1<<mm)-1)
    int tt;          // Number of error correction symbols
    int kk;          // Number of data symbols
    int pad;         // Padding bytes in shortened block
    uint16_t *alpha_to;  // log lookup table
    uint16_t *index_of;  // antilog lookup table
    uint16_t *genpoly;   // Generator polynomial
    int fcr;         // First consecutive root
    int prim;        // Primitive element
    int nroots;      // Number of generator roots = number of parity symbols
} rs_control_t;

typedef struct {
    uint8_t symsize;
    uint8_t gfpoly;
    uint8_t fcr;
    uint8_t prim;
    uint8_t nroots;
    rs_control_t* rs;  // Pointer to the Reed-Solomon control structure
} rs_codec_t;

static rs_codec_t* codec = NULL;

static rs_control_t* init_rs_char(int symsize, int gfpoly, int fcr, int prim, int nroots, int pad);
static void encode_rs(rs_control_t* rs, uint16_t* data, uint16_t* parity);
static inline int mod_nn(rs_control_t* rs, int x);
static int decode_rs(rs_control_t* rs, uint16_t* data);

#define MIN(a,b) ((a) < (b) ? (a) : (b))

static rs_control_t* init_rs_char(int symsize, int gfpoly, int fcr, int prim, int nroots, int pad)
{
    int i, j, sr, root, iprim;
    uint16_t* alpha_to;
    uint16_t* index_of;
    uint16_t* genpoly;
    int nn, tt, kk;

    if (symsize < 0 || symsize > 8) return NULL; // Invalid symbol size

    if (fcr < 0 || fcr >= (1<<symsize)) return NULL; // Invalid first consecutive root
    if (prim <= 0 || prim >= (1<<symsize)) return NULL; // Invalid primitive element
    if (nroots < 0 || nroots >= (1<<symsize)) return NULL; // Invalid number of roots

    nn = (1<<symsize) - 1;
    tt = nroots;
    kk = nn - tt;

    if (kk <= 0) return NULL; // Invalid parameters

    alpha_to = (uint16_t*)malloc(sizeof(uint16_t) * (nn + 1));
    index_of = (uint16_t*)malloc(sizeof(uint16_t) * (nn + 1));
    genpoly = (uint16_t*)malloc(sizeof(uint16_t) * (tt + 1));

    if (!alpha_to || !index_of || !genpoly) {
        free(alpha_to);
        free(index_of);
        free(genpoly);
        return NULL; // Memory allocation failed
    }

    // Generate Galois Field
    index_of[0] = nn;
    alpha_to[nn] = 0;
    sr = 1;
    for (i = 0; i < nn; i++) {
        index_of[sr] = i;
        alpha_to[i] = sr;
        sr <<= 1;
        if (sr & (1 << symsize)) sr ^= gfpoly;
        sr &= nn;
    }

    // Generate generator polynomial
    genpoly[0] = 1;
    for (i = 0, root = fcr * prim; i < tt; i++, root += prim) {
        genpoly[i+1] = 1;
        for (j = i; j > 0; j--) {
            if (genpoly[j] != 0)
                genpoly[j] = genpoly[j-1] ^ alpha_to[(index_of[genpoly[j]] + root) % nn];
            else
                genpoly[j] = genpoly[j-1];
        }
        genpoly[0] = alpha_to[(index_of[genpoly[0]] + root) % nn];
    }

    // Allocate and initialize RS control structure
    rs_control_t* rs = (rs_control_t*)malloc(sizeof(rs_control_t));
    if (!rs) {
        free(alpha_to);
        free(index_of);
        free(genpoly);
        return NULL;
    }

    rs->mm = symsize;
    rs->nn = nn;
    rs->tt = tt;
    rs->kk = kk;
    rs->pad = pad;
    rs->alpha_to = alpha_to;
    rs->index_of = index_of;
    rs->genpoly = genpoly;
    rs->fcr = fcr;
    rs->prim = prim;
    rs->nroots = nroots;

    return (void*)rs;
}

static void free_rs_char(rs_control_t* rs)
{
    if (rs) {
        free(rs->alpha_to);
        free(rs->index_of);
        free(rs->genpoly);
        free(rs);
    }
}

int8_t o4fm_rsfec_init(uint8_t symsize, uint8_t gfpoly, uint8_t fcr, uint8_t prim, uint8_t nroots)
{
    O4FM_ERR_ASSERT(codec == NULL, O4FM_ERR_INVALID_ARG);  // Ensure not already initialized

    codec = (rs_codec_t*)malloc(sizeof(rs_codec_t));
    O4FM_ERR_ASSERT(codec != NULL, O4FM_ERR_OOM);

    codec->symsize = symsize;
    codec->gfpoly = gfpoly;
    codec->fcr = fcr;
    codec->prim = prim;
    codec->nroots = nroots;

    // Initialize the actual Reed-Solomon control structure
    codec->rs = init_rs_char(symsize, gfpoly, fcr, prim, nroots, 0);
    O4FM_ERR_ASSERT(codec->rs != NULL, O4FM_ERR_INVALID_ARG);

    return O4FM_ERR_OK;
}

void o4fm_rsfec_cleanup()
{
    if (codec != NULL) {
        free_rs_char(codec->rs);
        free(codec);
        codec = NULL;
    }
}

int8_t o4fm_rsfec_encode(const uint8_t *data, size_t data_len, uint8_t *encoded, size_t *encoded_len)
{
    O4FM_ERR_ASSERT(codec != NULL, O4FM_ERR_INVALID_ARG);
    O4FM_ERR_ASSERT(data != NULL && encoded != NULL && encoded_len != NULL, O4FM_ERR_INVALID_ARG);

    rs_control_t* rs = (rs_control_t*)codec->rs;
    int data_symbols = data_len / rs->mm;
    int total_symbols = rs->nn;
    int parity_symbols = rs->nroots;

    uint16_t* data_and_parity = (uint16_t*)malloc(total_symbols * sizeof(uint16_t));
    if (!data_and_parity) {
        return O4FM_ERR_OOM;
    }

    // Copy data into the encoding buffer
    memset(data_and_parity, 0, total_symbols * sizeof(uint16_t));
    for (int i = 0; i < data_len; i++) {
        data_and_parity[i / rs->mm] |= (data[i] & 0xFF) << (8 * (i % rs->mm));
    }

    // Perform Reed-Solomon encoding
    encode_rs(rs, data_and_parity, data_and_parity + data_symbols);

    // Copy encoded data to output buffer
    *encoded_len = total_symbols * rs->mm;
    for (int i = 0; i < total_symbols; i++) {
        for (int j = 0; j < rs->mm; j++) {
            encoded[i * rs->mm + j] = (data_and_parity[i] >> (8 * j)) & 0xFF;
        }
    }

    free(data_and_parity);

    return O4FM_ERR_OK;
}

int8_t o4fm_rsfec_decode(const uint8_t *encoded, size_t encoded_len, uint8_t *decoded, size_t *decoded_len)
{
    O4FM_ERR_ASSERT(codec != NULL, O4FM_ERR_INVALID_ARG);
    O4FM_ERR_ASSERT(encoded != NULL && decoded != NULL && decoded_len != NULL, O4FM_ERR_INVALID_ARG);

    rs_control_t* rs = (rs_control_t*)codec->rs;
    int total_symbols = rs->nn;
    int data_symbols = rs->kk;
    int parity_symbols = rs->nroots;

    uint16_t* data_and_parity = (uint16_t*)malloc(total_symbols * sizeof(uint16_t));
    if (!data_and_parity) {
        return O4FM_ERR_OOM;
    }

    // Copy encoded data into the decoding buffer
    for (int i = 0; i < total_symbols; i++) {
        data_and_parity[i] = 0;
        for (int j = 0; j < rs->mm; j++) {
            data_and_parity[i] |= (encoded[i * rs->mm + j] & 0xFF) << (8 * j);
        }
    }

    // Perform Reed-Solomon decoding
    int error_count = decode_rs(rs, data_and_parity);

    if (error_count < 0) {
        free(data_and_parity);
        return O4FM_ERR_INVALID_ARG; // Decoding failed, too many errors
    }

    // Copy decoded data to output buffer
    *decoded_len = data_symbols * rs->mm;
    for (int i = 0; i < data_symbols; i++) {
        for (int j = 0; j < rs->mm; j++) {
            decoded[i * rs->mm + j] = (data_and_parity[i] >> (8 * j)) & 0xFF;
        }
    }

    free(data_and_parity);

    return O4FM_ERR_OK;
}

static void encode_rs(rs_control_t* rs, uint16_t* data, uint16_t* parity)
{
    int i, j;
    uint16_t feedback;

    memset(parity, 0, rs->nroots * sizeof(uint16_t));

    for (i = 0; i < rs->nn - rs->nroots; i++) {
        feedback = rs->index_of[data[i] ^ parity[0]];
        if (feedback != rs->nn) { // non-zero
            for (j = 1; j < rs->nroots; j++) {
                parity[j-1] = parity[j] ^ rs->alpha_to[mod_nn(rs, feedback + rs->genpoly[rs->nroots-j])];
            }
            parity[rs->nroots-1] = rs->alpha_to[mod_nn(rs, feedback + rs->genpoly[0])];
        } else {
            // Shift
            memmove(parity, parity + 1, (rs->nroots - 1) * sizeof(uint16_t));
            parity[rs->nroots-1] = 0;
        }
    }
}

static inline int mod_nn(rs_control_t* rs, int x)
{
    while (x >= rs->nn) {
        x -= rs->nn;
        x = (x >> rs->mm) + (x & rs->nn);
    }
    return x;
}

static int decode_rs(rs_control_t* rs, uint16_t* data)
{
    int i, j, r, k;
    uint16_t syndrome[rs->nroots];
    uint16_t loc[rs->nroots], elp[rs->nroots + 1], deg_elp;
    uint16_t reg[rs->nroots + 1], root[rs->nroots], locn[rs->nroots];
    int count = 0;

    // Compute syndrome
    for (i = 0; i < rs->nroots; i++) {
        syndrome[i] = data[0];
        for (j = 1; j < rs->nn; j++) {
            syndrome[i] = data[j] ^ rs->alpha_to[mod_nn(rs, rs->index_of[syndrome[i]] + (rs->fcr + i) * rs->prim)];
        }
        if (syndrome[i] != 0) {
            count++;
        }
    }

    if (count == 0) {
        return 0; // No errors
    }

    // Berlekamp-Massey algorithm
    elp[0] = 1;
    for (r = 0; r < rs->nroots; r++) {
        deg_elp = 0;
        for (i = 0; i < r; i++) {
            if (elp[i] != 0) {
                deg_elp = i;
            }
        }
        for (i = 1; i <= rs->nroots; i++) {
            reg[i] = elp[i];
        }
        for (i = 0; i < rs->nroots; i++) {
            if (syndrome[i] == 0) {
                continue;
            }
            k = rs->index_of[syndrome[i]];
            for (j = deg_elp; j >= 0; j--) {
                if (elp[j] != 0) {
                    reg[j + 1] ^= rs->alpha_to[mod_nn(rs, rs->index_of[elp[j]] + k)];
                }
            }
        }
        for (i = 0; i <= rs->nroots; i++) {
            elp[i] = reg[i];
        }
    }

    // Find roots of the error locator polynomial
    count = 0;
    for (i = 1; i <= rs->nn; i++) {
        uint16_t q = 1;
        for (j = 0; j < rs->nroots + 1; j++) {
            if (elp[j] != 0) {
                q ^= rs->alpha_to[mod_nn(rs, rs->index_of[elp[j]] + (rs->nn - i) * j)];
            }
        }
        if (q == 0) {
            root[count] = i;
            locn[count] = rs->nn - i;
            count++;
        }
    }

    if (count != deg_elp) {
        return -1; // Decoding failed
    }

    // Correct the errors
    for (i = 0; i < count; i++) {
        if (locn[i] < rs->nn - rs->nroots) {
            data[locn[i]] ^= 1; // Toggle the error bit
        }
    }

    return count;
}
