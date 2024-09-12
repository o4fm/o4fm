#pragma once

#define O4FM_ERR_OK 0
#define O4FM_ERR_NOT_SUPPORTED -1
#define O4FM_ERR_INVALID_ARG -2
#define O4FM_ERR_OOM -3
#define O4FM_ERR_CHECKSUM -4
#define O4FM_ERR_TEST -64

#define O4FM_ERR_ASSERT(cond, err) { if (!(cond)) return err; }
#define O4FM_ERR_RET(cond) { int8_t ret = cond; if (ret != O4FM_ERR_OK) return ret; }
