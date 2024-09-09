#pragma once

#define O4FM_ERR_OK 0x00
#define O4FM_ERR_NOT_SUPPORTED 0x01
#define O4FM_ERR_INVALID_ARG 0x02
#define O4FM_ERR_OOM 0x03
#define O4FM_ERR_TEST 0xF0

#define O4FM_ERR_ASSERT(cond, err) { if (!(cond)) return err; }
#define O4FM_ERR_RET(cond) { int8_t ret = cond; if (ret != O4FM_ERR_OK) return ret; }
