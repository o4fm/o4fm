#pragma once

#define O4FM_ERR_OK 0
#define O4FM_ERR_NOT_SUPPORTED -1
#define O4FM_ERR_INVALID_ARG -2
#define O4FM_ERR_OOM_ERROR -3
#define O4FM_ERR_DATA_ERROR -4
#define O4FM_ERR_ASSERT_ERROR -5

#define ASSERT(cond) { if (!(cond)) return O4FM_ERR_ASSERT_ERROR; }
#define ASSERT_RET(cond) { int8_t ret = cond; if (ret != O4FM_ERR_OK) return ret; }
