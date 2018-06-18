#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include <cstddef>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

// #define k_blockDimX    32
// #define k_blockDimMaxY 16
// #define k_blockSizeMin 128
// #define k_blockSizeMax (k_blockDimX * k_blockDimMaxY)

bool getTargetDeviceGlobalMemSize(memsize_t *result, int argc, char **argv);

#endif
