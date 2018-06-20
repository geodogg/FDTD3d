// C LIBRARY INCLUDES
#include <cstdio>
#include <cassert>  // useful for debugging
#include <cstdlib>
#include <cmath>
using namespace std;

// CUDA LIBRARY INCLUDES
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

// EXTRA
#include "functions.h"

bool getTargetDeviceGlobalMemSize(memsize_t *result, int argc, char **argv)
{
    int               deviceCount  = 0;
    int               targetDevice = 0;
    size_t            memsize      = 0;

    // Get the number of CUDA enabled GPU devices
    printf(" cudaGetDeviceCount\n");
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findCudaDevice(argc, (const char **)argv);

    // Query target device for maximum memory allocation
    printf(" cudaGetDeviceProperties\n");
    struct cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

    memsize = deviceProp.totalGlobalMem;

    // Save the result
    *result = (memsize_t)memsize;
    return true;
}

void generateRandomData(float *data, const int dimx, const int dimy, const int dimz, const float lowerBound, const float upperBound)
{
    srand(0);

    for (int iz = 0 ; iz < dimz ; iz++)
    {
        for (int iy = 0 ; iy < dimy ; iy++)
        {
            for (int ix = 0 ; ix < dimx ; ix++)
            {
                *data = (float)(lowerBound + ((float)rand() / (float)RAND_MAX) * (upperBound - lowerBound));
                ++data;
            }
        }
    }
}
