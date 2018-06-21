// C LIBRARY INCLUDES
#include <cstdio>
#include <cassert>  // useful for debugging
#include <cstdlib>
#include <ctime>
using namespace std;

// CUDA LIBRARY INCLUDES
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
namespace hf = helper_functions;
// EXTRA
#include "functions.h"

// GPU ERROR CHECKING MACROS
#define gpuErrchk(ans){ gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort = true){
  if (code != cudaSuccess){
    fprintf(stderr, "gpuAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define printline(ans) { fprintf(outfile, "file: %s line: %d\n - ", __FILE__, __LINE__); fprintf(outfile, ans); }

// DEFINE CONSTANTS DATA
#define dimx 376
#define dimy 376
#define dimz 376
#define radius 4
#define timesteps 5
#define outerDimx 384
#define outerDimy 384
#define outerDimz 384
#define volumeSize 56623104
#define lowerBound 0.0f
#define upperBound 1.0f
#define padding 28
#define paddedVolumeSize 56623132

// KERNELS
// #define RADIUS 4
// __constant__ float stencil[RADIUS + 1];
// __global__ void FiniteDifferencesKernel(float *bufferDst, float *bufferSrc)
// {
//     bool validr = true;
//     bool validw = true;
//     const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
//     const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
//     const int ltidx = threadIdx.x;
//     const int ltidy = threadIdx.y;
//     const int workx = blockDim.x;
//     const int worky = blockDim.y;
//     // Handle to thread block group
//     cg::thread_block cta = cg::this_thread_block();
//     __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];
//
//     const int stride_y = dimx + 2 * RADIUS;
//     const int stride_z = stride_y * (dimy + 2 * RADIUS);
//
//     int inputIndex  = 0;
//     int outputIndex = 0;
//
//     // Advance inputIndex to start of inner volume
//     inputIndex += RADIUS * stride_y + RADIUS;
//
//     // Advance inputIndex to target element
//     inputIndex += gtidy * stride_y + gtidx;
//
//     float infront[RADIUS];
//     float behind[RADIUS];
//     float current;
//
//     const int tx = ltidx + RADIUS;
//     const int ty = ltidy + RADIUS;
//
//     // Check in bounds
//     if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS))
//         validr = false;
//
//     if ((gtidx >= dimx) || (gtidy >= dimy))
//         validw = false;
//
//     // Preload the "infront" and "behind" data
//     for (int i = RADIUS - 2 ; i >= 0 ; i--)
//     {
//         if (validr)
//             behind[i] = bufferSrc[inputIndex];
//
//         inputIndex += stride_z;
//     }
//
//     if (validr)
//         current = bufferSrc[inputIndex];
//
//     outputIndex = inputIndex;
//     inputIndex += stride_z;
//
//     for (int i = 0 ; i < RADIUS ; i++)
//     {
//         if (validr)
//             infront[i] = bufferSrc[inputIndex];
//
//         inputIndex += stride_z;
//     }
//
//     // Step through the xy-planes
//     #pragma unroll 9
//
//     for (int iz = 0 ; iz < dimz ; iz++)
//     {
//         // Advance the slice (move the thread-front)
//         for (int i = RADIUS - 1 ; i > 0 ; i--)
//             behind[i] = behind[i - 1];
//
//         behind[0] = current;
//         current = infront[0];
//         #pragma unroll 4
//
//         for (int i = 0 ; i < RADIUS - 1 ; i++)
//             infront[i] = infront[i + 1];
//
//         if (validr)
//             infront[RADIUS - 1] = bufferSrc[inputIndex];
//
//         inputIndex  += stride_z;
//         outputIndex += stride_z;
//         cg::sync(cta);
//
//         // Note that for the work items on the boundary of the problem, the
//         // supplied index when reading the halo (below) may wrap to the
//         // previous/next row or even the previous/next xy-plane. This is
//         // acceptable since a) we disable the output write for these work
//         // items and b) there is at least one xy-plane before/after the
//         // current plane, so the access will be within bounds.
//
//         // Update the data slice in the local tile
//         // Halo above & below
//         if (ltidy < RADIUS)
//         {
//             tile[ltidy][tx]                  = bufferSrc[outputIndex - RADIUS * stride_y];
//             tile[ltidy + worky + RADIUS][tx] = bufferSrc[outputIndex + worky * stride_y];
//         }
//
//         // Halo left & right
//         if (ltidx < RADIUS)
//         {
//             tile[ty][ltidx]                  = bufferSrc[outputIndex - RADIUS];
//             tile[ty][ltidx + workx + RADIUS] = bufferSrc[outputIndex + workx];
//         }
//
//         tile[ty][tx] = current;
//         cg::sync(cta);
//
//         // Compute the output value
//         float value = stencil[0] * current;
//         #pragma unroll 4
//
//         for (int i = 1 ; i <= RADIUS ; i++)
//         {
//             value += stencil[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
//         }
//
//         // Store the output value
//         if (validw)
//             bufferDst[outputIndex] = value;
//     }
// }

int main(int argc, char * argv[]){
    printf("Running program: %s\n", argv[0]);

    // outfile for debugging
    FILE * outfile;
    outfile = fopen("debug.txt", "w");
    if (outfile == NULL){
      printf(".....there is an error opening debug file....\n");
      return 0;
    }

    // allocate 3D managed memory
    hf::cudaPitchedPtr* PDP;  // pitchedDevPtr
    hf::cudaExtent ext;    // extent
    cudaMalloc3D(PDP, ext);

    printf ("pitch: %d\npointer: %p\nxsize: %d\nysize: %d\n", PDP.pitch, PDP.ptr, PDP.xsize, PDP.ysize);
    printf ("width: %d\nheight: %d\ndepth: %d\n", ext.width, ext.height, ext.depth);




    // // Get the memory size of the target device and save in memsize
    // getTargetDeviceGlobalMemSize(&memsize, argc, argv);
    // memsize /= 2;
    // printf("Memory size: %d\n", memsize);
    //
    // printf(" generateRandomData\n\n");
    // generateRandomData(input, outerDimx, outerDimy, outerDimz, lowerBound, upperBound);
    // printf("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);
    //
    // gpuErrchk(cudaMemcpy(buffer_in + padding, input, volumeSize * sizeof(float), cudaMemcpyDefault));
    // gpuErrchk(cudaMemcpy(buffer_out + padding, input, volumeSize * sizeof(float), cudaMemcpyDefault));
    //
    // // Set up block and grid
    // dim3 dimBlock;
    // dim3 dimGrid;
    // dimBlock.x = 32;
    // dimBlock.y = 16;
    // dimGrid.x = 12;
    // dimGrid.y = 24;
    // // Execute the FDTD
    // float *bufferSrc = buffer_in + padding;
    // float *bufferDst = buffer_out + padding;
    // printf(" GPU FDTD loop\n");
    //
    // clock_t tic = clock();  // start clocking
    //
    // for (int it = 0 ; it < timesteps ; it++){
    //
    //   printf("\tt = %d ", it);
    //
    //   // Launch the kernel
    //   printf("launch kernel\n");
    //   FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc);
    //
    //   float *tmp = bufferDst;
    //   bufferDst = bufferSrc;
    //   bufferSrc = tmp;
    // }
    //
    // clock_t toc = clock() - tic;
    // float elapsed_time = ((float)toc) / CLOCKS_PER_SEC;   // finish clocking
    // printf("Vector addition on the DEVICE\nElapsed time: %f (sec)\n", elapsed_time);
    //
    // for(int i = 0; i < paddedVolumeSize; i+=(141376))
    //   fprintf(outfile, "input[%d] = %f\n", i, bufferDst[i]);

    fclose(outfile);
    return 0;
}
