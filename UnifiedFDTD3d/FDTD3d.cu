// C LIBRARY INCLUDES
#include <cstdio>
#include <cassert>  // useful for debugging
#include <cstdlib>
using namespace std;

// CUDA LIBRARY INCLUDES
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cooperative_groups.h>

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

// INITIALIZE DATA
const int dimx = 376;
const int dimy = 376;
const int dimz = 376;
const int radius = 4;
const int timesteps = 5;
const int outerDimx = dimx + 2 * radius;
const int outerDimy = dimy + 2 * radius;
const int outerDimz = dimz + 2 * radius;
const int volumeSize = outerDimx * outerDimy * outerDimz;
memsize_t memsize;
const float lowerBound = 0.0f;
const float upperBound = 1.0f;

// INITIALIZE UNIFIED MEMORY
__device__ __managed__ float input[volumeSize];

// KERNELS

int main(int argc, char * argv[]){
    printf("Running program: %s\n", argv[0]);

    // outfile for debugging
    FILE * outfile;
    outfile = fopen("debug.txt", "w");
    if (outfile == NULL){
      printf(".....there is an error opening debug file....\n");
      return 0;
    }
    printline("Hello! Welcome to the FDTD3d with unified memory.\n")

    // Get the memory size of the target device and save in memsize
    getTargetDeviceGlobalMemSize(&memsize, argc, argv);
    memsize /= 2;
    printf("Memory size: %d\n", memsize);

    printf(" generateRandomData\n\n");
    generateRandomData(input, outerDimx, outerDimy, outerDimz, lowerBound, upperBound);
    printf("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);

    fclose(outfile);
    return 0;
}
