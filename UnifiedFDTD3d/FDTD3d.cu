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

// INITIALIZE UNIFIED MEMORY

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
    memsize_t memsize;
    getTargetDeviceGlobalMemSize(&memsize, argc, argv);
    memsize /= 2;
    printf("Memory size: %d\n", memsize);

    fclose(outfile);
    return 0;
}
