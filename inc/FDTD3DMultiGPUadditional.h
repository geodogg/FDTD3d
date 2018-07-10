#ifndef _FDTD3DMULTIGPUADDITIONAL_H_
#define _FDTD3DMULTIGPUADDITIONAL_H_

#include <cuda_runtime.h>
////////////////////////////////////////////////////////////////////////////////
// Structure for Device Properties.
////////////////////////////////////////////////////////////////////////////////
typedef struct
{
    int device;            // device ID
    int data_size;         // bytes of data to be processed on device
    float *d_data;         // pointer to device data
    float *h_data;         // pointer to host location of data
    float *u_data;         // pointer to unified d_data
    dim3 dimBlock;
    dim3 dimGrid;
    cudaDeviceProp deviceProp;  // cuda device properties

} DEVICES;

// function to initialize the argument GPU
void initGPU(DEVICES *device);

#endif
