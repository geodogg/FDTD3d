/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
#include "FDTD3d.h"
#include "FDTD3dGPU.h"
#include "FDTD3dReference.h"

#include <iostream>
#include <iomanip>
#include <math.h>
#include <assert.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#define printline(ans) { fprintf(outfile, "file: %s line: %d\n - ", __FILE__, __LINE__); fprintf(outfile, ans); }

#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif

//// Name of the log file
//const char *printfFile = "FDTD3d.txt";

// Forward declarations
bool runTest(int argc, const char **argv);
void showHelp(const int argc, const char **argv);

int main(int argc, char **argv)
{
    bool bTestResult = false;
    // Start the log
    printf("%s Starting...\n\n", argv[0]);

    // Check help flag
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
        bTestResult = true;
    }
    else
    {
        // Execute
        bTestResult = runTest(argc, (const char **)argv);
    }

    // Finish
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void showHelp(const int argc, const char **argv)
{
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;

    std::cout << std::endl << "Syntax:" << std::endl;
    std::cout << std::left;
    std::cout << "    " << std::setw(20) << "--device=<device>" << "Specify device to use for execution" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimx=<N>" << "Specify number of elements in x direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimy=<N>" << "Specify number of elements in y direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimz=<N>" << "Specify number of elements in z direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--radius=<N>" << "Specify radius of stencil" << std::endl;
    std::cout << "    " << std::setw(20) << "--timesteps=<N>" << "Specify number of timesteps" << std::endl;
    std::cout << "    " << std::setw(20) << "--block-size=<N>" << "Specify number of threads per block" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(20) << "--noprompt" << "Skip prompt before exit" << std::endl;
    std::cout << std::endl;
}

bool runTest(int argc, const char **argv)
{

    // outfile for debugging
    FILE * outfile;
    outfile = fopen("debug.txt", "w");
    if (outfile == NULL){
      printf(".....there is an error opening debug file....\n");
      return 0;
    }
    printline("GOOD\n")

    float *host_output;
    float *device_output;
    float *input;
    float *coeff;

    int defaultDim;
    int dimx;
    int dimy;
    int dimz;
    int outerDimx;
    int outerDimy;
    int outerDimz;
    int radius;
    int timesteps;
    size_t volumeSize;
    memsize_t memsize;

    const float lowerBound = 0.0f;
    const float upperBound = 1.0f;

    // Determine default dimensions
    printf("Set-up, based upon target device GMEM size...\n");
    // Get the memory size of the target device
    printf(" getTargetDeviceGlobalMemSize\n");
    getTargetDeviceGlobalMemSize(&memsize, argc, argv);

    printline("GOOD\n")

    // We can never use all the memory so to keep things simple we aim to
    // use around half the total memory
    memsize /= 2;

    // Most of our memory use is taken up by the input and output buffers -
    // two buffers of equal size - and for simplicity the volume is a cube:
    //   dim = floor( (N/2)^(1/3) )
    defaultDim = (int)floor(pow((memsize / (2.0 * sizeof(float))), 1.0/3.0));

    // By default, make the volume edge size an integer multiple of 128B to
    // improve performance by coalescing memory accesses, in a real
    // application it would make sense to pad the lines accordingly
    int roundTarget = 128 / sizeof(float);
    defaultDim = defaultDim / roundTarget * roundTarget;
    defaultDim -= k_radius_default * 2;

    // Check dimension is valid
    if (defaultDim < k_dim_min)
    {
        printf("insufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
        exit(EXIT_FAILURE);
    }
    else if (defaultDim > k_dim_max)
    {
        defaultDim = k_dim_max;
    }

    // For QA testing, override default volume size
    if (checkCmdLineFlag(argc, argv, "qatest"))
    {
        defaultDim = MIN(defaultDim, k_dim_qa);
    }

    //set default dim
    dimx = defaultDim;
    dimy = defaultDim;
    dimz = defaultDim;
    radius    = k_radius_default;
    timesteps = k_timesteps_default;

    // Parse command line arguments
    if (checkCmdLineFlag(argc, argv, "dimx"))
    {
        dimx = CLAMP(getCmdLineArgumentInt(argc, argv, "dimx"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimy"))
    {
        dimy = CLAMP(getCmdLineArgumentInt(argc, argv, "dimy"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimz"))
    {
        dimz = CLAMP(getCmdLineArgumentInt(argc, argv, "dimz"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "radius"))
    {
        radius = CLAMP(getCmdLineArgumentInt(argc, argv, "radius"), k_radius_min, k_radius_max);
    }

    if (checkCmdLineFlag(argc, argv, "timesteps"))
    {
        timesteps = CLAMP(getCmdLineArgumentInt(argc, argv, "timesteps"), k_timesteps_min, k_timesteps_max);
    }

    ////////////////////////////////////////////////////////////////////////////
    //~~~~~~~~~~~~~~~~~~~~~~~~~~!!! UPDATED HERE !!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ////////////////////////////////////////////////////////////////////////////

    printline("GOOD\n")

    int num_devices = 0;
    checkCudaErrors(cudaGetDeviceCount(&num_devices));

    printf("Number of devices %d\n",num_devices);

    printline("GOOD\n")

    // Initialize an array of devices
    DEVICES *arr_device = new DEVICES[num_devices];

    // allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(num_devices * sizeof(cudaStream_t));
    cudaEvent_t *events = (cudaEvent_t *) malloc(num_devices * sizeof(cudaEvent_t));

    printline("GOOD\n")

    for (int i = 0; i < num_devices; i++)
    {
        arr_device[i].device = i;
        arr_device[i].num_devices = num_devices;
        checkCudaErrors(cudaSetDevice(arr_device[i].device));
        checkCudaErrors(cudaStreamCreate(&(streams[i])));
        checkCudaErrors(cudaEventCreate(&(events[i])));
        checkCudaErrors(cudaGetDeviceProperties(&(arr_device[i].deviceProp), arr_device[i].device));

        // Allocate intermediate memory for MC integrator
        // and initialize RNG state

    }

    ////////////////////////////////////////////////////////////////////////////
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ////////////////////////////////////////////////////////////////////////////

    // Determine volume size
    outerDimx = dimx + 2 * radius;
    outerDimy = dimy + 2 * radius;
    outerDimz = dimz + 2 * radius;
    volumeSize = outerDimx * outerDimy * outerDimz;

    // Allocate memory
    host_output = (float *)calloc(volumeSize, sizeof(float));
    input       = (float *)malloc(volumeSize * sizeof(float));
    coeff       = (float *)malloc((radius + 1) * sizeof(float));

    // Create coefficients
    for (int i = 0 ; i <= radius ; i++)
    {
        coeff[i] = 0.1f;
    }

    // Generate data
    printf(" generateRandomData\n\n");
    generateRandomData(input, outerDimx, outerDimy, outerDimz, lowerBound, upperBound);
    printf("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);

    // Execute on the host
    printf("fdtdReference...\n");
    fdtdReference(host_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
    printf("fdtdReference complete\n");

    // Allocate memory
    device_output = (float *)calloc(volumeSize, sizeof(float));

    // Execute on the device
    printf("fdtdGPU...\n");
    fdtdGPU(arr_device, device_output, input, coeff, dimx, dimy, dimz, radius, timesteps, argc, argv);
    printf("fdtdGPU complete\n");

    // Compare the results
    float tolerance = 0.0001f;
    printf("\nCompareData (tolerance %f)...\n", tolerance);
    return compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);


    delete[] arr_device;

}
