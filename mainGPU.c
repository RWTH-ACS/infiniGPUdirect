#include "ib.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>



#define NAME_LENGTH 128
#define MAX_LEN 32
#define LENGTH 5

#define MEMCOPY_ITERATIONS 100
#define DEFAULT_SIZE (32 * (1e6))      // 32 M
#define DEFAULT_INCREMENT (4 * (1e6))  // 4 M
#define CACHE_CLEAR_SIZE (16 * (1e6))  // 16 M

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

void testBandwidthServer(unsigned int memSize, char *peer_node)
{

    // ...... Host to Device

    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;

    void *d_odata;
    // allocate memory
    ib_allocate_memreg(&d_odata, memSize, 1, true);

    // copy data from Host to GPU

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_server_recv(d_odata, 1, memSize, true);
    }

    // clean up memory
    ib_free_memreg(d_odata, 1, true);

    //........Device to Host

    void *h_idata;
    void *d_idata;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ib_allocate_memreg(&h_idata, memSize, 0, false);
    ib_allocate_memreg(&d_idata, memSize, 1, true);

    memset(h_idata, 1, memSize);

    // initialize the device memory
    cudaMemcpy(d_idata, h_idata, memSize, cudaMemcpyHostToDevice);

    // copy data from GPU to Host
    cudaEventRecord(start, 0);

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_client_send(d_idata, 1, memSize, peer_node, true);
    }

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

    // calculate bandwidth in GB/s
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    // clean up memory

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    ib_free_memreg(h_idata, 0, false);
    ib_free_memreg(d_idata, 1, true);

    printf("Bandwidth:\n");
    printf("Device to Host (Server): %f GB/s\n", bandwidthInGBs);
}

void testBandwidthClient(unsigned int memSize, char *peer_node)
{

    //    Host to Device ............#############################################
    float elapsedTimeInMs = 0.0f;
    float bandwidthInGBs = 0.0f;
    void *h_idata;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate memory
    ib_allocate_memreg(&h_idata, memSize, 1, false);
    memset(h_idata, 1, memSize);

    // copy data from Host to GPU
    cudaEventRecord(start, 0);

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_client_send(h_idata, 1, memSize, peer_node, false);
    }

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTimeInMs, start, stop);

    // calculate bandwidth in GB/s
    double time_s = elapsedTimeInMs / 1e3;
    bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    // clean up memory

    cudaEventDestroy(stop);
    cudaEventDestroy(start);

    ib_free_memreg(h_idata, 1, false);

    printf("Bandwidth:\n");
    printf("Host to Device (Client): %f GB/s\n", bandwidthInGBs);

    //..... Device to Host

    void *h_odata = NULL;

    // allocate memory
    ib_allocate_memreg(&h_odata, memSize, 1, false);

    // copy data from GPU to Host

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_server_recv(h_odata, 1, memSize, false);
    }

    // clean up memory
    ib_free_memreg(h_odata, 1, false);
}

int main(int argc, char **argv)
{
    int arg             = 0;
    int32_t server          = -1;
    char peer_node[NAME_LENGTH]     = { [0 ... NAME_LENGTH-1] = 0 };
    int device_id_param = 0;
    int gpu_id = 0;

    while ((arg = getopt(argc, argv, "hscd:g:p:")) != -1) {
        switch (arg) {
        case 's':
            server = 1;
            break;
        case 'c':
            server = 0;
            break;
        case 'p':
            strcpy(peer_node, optarg);
            break;
        case 'd':
            device_id_param = atoi(optarg);
            break;
        case 'g':
            gpu_id = atoi(optarg);
            break;
        case 'h':
            printf("usage: %s "
                   "-s/-c -p <peer_node>\n -d   <IB device ID> (default 0)\n -g   <GPU ID> (default 0)\n", argv[0]);
                   return 0;
            break;

        }
    }

    srand48(getpid() * time(NULL));

    if (server == -1) {
        fprintf(stderr, "ERROR: You must either specify the "inf
                "-s or -c argument. Abort!\n");
        exit(-1);
    } else if (!server && (strlen(peer_node) == 0)) {
        fprintf(stderr, "ERROR: The client must specify the peer_node "
                "argument. Abort!\n");
        exit(-1);
    }

    size_t size = LENGTH * sizeof(int);

    void *memptr0, *memptr1, *memptr2; 
    void *gpumemptr0, *gpumemptr1, *gpumemptr2;

    void * h_odata, * h_idata;
    void * d_idata;

    printf("-----------------------------------------------\n");

    ib_init(device_id_param);
    if (server) {

        printf("-----------------------------------------------\n");
        printInfo();
        printf("-----------------------------------------------\n");


        cudaSetDevice(gpu_id);

        printf("[INFO] Using GPU: %d\n", gpu_id);

        printf("-----------------------------------------------\n");

        flush_buf = (char *)malloc(FLUSH_SIZE);

        testBandwidthServer(DEFAULT_SIZE, peer_node);

        printf("-----------------------------------------------\n");

        free(flush_buf);    

    }
    else
    { //client

//        printInfo();

        cudaSetDevice(1);

        flush_buf = (char *)malloc(FLUSH_SIZE);

        printf("-----------------------------------------------\n");

        testBandwidthClient(DEFAULT_SIZE, peer_node);

        printf("-----------------------------------------------\n");

        free(flush_buf);    

    }

    ib_cleanup();
    ib_final_cleanup();

    return 0;

}
