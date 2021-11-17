#include "ib.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>



#define NAME_LENGTH 128
#define MAX_LEN 32
#define LENGTH 5

#define MEMCOPY_ITERATIONS 25
#define WARMUP_ITERATIONS 3
//#define DEFAULT_SIZE (32 * (1e6))      // 32 M
#define DEFAULT_SIZE (512ULL*1024ULL*1024ULL)      // 512 M
#define DEFAULT_INCREMENT (4 * (1e6))  // 4 M
#define CACHE_CLEAR_SIZE (16 * (1e6))  // 16 M
struct timeval meas[MEMCOPY_ITERATIONS+1];

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

int printInfo()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
     }

    return 0;
}

enum print_flags {
    GB = 1,      // Bandwidth in GB/s
    GIB = 2,     // Bandwidth in GiB/s
    STDDEV = 4,  // Std. Deviation
    INDVAL = 8,  // Each individual value
    AVG = 16,    // Average time
    ALL = 0xFF,  // All above
};

void print_times(enum print_flags flags, size_t memSize)
{
    double times[MEMCOPY_ITERATIONS];
    double val;
    double avg;
    for (int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        struct timeval *start = &meas[i];
        struct timeval *end = &meas[i+1];
        times[i] = (end->tv_sec - start->tv_sec) * 1e6;
        times[i] = (times[i] + (end->tv_usec - start->tv_usec)) * 1e-6;
        avg += times[i];
        if (flags & INDVAL) {
            printf("%d: %f\n", i, times[i]);
        }
    }
    avg /= MEMCOPY_ITERATIONS;
    if (flags & AVG) {
        printf("Average Time: %f s\n", avg);
    }
    if (flags & STDDEV) {
        val = 0;
        for (int i = 0; i < MEMCOPY_ITERATIONS; i++)
        {
            val += (times[i]-avg)*(times[i]-avg);
        }
        val /= MEMCOPY_ITERATIONS;
        val = sqrt(val);
        printf("Std. Deviation %f s\n", val);
    }
    // calculate bandwidth in GB/s
    if (flags & GB) {
        val = (double)(memSize / (1000ULL*1000ULL)) / 1000.;
        val = val / avg;
        printf("Bandwidth: %f GB/s\n", val);
    }
    // calculate bandwidth in GiB/s
    if (flags & GIB) {
        val = (double)(memSize / (1024ULL*1024ULL)) / 1024.;
        val = val / avg;
        printf("Bandwidth %f GiB/s\n", val);
    }
}


void testBandwidthServer(size_t memSize, char *peer_node)
{

    // ...... Host to Device
    double bandwidthInGBs = 0.0;
    double bandwidthInGiBs = 0.0;

    void *d_odata;
    ib_allocate_memreg(&d_odata, memSize, 1, true);

    // copy data from GPU to Host

    printf("preparing server...\n");
    //ib_server_prepare(d_odata, 1, memSize, true);

    printf("receiving...\n");
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS+WARMUP_ITERATIONS; i++)
    {
        ib_server_recv(d_odata, 1, memSize, true);
    }
    printf("finished. cleaning up...\n");
    ib_free_memreg(d_odata, 1, true);

    //........Device to Host

    void *h_idata;
    void *d_idata;

    struct timeval start, end;

    // allocate memory
    ib_allocate_memreg(&h_idata, memSize, 0, false);
    ib_allocate_memreg(&d_idata, memSize, 1, true);

    memset(h_idata, 1, memSize);

    // initialize the device memory
    cudaMemcpy(d_idata, h_idata, memSize, cudaMemcpyHostToDevice);

    printf("preparing client...\n");
    //ib_client_prepare(d_idata, 1, memSize, peer_node, true);

    printf("warming up...\n");
    for (unsigned int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        ib_client_send(d_idata, 1, memSize, peer_node, true);
    }
    // copy data from GPU to Host
    printf("sending...\n");
    gettimeofday(&meas[0], NULL);
    
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_client_send(d_idata, 1, memSize, peer_node, true);
        gettimeofday(&meas[i+1], NULL);
    }
    printf("finished.\n");

    print_times(ALL, memSize);

    // clean up memory

    ib_free_memreg(h_idata, 0, false);
    ib_free_memreg(d_idata, 1, true);

}

void testBandwidthClient(size_t memSize, char *peer_node)
{

    //    Host to Device ............
    double bandwidthInGBs = 0.0;
    double bandwidthInGiBs = 0.0;
    void *h_idata;

    struct timeval start, end;

    // allocate memory
    ib_allocate_memreg(&h_idata, memSize, 1, false);
    memset(h_idata, 1, memSize);

    printf("preparing client...\n");
    //ib_client_prepare(h_idata, 1, memSize, peer_node, false);
    
    printf("warming up...\n");
    for (unsigned int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        ib_client_send(h_idata, 1, memSize, peer_node, false);
    }

    // copy data from GPU to Host
    printf("sending...\n");

    gettimeofday(&meas[0], NULL);
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_client_send(h_idata, 1, memSize, peer_node, false);
        gettimeofday(&meas[i+1], NULL);
    }

    print_times(ALL, memSize);

    printf("finished. cleaning up...\n");

    // clean up memory

    ib_free_memreg(h_idata, 1, false);

    //..... Device to Host

    void *h_odata = NULL;

    ib_allocate_memreg(&h_odata, memSize, 1, false);

    printf("preparing server...\n");
    //ib_server_prepare(h_odata, 1, memSize, false);

    // copy data from GPU to Host

    printf("receving...\n");
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS+WARMUP_ITERATIONS; i++)
    {
        ib_server_recv(h_odata, 1, memSize, false);
    }

    printf("finished. cleaning up...\n");
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
            printf("usage ./%s "
                   "-s/-c -p <peer_node>\n -d   IB device ID (default 0)\n -g   GPU ID (default 0)\n", argv[0]);
            break;

        }
    }

    srand48(getpid() * time(NULL));

    if (server == -1) {
        fprintf(stderr, "ERROR: You must either specify the "
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

        printf("Using GPU: %d\n", gpu_id);

        printf("-----------------------------------------------\n");

        flush_buf = (char *)malloc(FLUSH_SIZE);

        testBandwidthServer(DEFAULT_SIZE, peer_node);

        printf("-----------------------------------------------\n");

        free(flush_buf);    

    }
    else
    { //client

//        printInfo();

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
