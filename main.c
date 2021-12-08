#include "ib.h"
#include <bits/types/struct_timeval.h>
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
#include <getopt.h>

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>



#define NAME_LENGTH 128
#define MAX_LEN 32
#define LENGTH 5
#define TCP_PORT    (4211)

#define MEMCOPY_ITERATIONS 25
#define WARMUP_ITERATIONS 3
#define DEFAULT_SIZE (128 * (1e6))      // 32 M
//#define DEFAULT_SIZE (512ULL*1024ULL*1024ULL)      // 512 M
#define DEFAULT_INCREMENT (4 * (1e6))  // 4 M
#define CACHE_CLEAR_SIZE (16 * (1e6))  // 16 M

//#define GPU_TIMING 
#define TIME_INCL_PREPARE 0

#ifdef GPU_TIMING
static cudaEvent_t start, stop;
#else
static struct timeval startt;
#endif

static double times[MEMCOPY_ITERATIONS];

static int no_p2p = 0;

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

int printInfo()
{
    int nDevices;

    if(cudaGetDeviceCount(&nDevices) != cudaSuccess)
    {
        fprintf(stderr, "ERROR: Failed to get device count\n");
    }

    for (int i = 0; i < nDevices; i++) {
        struct cudaDeviceProp prop;

        if(cudaGetDeviceProperties(&prop, i) != cudaSuccess){
            fprintf(stderr, "Failed to get device properties\n");
        }

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

static inline void timer_start(void)
{
#ifdef GPU_TIMING
    if(cudaEventRecord(start, 0) != cudaSuccess){
        fprintf(stderr, "Failed to start gpu timer\n");
    }
#else
    gettimeofday(&startt, NULL);
#endif
}

void timer_stop(void)
{
    static int i = 0;

#ifdef GPU_TIMING
    float elapsedTimeInMs = 0.0f;
    if((cudaEventRecord(stop,0) || cudaDeviceSynchronize() || cudaEventElapsedTime(&elapsedTimeInMs, start, stop)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to stop gpu timer\n");
    }
    times[i] = elapsedTimeInMs / 1e3;
#else
    struct timeval stopt;
    gettimeofday(&stopt, NULL);
    times[i] = (stopt.tv_sec - startt.tv_sec) * 1e6;
    times[i] = (times[i] + (stopt.tv_usec - startt.tv_usec)) * 1e-6;
#endif
    if (i < MEMCOPY_ITERATIONS) {
        i++;
    }
}

void print_times(enum print_flags flags, size_t memSize, char * type)
{
    double val;
    double avg;
    printf("-----------------------------------------------\n");
    printf("%s transfer:\n", type);
    for (int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        avg += times[i];
        if (flags & INDVAL) {
            printf("%d: %f\n", i, times[i]);
        }
    }
    printf("-----------------------------------------------\n");
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
    printf("-----------------------------------------------\n");

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
    ib_init_oob_listener(TCP_PORT);
    //ib_server_prepare(d_odata, 1, memSize, true);

    printf("receiving...\n");
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS+WARMUP_ITERATIONS; i++)
    {
        ib_server_prepare(d_odata, 1, memSize, true);
        ib_msg_recv(memSize, 1);
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
        ib_client_prepare(d_idata, 1, memSize, peer_node, true);
        ib_msg_send();
    }
    // copy data from GPU to Host
    printf("sending...\n");
    
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        if (TIME_INCL_PREPARE) timer_start();
        ib_client_prepare(d_idata, 1, memSize, peer_node, true);
        if (!TIME_INCL_PREPARE) timer_start();
        ib_msg_send();
        timer_stop();
    }
    printf("finished.\n");

    print_times(ALL, memSize, "Device to Host");

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
    ib_init_oob_sender(peer_node, TCP_PORT);
    //ib_client_prepare(h_idata, 1, memSize, peer_node, false);
    
    printf("warming up...\n");
    for (unsigned int i = 0; i < WARMUP_ITERATIONS; i++)
    {
        ib_client_prepare(h_idata, 1, memSize, peer_node, false);
        ib_msg_send();
    }

    // copy data from GPU to Host
    printf("sending...\n");

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        if (TIME_INCL_PREPARE) timer_start();
        ib_client_prepare(h_idata, 1, memSize, peer_node, false);
        if (!TIME_INCL_PREPARE) timer_start();
        ib_msg_send();
        timer_stop();
    }

    print_times(ALL, memSize, "Host To Device");

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
        ib_server_prepare(h_odata, 1, memSize, false);
        ib_msg_recv(memSize, 1);
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


    while (1)
    {
        static struct option long_options[] =
        {
          {"nop2p", no_argument,  &no_p2p, 1},
          {0, 0, 0, 0}
        };

        int option_index = 0;

        arg = getopt_long(argc, argv, "hscd:g:p:", long_options, &option_index);

        if (arg == -1){
            break;
        } 

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
                   "-s/-c -p <peer_node>\n -d   IB device ID (default 0)\n -g   GPU ID (default 0)\n --nop2p    disable peer to peer", argv[0]);
            break;
        case '?': 
            printf("unknown option: %c\n", optopt);
            break;
        }
    }

    if(no_p2p)
    {
        printf("option no p2p has been set\n");
    }

    srand48(getpid() * time(NULL));

#ifdef GPU_TIMING
    if((cudaSetDevice(gpu_id) || cudaEventCreate(&start) || cudaEventCreate(&stop)) != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start and/or stop event\n"); 
        exit(1);   
    }
#endif

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

#ifdef GPU_TIMING
        printf("using GPU timer...\n");
#endif

        testBandwidthServer(DEFAULT_SIZE, peer_node);

        printf("-----------------------------------------------\n");

        free(flush_buf);    

    }
    else
    { //client

//        printInfo();

        flush_buf = (char *)malloc(FLUSH_SIZE);

        printf("-----------------------------------------------\n");

#ifdef GPU_TIMING
        printf("using GPU timer...\n");
#endif

        testBandwidthClient(DEFAULT_SIZE, peer_node);

        printf("-----------------------------------------------\n");

        free(flush_buf);    

    }

#ifdef GPU_TIMING
    if((cudaEventDestroy(start) || cudaEventDestroy(stop)) != cudaSuccess){
        fprintf(stderr, "Failed to destroy gpu stimer\n");
    }
#endif

    ib_cleanup();
    ib_final_cleanup();

    return 0;

}
