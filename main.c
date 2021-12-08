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
#define DEFAULT_TCP_PORT    (4211)

#define DEFAULT_MEMCOPY_ITERATIONS 25
#define DEFAULT_WARMUP_ITERATIONS 3
#define DEFAULT_SIZE (128 * (1e6))      // 32 M

//#define DEFAULT_SIZE (512ULL*1024ULL*1024ULL)      // 512 M
//#define GPU_TIMING 

#define TIME_INCL_PREPARE 0

#ifdef GPU_TIMING
static cudaEvent_t start, stop;
#else
static struct timeval startt;
#endif

static int no_p2p = 0;
static int extended_output = 0;
static int short_output = 0;
static int tcp_port = DEFAULT_TCP_PORT;
static int memcopy_iterations = DEFAULT_MEMCOPY_ITERATIONS;
static int warmup_iterations = DEFAULT_WARMUP_ITERATIONS;


// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

int printInfo()
{
    printf("-----------------------------------------------\n");
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
     printf("-----------------------------------------------\n");

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

void timer_stop(double* times)
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
    if (i < memcopy_iterations) {
        i++;
    }
}

void print_times(enum print_flags flags, size_t memSize, char * type, double* times)
{
    double val;
    double avg;
    printf("-----------------------------------------------\n");
    printf("%s transfer:\n", type);
    for (int i = 0; i < memcopy_iterations; i++)
    {
        avg += times[i];
        if (!short_output && flags && INDVAL) {
            printf("%d: %f\n", i, times[i]);
        }
    }
    if(!short_output) printf("-----------------------------------------------\n");
    avg /= memcopy_iterations;
    if (flags & AVG) {
        printf("Average Time: %f s\n", avg);
    }
    if (flags & STDDEV) {
        val = 0;
        for (int i = 0; i < memcopy_iterations; i++)
        {
            val += (times[i]-avg)*(times[i]-avg);
        }
        val /= memcopy_iterations;
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

void testBandwidthServer(size_t memSize, char* peer_node, double* times)
{

    // ...... Host to Device
    double bandwidthInGBs = 0.0;
    double bandwidthInGiBs = 0.0;

    void *d_odata;
    void *h_odata;

    ib_allocate_memreg(&d_odata, memSize, 1, true);

    // copy data from GPU to Host

    if(extended_output) printf("preparing server...\n");
    ib_init_oob_listener(tcp_port);
    //ib_server_prepare(d_odata, 1, memSize, true);

    if(extended_output) printf("receiving...\n");

    if (no_p2p)
    {
        ib_allocate_memreg(&h_odata, memSize, 0, false);
        for (unsigned int i = 0; i < memcopy_iterations + warmup_iterations; i++)
        {
            ib_server_prepare(h_odata, 0, memSize, false);
            ib_msg_recv(memSize, 0);
            cudaMemcpy(d_odata, h_odata, memSize, cudaMemcpyHostToDevice);
        }
        ib_free_memreg(h_odata, 0, false);
    }
    else
    {
        for (unsigned int i = 0; i < memcopy_iterations + warmup_iterations; i++)
        {
            ib_server_prepare(d_odata, 1, memSize, true);
            ib_msg_recv(memSize, 1);
        }
    }

    if(extended_output) printf("finished. cleaning up...\n");
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

    if(extended_output) printf("preparing client...\n");
    //ib_client_prepare(d_idata, 1, memSize, peer_node, true);

    if(extended_output) printf("warming up...\n");

    if (no_p2p)
    {
        for (unsigned int i = 0; i < warmup_iterations; i++)
        {   
            ib_client_prepare(h_idata, 0, memSize, peer_node, false);
            cudaMemcpy(h_idata, d_idata, memSize, cudaMemcpyDeviceToHost);
            ib_msg_send();
        }
        // copy data from GPU to Host
        if(extended_output) printf("sending...\n");

        for (unsigned int i = 0; i < memcopy_iterations; i++)
        {
            if (TIME_INCL_PREPARE)
                timer_start();
            ib_client_prepare(h_idata, 0, memSize, peer_node, false);
            if (!TIME_INCL_PREPARE)
                timer_start();
            cudaMemcpy(h_idata, d_idata, memSize, cudaMemcpyDeviceToHost);
            ib_msg_send();
            timer_stop(times);
        }
    }
    else
    {
        for (unsigned int i = 0; i < warmup_iterations; i++)
        {
            ib_client_prepare(d_idata, 1, memSize, peer_node, true);
            ib_msg_send();
        }
        // copy data from GPU to Host
        if(extended_output) printf("sending...\n");

        for (unsigned int i = 0; i < memcopy_iterations; i++)
        {
            if (TIME_INCL_PREPARE)
                timer_start();
            ib_client_prepare(d_idata, 1, memSize, peer_node, true);
            if (!TIME_INCL_PREPARE)
                timer_start();
            ib_msg_send();
            timer_stop(times);
        }
    }
    if(extended_output) printf("finished.\n");

    print_times(ALL, memSize, "Device to Host", times);

    // clean up memory

    ib_free_memreg(h_idata, 0, false);
    ib_free_memreg(d_idata, 1, true);
}

void testBandwidthClient(size_t memSize, char* peer_node, double* times)
{

    //    Host to Device ............
    double bandwidthInGBs = 0.0;
    double bandwidthInGiBs = 0.0;
    void *h_idata;

    struct timeval start, end;

    // allocate memory
    ib_allocate_memreg(&h_idata, memSize, 1, false);
    memset(h_idata, 1, memSize);

    if(extended_output) printf("preparing client...\n");
    ib_init_oob_sender(peer_node, tcp_port);
    //ib_client_prepare(h_idata, 1, memSize, peer_node, false);
    
    if(extended_output) printf("warming up...\n");
    for (unsigned int i = 0; i < warmup_iterations; i++)
    {
        ib_client_prepare(h_idata, 1, memSize, peer_node, false);
        ib_msg_send();
    }

    // copy data from GPU to Host
    if(extended_output) printf("sending...\n");

    for (unsigned int i = 0; i < memcopy_iterations; i++)
    {
        if (TIME_INCL_PREPARE) timer_start();
        ib_client_prepare(h_idata, 1, memSize, peer_node, false);
        if (!TIME_INCL_PREPARE) timer_start();
        ib_msg_send();
        timer_stop(times);
    }

    print_times(ALL, memSize, "Host To Device", times);

    if(extended_output) printf("finished. cleaning up...\n");

    // clean up memory

    ib_free_memreg(h_idata, 1, false);

    //..... Device to Host

    void *h_odata = NULL;

    ib_allocate_memreg(&h_odata, memSize, 1, false);

    if(extended_output) printf("preparing server...\n");
    //ib_server_prepare(h_odata, 1, memSize, false);

    // copy data from GPU to Host

    if(extended_output) printf("receving...\n");
    for (unsigned int i = 0; i < memcopy_iterations + warmup_iterations; i++)
    {
        ib_server_prepare(h_odata, 1, memSize, false);
        ib_msg_recv(memSize, 1);
    }

    if(extended_output) printf("finished. cleaning up...\n");
    ib_free_memreg(h_odata, 1, false);
}

void print_variables(int server, char* peer_node, int ib_device_id, int gpu_id, int mem_size, int iterations, int warmup, int tcp, int nop2p){
    printf("-----------------------------------------------\n");
    server ? printf("Peer (Client): %s\n", peer_node) : printf("Peer (Server): %s\n", peer_node);
    printf("InfiniBand device: %d\n", ib_device_id);
    printf("TCP port: %d\n", tcp);
    if(server) printf("GPU: %d\n", gpu_id);
    printf("Memory size: %d\n", mem_size);
    printf("Iterations: %d\n", iterations);
    printf("Warmup iterations: %d\n", warmup);
    if(nop2p) printf("Using NO peer to peer\n");
#ifdef GPU_TIMING
    printf("Using GPU timer\n");
#endif
}

int main(int argc, char **argv)
{
    int arg             = 0;
    int32_t server          = -1;
    char peer_node[NAME_LENGTH]     = { [0 ... NAME_LENGTH-1] = 0 };
    int device_id_param = 0;
    int gpu_id = 0;
    int mem_size = DEFAULT_SIZE;


    while (1)
    {
        static struct option long_options[] =
        {
          {"nop2p", no_argument,  &no_p2p, 1},
          {"extended", no_argument,  &extended_output, 1},
          {"short", no_argument,  &short_output, 1},
          {0, 0, 0, 0}
        };

        int option_index = 0;

        arg = getopt_long(argc, argv, "hscw:i:m:t:d:g:p:", long_options, &option_index);

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
        case 'm':
            mem_size = atoi(optarg);
            break;
        case 't':
            tcp_port = atoi(optarg);
            break;
        case 'w':
            warmup_iterations = atoi(optarg);
            break;
        case 'i':
            memcopy_iterations = atoi(optarg);
            break;
        case 'h':
            printf("Command line parameters:\n"
                   " -s/-c         server/client (required)\n -p            <peer node> (required)\n -d            <IB device ID> (default 0)\n -g            <GPU ID> (default 0)\n"
                    " -m            <memory size> (default: %d)\n -i            <memcopy iterations> (default: %d)\n "
                    "-w            <warmup iterations> (default: %d)\n -t            <TCP port> (default: %d)\n"
                    " --nop2p       disable peer to peer (flag)\n --extended    extended terminal output (flag)\n --short       short terminal output (flag)\n",
                     DEFAULT_SIZE, DEFAULT_MEMCOPY_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, DEFAULT_TCP_PORT);
            exit(0);
            break;
        case '?': 
            printf("unknown option: %c\n", optopt);
            break;
        }
    }

    if(extended_output)
    {
        print_variables(server, peer_node, device_id_param, gpu_id, mem_size, memcopy_iterations, warmup_iterations, tcp_port, no_p2p);
    }

    double times[memcopy_iterations];

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

    void *memptr0, *memptr1, *memptr2; 
    void *gpumemptr0, *gpumemptr1, *gpumemptr2;

    void * h_odata, * h_idata;
    void * d_idata;


    ib_init(device_id_param);
    if (server) {

        if(extended_output){
        printInfo();
        }

        cudaSetDevice(gpu_id);


        flush_buf = (char *)malloc(FLUSH_SIZE);


        testBandwidthServer(mem_size, peer_node, times);

        free(flush_buf);    

    }
    else
    { //client

//        printInfo();

        flush_buf = (char *)malloc(FLUSH_SIZE);

        testBandwidthClient(mem_size, peer_node, times);


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
