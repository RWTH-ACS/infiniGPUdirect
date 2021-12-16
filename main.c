#include "ib.h"
#include "output.h"
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

static int no_p2p = 0;
static int extended_output = 0;
static int short_output = 0;
static int time_incl_prepare = 0;
static int sysmem_only = 0;

static int tcp_port = DEFAULT_TCP_PORT;
static int memcopy_iterations = DEFAULT_MEMCOPY_ITERATIONS;
static int warmup_iterations = DEFAULT_WARMUP_ITERATIONS;

#ifdef GPU_TIMING
static cudaEvent_t start, stop;
#else
static struct timeval startt;
#endif

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;


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

    if(extended_output) printf("receiving...\n");

    if(sysmem_only)
    {
        ib_allocate_memreg(&h_odata, memSize, 0, false);
        ib_connect_responder(h_odata, 0);
        for (unsigned int i = 0; i < memcopy_iterations + warmup_iterations; i++)
        {
            ib_msg_recv(memSize, 0);
        }
        ib_free_memreg(h_odata, 0, false);
    }
    else if(no_p2p)
    {
        //does not work as intended?
        ib_allocate_memreg(&h_odata, memSize, 0, false);
        ib_connect_responder(h_odata, 0);
        for (unsigned int i = 0; i < memcopy_iterations + warmup_iterations; i++)
        {
            ib_msg_recv(memSize, 0);
            cudaMemcpy(d_odata, h_odata, memSize, cudaMemcpyHostToDevice);
        }
        ib_free_memreg(h_odata, 0, false);
    }
    else
    {
        ib_connect_responder(d_odata, 1);
        for (unsigned int i = 0; i < memcopy_iterations + warmup_iterations; i++)
        {
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

    if(extended_output) printf("warming up...\n");

    if(sysmem_only)
    {
        ib_connect_requester(h_idata, 0, peer_node);
        for (unsigned int i = 0; i < warmup_iterations; i++)
        {   
            ib_msg_send(h_idata, 0, memSize, false);
        }
        // copy data from GPU to Host
        if(extended_output) printf("sending...\n");

        for (unsigned int i = 0; i < memcopy_iterations; i++)
        {
            timer_start();
            ib_msg_send(h_idata, 0, memSize, false);
            timer_stop(times);
        }
    }
    else if(no_p2p)
    {
        ib_connect_requester(h_idata, 0, peer_node);
        for (unsigned int i = 0; i < warmup_iterations; i++)
        {
            cudaMemcpy(h_idata, d_idata, memSize, cudaMemcpyDeviceToHost);
            ib_msg_send(h_idata, 0, memSize, false);
        }
        // copy data from GPU to Host
        if(extended_output) printf("sending...\n");

        for (unsigned int i = 0; i < memcopy_iterations; i++)
        {
            timer_start();
            cudaMemcpy(h_idata, d_idata, memSize, cudaMemcpyDeviceToHost);
            ib_msg_send(h_idata, 0, memSize, false);
            timer_stop(times);
        }
    }
    else
    {
        ib_connect_requester(d_idata, 1, peer_node);
        for (unsigned int i = 0; i < warmup_iterations; i++)
        {
            ib_msg_send(d_idata, 1, memSize, true);
        }
        // copy data from GPU to Host
        if(extended_output) printf("sending...\n");

        for (unsigned int i = 0; i < memcopy_iterations; i++)
        {
            timer_start();
            ib_msg_send(d_idata, 1, memSize, true);
            timer_stop(times);
        }
    }
    if(extended_output) printf("finished.\n");

    print_times(ALL, memSize, "Device to Host", times, memcopy_iterations, short_output);

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
    
    if(extended_output) printf("warming up...\n");

    ib_connect_requester(h_idata, 1, peer_node);
    for (unsigned int i = 0; i < warmup_iterations; i++)
    {
        ib_msg_send(h_idata, 1, memSize, false);
    }

    // copy data from GPU to Host
    if(extended_output) printf("sending...\n");

    for (unsigned int i = 0; i < memcopy_iterations; i++)
    {
        timer_start();
        ib_msg_send(h_idata, 1, memSize, false);
        timer_stop(times);
    }

    print_times(ALL, memSize, "Host To Device", times, memcopy_iterations, short_output);

    if(extended_output) printf("finished. cleaning up...\n");

    // clean up memory

    ib_free_memreg(h_idata, 1, false);

    //..... Device to Host

    void *h_odata = NULL;

    ib_allocate_memreg(&h_odata, memSize, 1, false);

    if(extended_output) printf("preparing server...\n");

    // copy data from GPU to Host

    if(extended_output) printf("receving...\n");

    ib_connect_responder(h_odata, 1);
    for (unsigned int i = 0; i < memcopy_iterations + warmup_iterations; i++)
    {
        ib_msg_recv(memSize, 1);
    }

    if(extended_output) printf("finished. cleaning up...\n");
    ib_free_memreg(h_odata, 1, false);
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
          {"inclprep", no_argument,  &time_incl_prepare, 1},
          {"sysmem", no_argument,  &sysmem_only, 1},
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
            print_help(DEFAULT_SIZE, DEFAULT_MEMCOPY_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, DEFAULT_TCP_PORT);
            exit(0);
            break;
        case '?': 
            printf("unknown option: %c\n", optopt);
            break;
        }
    }

    if(extended_output)
    {
        print_variables(server, peer_node, device_id_param, gpu_id, mem_size, memcopy_iterations, warmup_iterations, tcp_port, no_p2p, time_incl_prepare, sysmem_only);
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
