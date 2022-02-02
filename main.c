/*
 * Copyright 2020-2022 Niklas Eiling
 *           2021-2022 Laura Fuentes Grau
 *        Instiute for Automation of Complex Power Systems,
 *        RWTH Aachen University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */


#include "ib.h"
#include "ib-plus.h"
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
#define PAGE_ROUND_UP(x) ( (((x)) + 0x1000-1)  & (~(0x1000-1)) )

//make into struct

typedef struct test_params {
    int no_p2p;
    int extended_output;
    int short_output;
    int sysmem_only;
    int send_list;
    int only_mem_size;
    int tcp_port;
    int memcopy_iterations;
    int warmup_iterations;
} test_params_t; 

static test_params_t test_params;

/*static int no_p2p = 0;
static int extended_output = 0;
static int short_output = 0;
static int sysmem_only = 0;
static int send_list = 0;
static int only_mem_size = 0;

static int tcp_port = DEFAULT_TCP_PORT;
static int memcopy_iterations = DEFAULT_MEMCOPY_ITERATIONS;
static int warmup_iterations = DEFAULT_WARMUP_ITERATIONS;*/

#ifdef GPU_TIMING
static cudaEvent_t start, stop;
#else
static struct timeval startt;
#endif

// CPU cache flush
#define FLUSH_SIZE (256 * 1024 * 1024)
char *flush_buf;

static oob_t oob;

static inline void init_params(test_params_t *test_params)
{
    test_params->no_p2p = 0;
    test_params->extended_output = 0;
    test_params->short_output = 0;
    test_params->sysmem_only = 0;
    test_params->send_list = 0;
    test_params->only_mem_size = 0;

    test_params->tcp_port = DEFAULT_TCP_PORT;
    test_params->memcopy_iterations = DEFAULT_MEMCOPY_ITERATIONS;
    test_params->warmup_iterations = DEFAULT_WARMUP_ITERATIONS;
}

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
    if (i < test_params.memcopy_iterations) {
        i++;
    }
}


void testBandwidthServer(size_t memSize, char* peer_node, double* times)
{

    // ...... Host to Device
    double bandwidthInGBs = 0.0;
    double bandwidthInGiBs = 0.0;

    int warmup = test_params.send_list ? test_params.warmup_iterations : 1;
    int benchmark = test_params.send_list ? test_params.memcopy_iterations : 1;
    int warmup_loop = test_params.send_list ? 1 : test_params.warmup_iterations;
    int benchmark_loop = test_params.send_list ? 1 : test_params.memcopy_iterations;
    int recv_loop = test_params.send_list ? 1 : warmup_loop + benchmark_loop;
    int recv = test_params.send_list ? warmup + benchmark : 1;

    void *d_odata;
    void *h_odata;

    ib_allocate_memreg(&d_odata, memSize, 1, true);

    // copy data from GPU to Host

    if(test_params.extended_output) printf("preparing server...\n");
    oob_init_listener(&oob, test_params.tcp_port);

    if(test_params.extended_output) printf("receiving...\n");

    if(test_params.sysmem_only)
    {
        ib_allocate_memreg(&h_odata, memSize, 0, false);
        ib_connect_responder(h_odata, 0, &oob);
        if(test_params.send_list) prepare_recv_list(0, recv);
        for (unsigned int i = 0; i < recv_loop; i++)
        {
            if(!test_params.send_list) ib_create_recv_wr(0, NULL);
            ib_post_recv_queue(recv);
        }
        ib_free_memreg(h_odata, 0, false);
    }
    else if(test_params.no_p2p)
    {
        //does not work as intended?
        ib_allocate_memreg(&h_odata, memSize, 0, false);
        ib_connect_responder(h_odata, 0, &oob);
        if(test_params.send_list) prepare_recv_list(0, recv);
        for (unsigned int i = 0; i < recv_loop; i++)
        {
            if(!test_params.send_list) ib_create_recv_wr(0, NULL);
            ib_post_recv_queue(recv);
            cudaMemcpy(d_odata, h_odata, memSize, cudaMemcpyHostToDevice);
        }
        ib_free_memreg(h_odata, 0, false);
    }
    else
    {
        ib_connect_responder(d_odata, 1, &oob);
        if(test_params.send_list) prepare_recv_list(1, recv);
        for (unsigned int i = 0; i < recv_loop ; i++)
        {
            if(!test_params.send_list) ib_create_recv_wr(1, NULL);
            ib_post_recv_queue(recv);
        }
    }

    if(test_params.extended_output) printf("finished. cleaning up...\n");
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

    if(test_params.extended_output) printf("preparing client...\n");

    if(test_params.extended_output) printf("warming up...\n");

    if(test_params.sysmem_only)
    {
        ib_connect_requester(h_idata, 0, peer_node, &oob);
        if(test_params.send_list) prepare_send_list(h_idata, 0, memSize, false, test_params.warmup_iterations);
        for (unsigned int i = 0; i < warmup_loop; i++)
        {   
            if(!test_params.send_list) ib_create_send_wr(h_idata, memSize, 0, false, NULL);
            ib_post_send_queue(warmup);
        }
        // copy data from GPU to Host
        if(test_params.extended_output) printf("sending...\n");
        if(test_params.send_list) prepare_send_list(h_idata, 0, memSize, false, test_params.memcopy_iterations);

        for (unsigned int i = 0; i < benchmark_loop; i++)
        {
            timer_start();
            if(!test_params.send_list) ib_create_send_wr(h_idata, memSize, 0, false, NULL);
            ib_post_send_queue(benchmark);
            timer_stop(times);
        }
    }
    else if(test_params.no_p2p)
    {
        ib_connect_requester(h_idata, 0, peer_node, &oob);
        if(test_params.send_list) prepare_send_list(h_idata, 0, memSize, false, test_params.warmup_iterations);

        for (unsigned int i = 0; i < warmup_loop; i++)
        {
            cudaMemcpy(h_idata, d_idata, memSize, cudaMemcpyDeviceToHost);
            if(!test_params.send_list) ib_create_send_wr(h_idata, memSize, 0, false, NULL);
            ib_post_send_queue(warmup);
        }
        // copy data from GPU to Host
        if(test_params.extended_output) printf("sending...\n");
        if(test_params.send_list) prepare_send_list(h_idata, 0, memSize, false, test_params.memcopy_iterations);

        for (unsigned int i = 0; i < benchmark_loop; i++)
        {
            timer_start();
            cudaMemcpy(h_idata, d_idata, memSize, cudaMemcpyDeviceToHost);
            if(!test_params.send_list) ib_create_send_wr(h_idata, memSize, 0, false, NULL);
            ib_post_send_queue(benchmark);
            timer_stop(times);
        }
    }
    else
    {
        ib_connect_requester(d_idata, 1, peer_node, &oob);

        if(test_params.send_list) prepare_send_list(d_idata, 1, memSize, true, test_params.warmup_iterations);

        for (unsigned int i = 0; i < warmup_loop; i++)
        {
            if(!test_params.send_list) ib_create_send_wr(d_idata, memSize, 1, true, NULL);
            ib_post_send_queue(warmup);
        }
        // copy data from GPU to Host
        if(test_params.extended_output) printf("sending...\n");

        if(test_params.send_list) prepare_send_list(d_idata, 1, memSize, true, test_params.memcopy_iterations);

        for (unsigned int i = 0; i < benchmark_loop; i++)
        {
            timer_start();
            if(!test_params.send_list) ib_create_send_wr(d_idata, memSize, 1, true, NULL);
            ib_post_send_queue(benchmark);
            timer_stop(times);
        }
    }
    if(test_params.extended_output) printf("finished.\n");

    print_times(ALL, memSize, "Device to Host", times, test_params.memcopy_iterations, test_params.short_output, test_params.send_list);

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
    
    int warmup = test_params.send_list ? test_params.warmup_iterations : 1;
    int benchmark = test_params.send_list ? test_params.memcopy_iterations : 1;
    int warmup_loop = test_params.send_list ? 1 : test_params.warmup_iterations;
    int benchmark_loop = test_params.send_list ? 1 :test_params. memcopy_iterations;
    int recv_loop = test_params.send_list ? 1 : warmup_loop + benchmark_loop;
    int recv = test_params.send_list ? warmup + benchmark : 1;

    struct timeval start, end;

    // allocate memory
    ib_allocate_memreg(&h_idata, memSize, 1, false);
    memset(h_idata, 1, memSize);

    if(test_params.extended_output) printf("preparing client...\n");
    oob_init_sender(&oob, peer_node, test_params.tcp_port);
    
    if(test_params.extended_output) printf("warming up...\n");

    ib_connect_requester(h_idata, 1, peer_node, &oob);

    if(test_params.send_list) prepare_send_list(h_idata, 1, memSize, false, test_params.warmup_iterations);


    for (unsigned int i = 0; i < warmup_loop; i++)
    {
        if(!test_params.send_list) ib_create_send_wr(h_idata, memSize, 1, false, NULL);
        ib_post_send_queue(warmup);
    }

    // copy data from GPU to Host
    if(test_params.extended_output) printf("sending...\n");

    if(test_params.send_list) prepare_send_list(h_idata, 1, memSize, false, test_params.memcopy_iterations);

    for (unsigned int i = 0; i < benchmark_loop; i++)
    {
        timer_start();
        if(!test_params.send_list) ib_create_send_wr(h_idata, memSize, 1, false, NULL);
        ib_post_send_queue(benchmark);
        timer_stop(times);
    }

    print_times(ALL, memSize, "Host To Device", times, test_params.memcopy_iterations, test_params.short_output, test_params.send_list);

    if(test_params.extended_output) printf("finished. cleaning up...\n");

    // clean up memory

    ib_free_memreg(h_idata, 1, false);

    //..... Device to Host

    void *h_odata = NULL;

    ib_allocate_memreg(&h_odata, memSize, 1, false);

    if(test_params.extended_output) printf("preparing server...\n");

    // copy data from GPU to Host

    if(test_params.extended_output) printf("receving...\n");

    ib_connect_responder(h_odata, 1, &oob);
    if(test_params.send_list) prepare_recv_list(1, recv);

    for (unsigned int i = 0; i < recv_loop ; i++)
    {
        if(!test_params.send_list) ib_create_recv_wr(1, NULL);
        ib_post_recv_queue(recv);
    }

    if(test_params.extended_output) printf("finished. cleaning up...\n");
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
    uint32_t max_msg_size = 0;
    init_params(&test_params);


    while (1)
    {
        static struct option long_options[] =
        {
          {"nop2p", no_argument,  &test_params.no_p2p, 1},
          {"extended", no_argument,  &test_params.extended_output, 1},
          {"short", no_argument,  &test_params.short_output, 1},
          {"sysmem", no_argument,  &test_params.sysmem_only, 1},
          {"sendlist", no_argument,  &test_params.send_list, 1},
          {"onlymemsize", no_argument,  &test_params.only_mem_size, 1},
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
            test_params.tcp_port = atoi(optarg);
            break;
        case 'w':
            test_params.warmup_iterations = atoi(optarg);
            break;
        case 'i':
            test_params.memcopy_iterations = atoi(optarg);
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

    if(test_params.only_mem_size) test_params.memcopy_iterations = 1;

    double times[test_params.memcopy_iterations];

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


    ib_init(device_id_param, &max_msg_size);

    if(PAGE_ROUND_UP(mem_size + 2) > max_msg_size){
        if(test_params.extended_output) fprintf(stderr, "the mem_size is too large, start splitting...\n");
        int splits = round(mem_size/max_msg_size);
        test_params.warmup_iterations = test_params.warmup_iterations*splits;
        test_params.memcopy_iterations = test_params.memcopy_iterations*splits;
        mem_size = max_msg_size - (PAGE_ROUND_UP(max_msg_size + 2) - max_msg_size);
    }

    if(test_params.extended_output)
    {
        print_variables(server, peer_node, device_id_param, gpu_id, mem_size, test_params.memcopy_iterations, test_params.warmup_iterations, test_params.tcp_port, test_params.no_p2p, test_params.sysmem_only, test_params.send_list, test_params.only_mem_size);
    }

    if (server) {

        if(test_params.extended_output){
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
