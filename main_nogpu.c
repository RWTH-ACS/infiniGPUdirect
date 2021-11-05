#include "ib.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

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

void testBandwidthServer(unsigned int memSize, char *peer_node, float * result)
{

    // ...... Client to Server
    float bandwidthInGBs = 0.0f;

    void *s_odata;
    ib_allocate_memreg(&s_odata, memSize, 1, false);

    // copy data from Client to Server

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_server_recv(s_odata, 1, memSize, false);
    }

    ib_free_memreg(s_odata, 1, false);

    //........Server to Client

    void *s_idata;

    struct timeval start, end;

    // allocate memory
    ib_allocate_memreg(&s_idata, memSize, 0, false);

    memset(s_idata, 1, memSize);

    // copy data from Server to Client
    gettimeofday(&start, NULL);
    
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_client_send(s_idata, 0, memSize, peer_node, false);
    }


    gettimeofday(&end, NULL);

    // calculate bandwidth in GB/s
    double time_s = (end.tv_sec - start.tv_sec) * 1e6;
    time_s = (time_s + (end.tv_usec - start.tv_usec)) * 1e-6;

    bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    // clean up memory

    ib_free_memreg(s_idata, 0, false);

    *result = bandwidthInGBs;

    printf("Bandwidth:\n");
    printf("Server to Client (Server): %f GB/s\n", bandwidthInGBs);
}

void testBandwidthClient(unsigned int memSize, char *peer_node, float *result)
{

    //    Client to Server ............
    float bandwidthInGBs = 0.0f;
    void *h_idata;

    struct timeval start, end;

    // allocate memory
    ib_allocate_memreg(&h_idata, memSize, 1, false);
    memset(h_idata, 1, memSize);

    // copy data from Client to Server
    gettimeofday(&start, NULL);


    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_client_send(h_idata, 1, memSize, peer_node, false);
    }

    gettimeofday(&end, NULL);

    // calculate bandwidth in GB/s
    double time_s = (end.tv_sec - start.tv_sec) * 1e6;
    time_s = (time_s + (end.tv_usec - start.tv_usec)) * 1e-6;

    bandwidthInGBs = (memSize * (float)MEMCOPY_ITERATIONS) / (double)1e9;
    bandwidthInGBs = bandwidthInGBs / time_s;

    // clean up memory

    ib_free_memreg(h_idata, 1, false);

    printf("Bandwidth:\n");
    printf("Client to Server (Client): %f GB/s\n", bandwidthInGBs);

    //..... Server to Client

    void *h_odata = NULL;

    ib_allocate_memreg(&h_odata, memSize, 1, false);

    // copy data from Client to Server

    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++)
    {
        ib_server_recv(h_odata, 1, memSize, false);
    }

    ib_free_memreg(h_odata, 1, false);

    *result = bandwidthInGBs;
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

    printf("-----------------------------------------------\n");

    ib_init(device_id_param);

    int iterations = 10;
    float tempResult;

    if (server) {

        float DtH = 0;

        printf("-----------------------------------------------\n");

        printf("-----------------------------------------------\n");

        flush_buf = (char *)malloc(FLUSH_SIZE);

        for(int i = 0; i < iterations; i++){

        testBandwidthServer(DEFAULT_SIZE, peer_node, &tempResult);

        DtH += tempResult;

        printf("-----------------------------------------------\n");

        sleep(5);

        }

        printf("-----------------------------------------------\n");

        DtH = DtH / iterations;

        printf("The average Server to Client bandwidth: %f GB/s\n", DtH);

        free(flush_buf);    

    }
    else
    { //client

//        printInfo();

        float HtD = 0;

        flush_buf = (char *)malloc(FLUSH_SIZE);

        printf("-----------------------------------------------\n");

        for(int i = 0; i < iterations; i++){

        testBandwidthClient(DEFAULT_SIZE, peer_node, &tempResult);


        HtD += tempResult;

        printf("-----------------------------------------------\n");

        sleep(5);

        }

        printf("-----------------------------------------------\n");

        HtD = HtD / iterations;

        printf("The average Client to Server bandwidth: %f GB/s\n", HtD);

        free(flush_buf);    

    }

    ib_cleanup();
    ib_final_cleanup();

    return 0;

}