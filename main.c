#include "ib.h"
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>



#define NAME_LENGTH 128
#define MAX_LEN 32
#define LENGTH 5
int main(int argc, char **argv)
{
        int arg             = 0;
    int32_t server          = -1;
    char peer_node[NAME_LENGTH]     = { [0 ... NAME_LENGTH-1] = 0 };
    int device_id_param = 0;

    while ((arg = getopt(argc, argv, "hscbd:p:")) != -1) {
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
        case 'h':
            printf("usage ./%s "
                   "-s/-c -p <peer_node>\n", argv[0]);
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

    ib_init(device_id_param);
    if (server) {
        ib_allocate_memreg(&gpumemptr0, size, 0, true);
        ib_allocate_memreg(&gpumemptr1, size, 1, true);
        ib_allocate_memreg(&gpumemptr2, size, 2, true);

        ib_server_recv(gpumemptr0, 0, size, true);
        ib_server_recv(gpumemptr1, 1, size, true);

        ib_client_send(gpumemptr2, 2 , size, peer_node,true);

        // Invoke kernel
/*        int threadsPerBlock = 256;
        int blocksPerGrid = (MAX_LEN + threadsPerBlock - 1) / threadsPerBlock;
        VecAdd<<<blocksPerGrid, threadsPerBlock>>>(gpumemptr0, gpumemptr1, gpumemptr2, MAX_LEN);

        char *str = (char *)malloc(length);

        cudaMemcpy(str, gpumemptr2, MAX_LEN, cudaMemcpyDeviceToHost);
        printf("received: %s\n", str);*/
        ib_free_memreg(gpumemptr0, 0, true);
        ib_free_memreg(gpumemptr1, 1, true);
        ib_free_memreg(gpumemptr2, 2, true);

    }
    else
    { //client
        ib_allocate_memreg(&memptr0, size, 0, false);
        ib_allocate_memreg(&memptr1, size, 1, false);
        ib_allocate_memreg(&memptr2, size, 2, false);

        ib_client_send(memptr0, 0, size, peer_node, false);
        ib_client_send(memptr1, 1, size, peer_node, false);

        ib_server_recv(memptr2, 2, size, false);

        ib_free_memreg(memptr0, 0, false);
        ib_free_memreg(memptr1, 1, false);
        ib_free_memreg(memptr2, 2, false);
    }
    /* wait for opponent */
    printf("Enter barrier ...\n");
    //ib_pp_barrier(0, server);
    printf("Benchmark finished!\n");

    ib_cleanup();
    ib_final_cleanup();

    return 0;

}

