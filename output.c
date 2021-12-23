#include "output.h"



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
     printf("-----------------------------------------------\n");

    return 0;
}

void print_times(enum print_flags flags, size_t memSize, char * type, double* times, int memcopy_iterations, int short_output)
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

void print_variables(int server, char* peer_node, int ib_device_id, int gpu_id, int mem_size, int iterations, int warmup, int tcp, int nop2p, int inclprep, int sysmem, int sendlist){
    printf("-----------------------------------------------\n");
    server ? printf("Peer (Client): %s\n", peer_node) : printf("Peer (Server): %s\n", peer_node);
    printf("InfiniBand device: %d\n", ib_device_id);
    printf("TCP port: %d\n", tcp);
    if(server) printf("GPU: %d\n", gpu_id);
    printf("Memory size: %d\n", mem_size);
    printf("Iterations: %d\n", iterations);
    printf("Warmup iterations: %d\n", warmup);
    if(nop2p) printf("Using NO peer to peer\n");
    if(inclprep) printf("Including preparation time\n");
    if(sysmem) printf("Data transfer only between system memory\n");
    if(sendlist) printf("Data transfer via only one post of a list\n");
#ifdef GPU_TIMING
    printf("Using GPU timer\n");
#endif
    printf("-----------------------------------------------\n");
}

void print_help(int default_size, int default_memcopy_iterations, int default_warmup_iterations, int default_tcp_port)
{
    printf("Command line parameters:\n"
                   " -s/-c         server/client (required)\n -p            <peer node> (required)\n -d            <IB device ID> (default 0)\n -g            <GPU ID> (default 0)\n"
                    " -m            <memory size> (default %d)\n -i            <memcopy iterations> (default %d)\n "
                    "-w            <warmup iterations> (default %d)\n -t            <TCP port> (default %d)\n"
                    " --nop2p       disable peer to peer (flag)\n --extended    extended terminal output (flag)\n --short       short terminal output (flag)\n"
                    " --inclprep    include preparation time (flag)\n --sysmem      data transfer only between system memory (flag)\n",
                     default_size, default_memcopy_iterations, default_warmup_iterations, default_tcp_port);
}