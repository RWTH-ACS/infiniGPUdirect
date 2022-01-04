#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

enum print_flags {
    GB = 1,      // Bandwidth in GB/s
    GIB = 2,     // Bandwidth in GiB/s
    STDDEV = 4,  // Std. Deviation
    INDVAL = 8,  // Each individual value
    AVG = 16,    // Average time
    ALL = 0xFF,  // All above
};


int printInfo();
void print_times(enum print_flags flags, size_t memSize, char * type, double* times, int memcopy_iterations, int short_output, int send_list);
void print_variables(int server, char* peer_node, int ib_device_id, int gpu_id, int mem_size, int iterations, int warmup, int tcp, int nop2p, int sysmem, int sendlist);
void print_help(int default_size, int default_memcopy_iterations, int default_warmup_iterations, int default_tcp_port);

