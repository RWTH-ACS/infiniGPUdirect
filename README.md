# InfiniGPUDirect

InfiniGPUDirect is an Infiniband benchmark application that is highly modifiable and capable of testing GPUDirect RDMA over Infiniband.

# Dependencies
InfiniGPUDirect requires
- Mellanox OFED
- Nvidia Driver
- CUDA Toolkit (E.g. CUDA 11.1)

# Building

```
git clone https://git.rwth-aachen.de/laurafuentesgrau/infinigpudirect.git
cd InfiniGPUDirect
make
```

Environment variables for Makefile:
- `GPU_TIMING`: If set to `YES` application will use the GPU timer for benchmarking.

# Execution

IGD sends data back and forth between a server (-s) which includes the GPUDirect GPU used for testing GPUDirect RDMA over InfiniBand and a client (-c) which does not need a GPU.
For each type of transfer (Device to Host or Host to Device) the sending side measures the time and calculates the bandwidth. 

```
<path-to-IGD>/ibTest -c/-s -p <peer node> <options>
```

Options (displayable vie -h):
- -d            <IB device ID> (default 0)
- -g            <GPU ID> (default 0)
- -m            <memory size> (default 128000000)
- -i            <memcopy iterations> (default 25)
- -w            <warmup iterations> (default 3)
- -t            <TCP port> (default 4211)
- --nop2p       disable peer to peer (flag)
- --extended    extended terminal output (flag)
- --short       short terminal output (flag)
- --sysmem      data transfer only between system memory (flag)
- --sendlist    send all iterations at once as a list of WRs (flag)

# Output
Default:
- Time measurement results of individual iterations
- Average time needed
- Standard deviation of the individual results
- Bandwidth in GB/s
- Bandwidth in GiB/s

Short:
- Average time needed
- Standard deviation of the individual results
- Bandwidth in GB/s
- Bandwidth in GiB/s

Extended:
- Options (default or selected) of current run
- List of available GPUs (ID, name, memory clock rate, memory bus width, peak memory bandwidth)
- Basic debugging/execution information
- Time measurement results of individual iterations
- Average time needed
- Standard deviation of the individual results
- Bandwidth in GB/s
- Bandwidth in GiB/s



### Example: Running a benchmark without peer to peer using 30 iterations and extended output

On server "ghost" via InfiniBand device 1 on GPU 1
```
./ibTest -s -p "elmo" -d 1 -g 1 -i 30 --nop2p --extended
```

On client "elmo" via InfiniBand device 1 on GPU 1
```
./ibTest -c -p "ghost" -d 1 -i 30 --nop2p --extended
```

