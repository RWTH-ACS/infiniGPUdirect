#ifndef _CPU_IB_H_
#define _CPU_IB_H_
#include <stdbool.h>
#include <stdlib.h>

int ib_init(int _device_id);
int ib_connect_server(void *memreg, int mr_id);
int ib_connect_client(void *memreg, int mr_id, char *server_address);
void ib_free_memreg(void* memreg, int mr_id, bool gpumemreg);
void ib_cleanup(void);
void ib_final_cleanup(void);
int ib_allocate_memreg(void** mem_address, int memsize, int mr_id, bool gpumemreg);
int ib_server_recv(void *memptr, int mr_id, int length, bool togpumem);
int ib_client_send(void *memptr, int mr_id, int length, char *peer_node, bool fromgpumem);
int ib_server_send_result(void *memptr, int mr_id, int length, char *peer_node);
int ib_server_prepare(void *memptr, int mr_id, size_t length, bool togpumem);
int ib_client_prepare(void *memptr, int mr_id, size_t length, char *peer_node, bool fromgpumem);

#include <stdint.h>
int ib_init_oob_listener(uint16_t port);
int ib_init_oob_sender(const char* address, uint16_t port);

#endif


