#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>


int prepare_send_list(void *memptr, int mr_id, size_t length, bool fromgpumem, int number);
int prepare_recv_list(int mr_id, int number);
