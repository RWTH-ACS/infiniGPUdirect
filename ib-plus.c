#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include "ib.h"
//create array of possible sendwr


/**
 * \brief Prepares list of dublicate send wrs
 * 
 * For sendlist operation. Links all iterations of wrs together so only one ibv_post_send is needed
 */
int prepare_send_list(void *memptr, int mr_id, size_t length, bool fromgpumem, int number)
{
    struct ibv_send_wr * last_send_wr = NULL;
    for (unsigned int i = 0; i < number; i++){
    last_send_wr = ib_create_send_wr(memptr, length, mr_id, fromgpumem, last_send_wr);
    }
}

/**
 * \brief Prepares list of dublicate recv wrs
 * 
 * For sendlist operation. Links all iterations of wrs together so only one ibv_recv_send is needed
 */
int prepare_recv_list(int mr_id, int number)
{
    struct ibv_recv_wr * last_recv_wr = NULL;
    for (unsigned int i = 0; i < number; i++){
    last_recv_wr = ib_create_recv_wr(mr_id, last_recv_wr);
    }
}