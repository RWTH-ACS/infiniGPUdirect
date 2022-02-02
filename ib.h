/*
 * Copyright 2014 Simon Pickartz,
 *           2020-2022 Niklas Eiling
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
/** InfiniBand functions
 * Provides all neccessary functions and verbs for IB communication
 */

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include "oob.h"


int ib_init(int _device_id, uint32_t *max_msg_size);
int ib_connect_responder(void *memreg, int mr_id, oob_t *oob);
int ib_connect_requester(void *memreg, int mr_id, char *responder_address, oob_t *oob);
void ib_free_memreg(void* memreg, int mr_id, bool gpumemreg);
void ib_cleanup(void);
void ib_final_cleanup(void);
int ib_allocate_memreg(void** mem_address, int memsize, int mr_id, bool gpumemreg);
struct ibv_send_wr * ib_create_send_wr(void *memreg, size_t length, int mr_id, bool gpumemreg, struct ibv_send_wr * next_send_wr);
void ib_post_send_queue(int number);
struct ibv_recv_wr * ib_create_recv_wr(int mr_id, struct ibv_recv_wr * next_recv_wr);
void ib_post_recv_queue(int number);


