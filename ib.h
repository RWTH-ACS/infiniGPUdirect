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

int ib_init(int _device_id);
int ib_connect_responder(void *memreg, int mr_id);
int ib_connect_requester(void *memreg, int mr_id, char *responder_address);
void ib_free_memreg(void* memreg, int mr_id, bool gpumemreg);
void ib_cleanup(void);
void ib_final_cleanup(void);
int ib_allocate_memreg(void** mem_address, int memsize, int mr_id, bool gpumemreg);
int ib_responder_send_result(void *memptr, int mr_id, int length, char *peer_node);
void ib_msg_send(void *memptr, int mr_id, size_t length, bool fromgpumem, int send_list, int iterations);
void ib_msg_recv(uint32_t length, int mr_id, int iterations);
int ib_prepare_send_list(void *memptr, int mr_id, size_t length, bool fromgpumem, int iterations);


#include <stdint.h>
int ib_init_oob_listener(uint16_t port);
int ib_init_oob_sender(const char* address, uint16_t port);

