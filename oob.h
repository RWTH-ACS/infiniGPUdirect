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
/** Out-of-band communication primitives
 * Uses a TCP connection to communicate auxiliary information between client
 * and host
 */

#include <stdint.h>
#include <netdb.h>

typedef struct oob {
    int server_socket;
    int socket;
    char peer_address[INET_ADDRSTRLEN];

} oob_t;

int oob_init_listener(oob_t *oob, uint16_t port);

int oob_init_sender(oob_t *oob, const char* address, uint16_t port);

int oob_send(oob_t *oob, void* buffer, size_t len);
int oob_receive(oob_t *oob, void *buffer, size_t len);

int oob_synchronize(oob_t *oob);

int oob_close(oob_t *oob);

