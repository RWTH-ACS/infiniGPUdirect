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

#include <string.h>
#include <stdio.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>

#include "oob.h"

int oob_init_listener(oob_t *oob, uint16_t port)
{
    struct sockaddr_in addr = {0};
    struct sockaddr_in peer_addr = {0};
    size_t peer_addr_len = 0;

    if (oob == NULL) return 1;
    memset(oob, 0, sizeof(oob_t));

    if((oob->server_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("oob: creating server socket failed.\n");
        return 1;
    }
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);

    /* Allow port reuse */
    int sockopt = 1;
    setsockopt(oob->server_socket, SOL_SOCKET, SO_REUSEADDR, &sockopt, sizeof(int));

    bind(oob->server_socket, (struct sockaddr*)&addr, sizeof(addr));

    listen(oob->server_socket, 10);

    peer_addr_len = sizeof(struct sockaddr_in);
    if ((oob->socket = accept(oob->server_socket, (struct sockaddr *)&peer_addr, (socklen_t*)&peer_addr_len)) < 0) {
        printf("oob: accept failed.\n");
        return 1;
    }
    if (inet_ntop(AF_INET, (const void*)&peer_addr.sin_addr, oob->peer_address, INET_ADDRSTRLEN) == NULL) {
        printf("oob: inet_ntop failed\n");
        return 1;
    }

    return 0;
}

int oob_init_sender(oob_t *oob, const char* address, uint16_t port)
{
    struct sockaddr_in addr = {0};
    struct hostent *hp;

    if (oob == NULL) return 1;
    memset(oob, 0, sizeof(oob_t));

    if((oob->socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("oob: creating server socket failed.\n");
        return 1;
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if ((hp = gethostbyname(address)) == 0) {
        printf("error resolving hostname: %s\n", address);
        return 1;
    }
    addr.sin_addr = *(struct in_addr*)hp->h_addr;

    if (connect(oob->socket, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        printf("oob: connect failed\n");
        return 1;
    }

    if (inet_ntop(AF_INET, (const void*)&hp->h_addr, oob->peer_address, INET_ADDRSTRLEN) == NULL) {
        printf("oob: inet_ntop failed\n");
        return 1;
    }
    return 0;
}

int oob_send(oob_t *oob, void* buffer, size_t len) 
{
    size_t bytes_sent = 0;
    if (oob == NULL || buffer == NULL) return 1;

    while(bytes_sent < len) {
        bytes_sent += send(oob->socket, (void*)((uint64_t)buffer+bytes_sent), len-bytes_sent, 0);
    }

    return bytes_sent;
}

int oob_receive(oob_t *oob, void *buffer, size_t len)
{
    size_t bytes_received = 0;
    while(bytes_received < len) {
        bytes_received += recv(oob->socket, (void*)((uint64_t)buffer+bytes_received), len-bytes_received, 0);
    }

    return bytes_received;
}

int oob_synchronize(oob_t *oob)
{
    return 0;
}

int oob_close(oob_t *oob)
{
    int ret = 0;
    if (oob == NULL) {
        return ret;
    }
    if (close(oob->socket) != 0) {
        printf("error closing socket: %s\n", strerror(errno));
        ret = 1;
    }
    if (oob->server_socket) {
        if (close(oob->server_socket) != 0) {
            printf("error closing socket: %s\n", strerror(errno));
            ret = 1;
        }
    }
    return ret;
}

