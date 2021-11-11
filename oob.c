/*
 * Copyright 2014 Simon Pickartz,
 *           2020-2021 Niklas Eiling
 *           2021 Laura Fuentes Grau
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

#include <stdio.h>
#include <stdint.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "oob.h"

int oob_init_listener(oob_t *oob, uint16_t port)
{
    struct sockaddr_in addr = {0}; 

    if (port == NULL) return 1;
    memset(port, 0, sizeof(oob_t));

    if((oob->socket = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("oob: creating socket failed.\n");
        return 1;
    }
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    server_address.sin_port = htons(port);

    /* Allow port reuse */
    int sockopt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &sockopt, sizeof(int));

    int client_addr_len = 0;
    struct sockaddr_in serv_addr;
    struct sockaddr_in client_addr;


    bind(listen_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

    listen(listen_sock, 10);

    client_addr_len = sizeof(struct sockaddr_in);
    if ((com_sock = accept(listen_sock, (struct sockaddr *)&client_addr, (socklen_t*)&client_addr_len)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    char buf[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, (const void*)&client_addr.sin_addr, buf, INET_ADDRSTRLEN) == NULL) {
        perror("inet_ntop");
        exit(EXIT_FAILURE);
    }
    //fprintf(stderr, "[INFO] Incoming from: %s\n", buf);
    //fprintf(stderr, "[INFO] Trying to connect to migration server: %s\n", buf);
    while (connect(com_sock, (struct sockaddr *)&ib_pp_server, sizeof(ib_pp_server)) < 0);
    /*if (connect(com_sock, (struct sockaddr *)&ib_pp_server, sizeof(ib_pp_server)) < 0) {
        perror("connect");
        return -1;
        }*/
    //fprintf(stderr, "[INFO] Successfully connected to: %s\n", buf);
    return 0;
}

int oob_init_sender(oob_t *oob, const char* address);

int oob_send(oob_t *oob, void* buffer, size_t len);
int oob_receive(oob_t *oob, void *buffer, size_t len);

int oob_synchronize(oob_t *oob);

int connect_to_server(void)
{
    char buf[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, (const void*)&ib_pp_server.sin_addr, buf, INET_ADDRSTRLEN) == NULL) {
        perror("inet_ntop");
        return -1;
    }

    if((com_sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        perror("socket");
        return -1;
    }
    return 0;
}


/**
 * \brief Waits for a migration source to connect via TCP/IP
 *
 * \param listen_portno the port of the migration socket
 */
void wait_for_client(uint16_t listen_portno)
{   
    int client_addr_len = 0;
    struct sockaddr_in serv_addr;
    struct sockaddr_in client_addr;

    /* open migration socket */
    //fprintf(stderr, "[INFO] Waiting for the client side...\n");
    listen_sock = socket(AF_INET, SOCK_STREAM, 0);
    memset(&serv_addr, '0', sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(listen_portno);

    int yes = 1;

    setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (void*) &yes, (socklen_t) sizeof(yes));

    bind(listen_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));

    listen(listen_sock, 10);

    client_addr_len = sizeof(struct sockaddr_in);
    if ((com_sock = accept(listen_sock, (struct sockaddr *)&client_addr, (socklen_t*)&client_addr_len)) < 0) {
        perror("accept");
        exit(EXIT_FAILURE);
    }
    char buf[INET_ADDRSTRLEN];
    if (inet_ntop(AF_INET, (const void*)&client_addr.sin_addr, buf, INET_ADDRSTRLEN) == NULL) {
        perror("inet_ntop");
        exit(EXIT_FAILURE);
    }
    //fprintf(stderr, "[INFO] Incoming from: %s\n", buf);
}

/**
 * \brief Receives data from the migration socket
 *
 * \param buffer the destination buffer
 * \param length the buffer size
 */
int recv_data(void *buffer, size_t length)
{
    size_t bytes_received = 0;
    while(bytes_received < length) {
        bytes_received += recv(
                com_sock,
                (void*)((uint64_t)buffer+bytes_received),
                length-bytes_received,
                    0);
    }

    return bytes_received;
}

/**
 * \brief Sends data via the migration socket
 *
 * \param buffer the source buffer
 * \param length the buffer size
 */
int send_data(void *buffer, size_t length)
{
    size_t bytes_sent = 0;
    while(bytes_sent < length) {
        bytes_sent += send(
                com_sock,
                (void*)((uint64_t)buffer+bytes_sent),
                length-bytes_sent,
                    0);
    }

    return bytes_sent;
}

static inline void
close_sock(int sock)
{
    if (close(sock) < 0) {
        fprintf(stderr,
                "ERROR: Could not close the communication socket "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(EXIT_FAILURE);
    }
}

/**
 * \brief Closes the TCP connection
 */
void close_comm_channel(void)
{
    if (listen_sock) {
        close_sock(listen_sock);
    }

    close_sock(com_sock);
}




