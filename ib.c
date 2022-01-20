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
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "oob.h"

#define PAGE_ROUND_UP(x) ( (((x)) + 0x1000-1)  & (~(0x1000-1)) )
#define PAGE_SIZE       (0x1000)

/* IB definitions */
#define CQ_ENTRIES      (100)
#define IB_WRITE_WR_ID   (2)
#define IB_RECV_WR_ID    (1)
#define IB_SEND_WR_ID    (0)
#define IB_MTU       (IBV_MTU_2048)
#define MAX_DEST_RD_ATOMIC  (1)
#define MIN_RNR_TIMER       (1)
#define MAX_SEND_WR         (8192)  // TODO: should be
                    // com_hndl.dev_attr_ex.orig_attr.max_qp_wr
                    // fix for mlx_5 adapter
#define MAX_INLINE_DATA (0)
#define MAX_RECV_WR (100)
//max here seems to be 3:
#define MAX_SEND_SGE    (1)
#define MAX_RECV_SGE    (1)



/*
 * Helper data types
 * QP: Queue Pair: consists typically of a sending and receiving queue where Work Requests (WR) are stored
 */
typedef struct ib_qp_info {
    uint32_t qpn;
    uint16_t lid;
    uint16_t psn;
    uint32_t key;
    uint64_t addr;
} ib_qp_info_t;

typedef struct ib_com_buf { /*Communication Buffer information*/
    uint8_t *send_buf;
    uint8_t *recv_buf;
    ib_qp_info_t qp_info;
    volatile char *new_msg;
    volatile char *send_flag;
} ib_com_buf_t;


typedef struct ib_com_hndl { /*Communication Handler: contains all necessary information for InfiniBand communication*/
    struct ibv_context      *ctx;       /* device context */
    struct ibv_device_attr_ex   dev_attr_ex;    /* extended device attributes */
    struct ibv_port_attr        port_attr;  /* port attributes */
    struct ibv_pd           *pd;        /* protection domain */
    struct ibv_mr           *mr;        /* memory region */
    struct ibv_cq           *cq;        /* completion queue */
    struct ibv_qp           *qp;        /* queue pair */
    struct ibv_comp_channel     *comp_chan;     /* completion event channel */
    struct ibv_send_wr      *send_wr;   /* data send list */
    struct ibv_recv_wr      *recv_wr;   /* data send list */
    ib_com_buf_t         loc_com_buf;
    ib_com_buf_t         rem_com_buf;
    uint8_t             used_port;  /* port of the IB device */
    uint32_t            buf_size;   /* size of the buffer */
} ib_com_hndl_t;
/*
 * Global variables
 */
uint8_t my_mask, rem_mask;

static ib_com_hndl_t ib_com_hndl;
static int device_id = 0;
static uint32_t max_qp_wr = 8192; /*device parameter: max number of Work Requests in on QP*/


static struct ibv_mr *mrs[32]; /*TODO make into list for dynamic length, wiederverwendbar*/
//klare trennung benchmark infiniband library

/**
 * \brief IB synchronization barrier
 * 
 * synchronizes requester and responder in case of one_sided communication
 */
void ib_barrier(int mr_id, int32_t responder)
{
    if (responder) {
        struct ibv_sge sg_list = {
            .addr   = 0,
            .length = 0,
            .lkey   = mrs[mr_id]->lkey
        };
        struct ibv_recv_wr recv_wr = {
            .wr_id      = IB_RECV_WR_ID,
            .sg_list    = &sg_list,
            .num_sge    = 1,
        };
        struct ibv_recv_wr *bad_wr;

        if (ibv_post_recv(ib_com_hndl.qp, &recv_wr, &bad_wr) < 0) {
            fprintf(stderr,
                    "ERROR: Could post recv "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(errno);
        }
    } else {
        struct ibv_sge sg_list = {
            .addr   = 0,
            .length = 0,
            .lkey   = mrs[mr_id]->lkey
        };
        struct ibv_send_wr send_wr = {
            .wr_id      = IB_SEND_WR_ID,
            .sg_list    = &sg_list,
            .num_sge    = 1,
            .opcode     = IBV_WR_SEND,
            .send_flags = IBV_SEND_SIGNALED,
        };
        struct ibv_send_wr *bad_wr;

        if (ibv_post_send(ib_com_hndl.qp, &send_wr, &bad_wr) < 0) {
            fprintf(stderr,
                    "ERROR: Could post send "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(errno);
        }
    }

    /* wait for completion */
    struct ibv_wc wc;
        int ne;
    do {
        if ((ne = ibv_poll_cq(ib_com_hndl.cq, 1, &wc)) < 0) {
            fprintf(stderr,
                    "ERROR: Could poll on CQ (for barrier)"
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(errno);
        }

    } while (ne < 1);
    if (wc.status != IBV_WC_SUCCESS) {
        fprintf(stderr,
            "ERROR: WR failed status %s (%d) for wr_id %d (for barrier)\n",
            ibv_wc_status_str(wc.status),
            wc.status,
            (int)wc.wr_id);
    }
}

/**
 * \brief registers a memory region with the protection domain
 * 
 * This doesn NOT allocate memory
 */
size_t ib_register_memreg(void** mem_address, size_t memsize, int mr_id)
{
    if (mem_address == NULL) return 0;

    if ((mrs[mr_id] = ibv_reg_mr(ib_com_hndl.pd,
                                    *mem_address,
                                    memsize,
                        IBV_ACCESS_LOCAL_WRITE |
                        IBV_ACCESS_REMOTE_WRITE)) == NULL) {
        fprintf(stderr,
                "ERROR: Could not register the memory region "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
    return 0;
}


/**
 * \brief allocates and registers a memory region with the protection domain
 * 
 * allocates a rounded up memory region. Parameter gpumemreg signals if it should be system or gpu memory
 */
size_t ib_allocate_memreg(void **mem_address, size_t memsize, int mr_id, bool gpumemreg)
{
    int res;
    size_t real_size = PAGE_ROUND_UP(memsize + 2);
    if (mem_address == NULL)
        return 0;
//    fprintf(stderr, "[INFO] Communication buffer size: %u KiB\n", real_size / 1024);

    if (gpumemreg)
    {
        if ((res = cudaMalloc(mem_address, real_size)) != cudaSuccess)
        {
            fprintf(stderr,
                    "ERROR: Could not allocate mem for communication bufer "
                    " - %d (%s). Abort!\n",
                    res, strerror(res));
            exit(-1);
        }
        if ((res = cudaMemset(*mem_address, 0, real_size)) != cudaSuccess)
        {
            fprintf(stderr,
                    "ERROR: Could not initialize mem for communication bufer "
                    " - %d (%s). Abort!\n",
                    res, strerror(res));
            exit(-1);
        }
    }
    else
    {
        if ((res = posix_memalign((void *)mem_address,
                                  0x1000,
                                  real_size)) < 0)
        {
            fprintf(stderr,
                    "ERROR: Could not allocate mem for communication bufer "
                    " - %d (%s). Abort!\n",
                    res, strerror(res));
            exit(-1);
        }
        memset(*mem_address, 0x0, real_size);
    }

    if ((mrs[mr_id] = ibv_reg_mr(ib_com_hndl.pd,
                                 *mem_address,
                                 real_size,
                                 IBV_ACCESS_LOCAL_WRITE |
                                     IBV_ACCESS_REMOTE_WRITE)) == NULL)
    {
        fprintf(stderr,
                "ERROR: Could not register the memory region "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
        exit(errno);
    }
    return 0;
}

/**
 * \brief initializes communication buffer for data transfer
 * 
 * Creates completion event channel and completion queue. Creates send and recv queue pair and initializes it. Sets QP into INIT state and
 * fills in local QP info.
 */
void ib_init_com_hndl(int mr_id)
{
    /* create completion event channel */
    if ((ib_com_hndl.comp_chan =
        ibv_create_comp_channel(ib_com_hndl.ctx)) == NULL) {
        fprintf(stderr,
            "[ERROR] Could not create the completion channel "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(EXIT_FAILURE);
    }


    /* create the completion queue */
    if ((ib_com_hndl.cq = ibv_create_cq(ib_com_hndl.ctx,
                               CQ_ENTRIES,
                               NULL,        /* TODO: check cq_context */
                           ib_com_hndl.comp_chan,
                           0)) == NULL) {   /* TODO: check comp_vector */
        fprintf(stderr,
                "ERROR: Could not create the completion queue "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* create send and recv queue pair  and initialize it */
    struct ibv_qp_init_attr init_attr = {
        .send_cq = ib_com_hndl.cq,
        .recv_cq = ib_com_hndl.cq,
        .cap     = {
            .max_inline_data    = MAX_INLINE_DATA,
            .max_send_wr        = MAX_SEND_WR,
            .max_recv_wr        = MAX_RECV_WR,
            .max_send_sge       = MAX_SEND_SGE,
            .max_recv_sge       = MAX_RECV_SGE,
        },
        .qp_type = IBV_QPT_RC
//      .sq_sig_all = 0 /* we do not want a CQE for each WR */
    };
    if ((ib_com_hndl.qp = ibv_create_qp(ib_com_hndl.pd,
                               &init_attr)) == NULL) {
        fprintf(stderr,
                "ERROR: Could not create the queue pair "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    struct ibv_qp_attr attr = {
        .qp_state           = IBV_QPS_INIT,
        .pkey_index         = 0,
        .port_num       = ib_com_hndl.used_port,
        .qp_access_flags    = (IBV_ACCESS_REMOTE_WRITE)
    };
    if (ibv_modify_qp(ib_com_hndl.qp,
              &attr,
              IBV_QP_STATE |
              IBV_QP_PKEY_INDEX |
              IBV_QP_PORT |
              IBV_QP_ACCESS_FLAGS) < 0) {
        fprintf(stderr,
                "ERROR: Could not set QP into init state "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* fill in local qp_info */
    ib_com_hndl.loc_com_buf.qp_info.qpn  = ib_com_hndl.qp->qp_num;
    ib_com_hndl.loc_com_buf.qp_info.psn  = lrand48() & 0xffffff;
    ib_com_hndl.loc_com_buf.qp_info.key  = mrs[mr_id]->lkey;
    ib_com_hndl.loc_com_buf.qp_info.addr = (uint64_t)ib_com_hndl.loc_com_buf.recv_buf;
    ib_com_hndl.loc_com_buf.qp_info.lid  = ib_com_hndl.port_attr.lid;

    ib_com_hndl.send_wr = NULL;
    ib_com_hndl.recv_wr = NULL;
}

/**
 * \brief connects to remote communication buffer
 * 
 * Connects QPs and sets QP into RTS (Ready to Send) state
 */
void ib_con_com_buf()
{
    /* connect QPs */
    struct ibv_qp_attr qp_attr = {
        .qp_state       = IBV_QPS_RTR,
        .path_mtu       = IBV_MTU_2048,
        .dest_qp_num        = ib_com_hndl.rem_com_buf.qp_info.qpn,
        .rq_psn         = ib_com_hndl.rem_com_buf.qp_info.psn,
        .max_dest_rd_atomic = MAX_DEST_RD_ATOMIC,
        .min_rnr_timer      = MIN_RNR_TIMER,
        .ah_attr        = {
            .is_global  = 0,
            .sl         = 0,
            .src_path_bits  = 0,
            .dlid       = ib_com_hndl.rem_com_buf.qp_info.lid,
            .port_num   = ib_com_hndl.used_port,
        }
    };
    if (ibv_modify_qp(ib_com_hndl.qp,
              &qp_attr,
              IBV_QP_STATE |
              IBV_QP_PATH_MTU |
              IBV_QP_DEST_QPN |
              IBV_QP_RQ_PSN |
              IBV_QP_MAX_DEST_RD_ATOMIC |
              IBV_QP_MIN_RNR_TIMER |
              IBV_QP_AV)) {
        fprintf(stderr,
                "ERROR: Could not put QP into RTR state"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
    qp_attr.qp_state        = IBV_QPS_RTS;
    qp_attr.timeout         = 14;
    qp_attr.retry_cnt       = 7;
    qp_attr.rnr_retry       = 7;
    qp_attr.sq_psn          = ib_com_hndl.loc_com_buf.qp_info.psn;
    qp_attr.max_rd_atomic   = 1;
    if (ibv_modify_qp(ib_com_hndl.qp, &qp_attr,
              IBV_QP_STATE              |
              IBV_QP_TIMEOUT            |
              IBV_QP_RETRY_CNT          |
              IBV_QP_RNR_RETRY          |
              IBV_QP_SQ_PSN             |
              IBV_QP_MAX_QP_RD_ATOMIC)) {
        fprintf(stderr,
                "ERROR: Could not put QP into RTS state"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
}


/**
 * \brief Cleans up list of send work requests
 *
 * Frees up all wr that are still saved in the list headed by ib_com_hndl.send_wr (each wr contains a pointer to the next wr)
 * As ib_com_hndl.send_wr is the list given to ibv_post_send this needs to be called after each post to free memory and to prevent dublicate transmissions
 */
static inline void cleanup_send_list(void)
{
    struct ibv_send_wr *cur_send_wr = ib_com_hndl.send_wr;
    struct ibv_send_wr *tmp_send_wr = NULL;
    while (cur_send_wr != NULL) {
        free(cur_send_wr->sg_list);
        tmp_send_wr = cur_send_wr;
        cur_send_wr = cur_send_wr->next;
        free(tmp_send_wr);
    }

    ib_com_hndl.send_wr = NULL;
}

/**
 * \brief Cleans up list of recv work requests
 *
 * Frees up all wr that are still saved in the list headed by ib_com_hndl.recv_wr (each wr contains a pointer to the next wr)
 * As ib_com_hndl.send_wr is the list given to ibv_post_send this needs to be called after each post to free memory and to prevent dublicate transmissions
 */
static inline void cleanup_recv_list(void)
{
    struct ibv_recv_wr *cur_recv_wr = ib_com_hndl.recv_wr;
    struct ibv_recv_wr *tmp_recv_wr = NULL;
    while (cur_recv_wr != NULL) {
        free(cur_recv_wr->sg_list);
        tmp_recv_wr = cur_recv_wr;
        cur_recv_wr = cur_recv_wr->next;
        free(tmp_recv_wr);
    }

    ib_com_hndl.recv_wr = NULL;
}

/**
 * \brief Prepares Send Work Request 'ibv_send_wr'
 *
 * This function prepares an 'ibv_send_wr' structure for the
 * transmission of a single memory page using the IBV_WR_RDMA_WRITE verb.
 */
struct ibv_send_wr * ib_create_send_wr(void *memreg, size_t length, int mr_id, bool gpumemreg, struct ibv_send_wr * next_send_wr)
{

    //memset(ib_com_hndl.loc_com_buf.send_buf, 0x42, ib_com_hndl.buf_size);
    static uint8_t one = 1;
    /* create work request */
    struct ibv_send_wr *send_wr =  (struct ibv_send_wr*)calloc(1, sizeof(struct ibv_send_wr));
    struct ibv_sge *sge =  (struct ibv_sge*)calloc(1, sizeof(struct ibv_sge));

    /* basic work request configuration */
    send_wr->next       = next_send_wr;
    send_wr->sg_list    = sge;
    send_wr->num_sge    = 1;

    if (gpumemreg)
    {
        if (cudaMemcpy(memreg + length, &one, 1, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            fprintf(stderr, "error");
        }
    }
    else
    {
        *((uint8_t *)memreg + length) = 1;
    }


    send_wr->sg_list->addr = (uintptr_t)memreg;
    send_wr->sg_list->length = length + 1;
    send_wr->sg_list->lkey = mrs[mr_id]->lkey;

    send_wr->wr.rdma.rkey = ib_com_hndl.rem_com_buf.qp_info.key;
    send_wr->wr.rdma.remote_addr    = (uintptr_t)ib_com_hndl.rem_com_buf.recv_buf;


    send_wr->wr_id          = IB_WRITE_WR_ID;
    send_wr->opcode			= IBV_WR_RDMA_WRITE_WITH_IMM;
	send_wr->send_flags		= IBV_SEND_SIGNALED | IBV_SEND_SOLICITED;
	send_wr->imm_data		= htonl(0x1);

    ib_com_hndl.send_wr = send_wr;
    return send_wr;

}



/**
 * \brief Sends data
 * 
 * Posts linked list of send WRs via ibv_post_send verb to the QP for transmission
 * ibv_post_send() processes the whole linked list. Pointer to next WR in current ibv_send_wr
 *
 */
void ib_post_send_queue(int number)
{
    struct ibv_wc wc;
    struct ibv_send_wr *bad_wr;

    if (ibv_post_send(ib_com_hndl.qp, ib_com_hndl.send_wr, &bad_wr) && (errno != ENOMEM))
    {
        fprintf(stderr,
                "[ERROR] Could not post send - %d (%s). Abort!\n",
                errno,
                strerror(errno));
        exit(EXIT_FAILURE);
    }


    /* wait for send WRs if CQ is full */
    int res = 0;

    do
    {        
        if ((res += ibv_poll_cq(ib_com_hndl.cq, number, &wc)) < 0)
        {
            fprintf(stderr,
                    "[ERROR] Could not poll on CQ - %d (%s). Abort!\n",
                    errno,
                    strerror(errno));
            exit(EXIT_FAILURE);
        }


    } while (res < number);

    if (wc.status != IBV_WC_SUCCESS)
    {
        fprintf(stderr,
                "###[ERROR] WR failed status %s (%d) for wr_id %lu ###\n",
                ibv_wc_status_str(wc.status),
                wc.status,
                wc.wr_id);
    }

    cleanup_send_list();

}

/**
 * \brief Prepares Receive Work Request 'ibv_recv_wr'
 *
 * This function prepares an 'ibv_recv_wr' structure for the
 * transmission of a single memory page using the IBV_WR_RDMA_WRITE verb.
 */
struct ibv_recv_wr * ib_create_recv_wr(int mr_id, struct ibv_recv_wr * next_recv_wr)
{
    /* create work request */
    struct ibv_recv_wr *recv_wr =  (struct ibv_recv_wr*)calloc(1, sizeof(struct ibv_recv_wr));
    struct ibv_sge *sge =  (struct ibv_sge*)calloc(1, sizeof(struct ibv_sge));
    uint32_t recv_buf = 0;

    /* basic work request configuration */
	recv_wr->wr_id      = 0;
    recv_wr->next       = next_recv_wr;
	recv_wr->sg_list    = sge;
	recv_wr->num_sge    = 1;

    recv_wr->sg_list->addr = (uintptr_t)&recv_buf;
    recv_wr->sg_list->length = sizeof(recv_buf);
    recv_wr->sg_list->lkey = mrs[mr_id]->lkey;
    
    ib_com_hndl.recv_wr = recv_wr;
    return recv_wr;

}

/**
 * \brief Receives data
 * length used to be a prop? 
 * Posts linked list of receive WRs via ibv_post_recv verb to the QP and waits for incoming completion queue events
 * ibv_post_recv() processes the whole linked list. Pointer to next WR in current ibv_recv_wr
 *
 */
void ib_post_recv_queue(int number)
{
	/* post recv matching IBV_RDMA_WRITE_WITH_IMM */
	struct ibv_cq *ev_cq;
	void *ev_ctx;
	struct ibv_recv_wr *bad_wr;


	if (ibv_post_recv(ib_com_hndl.qp, ib_com_hndl.recv_wr, &bad_wr) < 0) {
	        fprintf(stderr,
			"[ERROR] Could post recv - %d (%s). Abort!\n",
			errno,
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* wait for requested event */
    int res = 0;
    do
    {
        /* request notification on the event channel */
        if (ibv_req_notify_cq(ib_com_hndl.cq, 1) < 0)
        {
            fprintf(stderr,
                    "[ERROR] Could request notify for completion queue "
                    "- %d (%s). Abort!\n",
                    errno,
                    strerror(errno));
            exit(EXIT_FAILURE);
        }

        if (ibv_get_cq_event(ib_com_hndl.comp_chan, &ev_cq, &ev_ctx) < 0)
        {
            fprintf(stderr,
                    "[ERROR] Could get event from completion channel "
                    "- %d (%s). Abort!\n",
                    errno,
                    strerror(errno));
            exit(EXIT_FAILURE);
        }

        /* acknowledge the event */
        ibv_ack_cq_events(ib_com_hndl.cq, 1);

        res++;
    } while (res < number);

    cleanup_recv_list();

}

/**
 * \brief Initialises InfiniBand infastructure
 * 
 * Finds and opens IB device and corresponding port.
 * Gets port attributes and allocates protection domain
 */
int ib_init(int _device_id, uint32_t *max_msg_size)
{
    device_id = _device_id;
    /* initialize communication handler */
    memset(&ib_com_hndl, 0, sizeof(ib_com_hndl));

    struct ibv_device **device_list = NULL;
    int num_devices = 0;
    bool active_port_found = false;

    /* determine first available device */
    if ((device_list = ibv_get_device_list(&num_devices)) == NULL) {
        fprintf(stderr,
                "ERROR: Could not determine available IB devices "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
    if (num_devices == 0) {
        fprintf(stderr,
                "ERROR: Could not find any IB device. Abort!\n");
        exit(-1);
    }

    /* find device with active port */
    size_t cur_dev = device_id;

    for (; cur_dev<(size_t)num_devices; ++cur_dev){
        /* open the device context */
        if ((ib_com_hndl.ctx = ibv_open_device(device_list[cur_dev])) == NULL) {
            fprintf(stderr,
                "[ERROR] Could not open the device context "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(EXIT_FAILURE);
        }

        /* determine port count via normal device query (necessary for mlx_5) */
        if (ibv_query_device(ib_com_hndl.ctx, &ib_com_hndl.dev_attr_ex.orig_attr) < 0) {
            fprintf(stderr,
                "[ERROR] Could not query normal device attributes "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(EXIT_FAILURE);
        }

        /*determine max number of work requests*/
        max_qp_wr = (uint32_t)ib_com_hndl.dev_attr_ex.orig_attr.max_qp_wr;

        /* check all ports */
        size_t num_ports = ib_com_hndl.dev_attr_ex.orig_attr.phys_port_cnt;
        for (size_t cur_port=0; cur_port<=num_ports; ++cur_port) {
            /* query current port */
            if (ibv_query_port(ib_com_hndl.ctx, cur_port, &ib_com_hndl.port_attr) < 0){
                fprintf(stderr,
                    "[ERROR] Could not query port %lu "
                    "- %d (%s). Abort!\n",
                    cur_port,
                    errno,
                    strerror(errno));
                exit(EXIT_FAILURE);
            }

            if (ib_com_hndl.port_attr.state == IBV_PORT_ACTIVE) {
                active_port_found = 1;
                ib_com_hndl.used_port = cur_port;
                break;
            }
        }

        /* close this device if no active port was found */
        if (!active_port_found) {
               if (ibv_close_device(ib_com_hndl.ctx) < 0) {
            fprintf(stderr,
                "[ERROR] Could not close the device context "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(EXIT_FAILURE);
               }
        } else {
            break;
        }
    }

    if (!active_port_found) {
        fprintf(stderr, "[ERROR] No active port found. Abort!\n");
        exit(EXIT_FAILURE);
    }
    *max_msg_size = ib_com_hndl.port_attr.max_msg_sz;
/*    fprintf(stderr, "[INFO] Using device '%s' and port %u\n",
            ibv_get_device_name(device_list[cur_dev]),
            ib_com_hndl.used_port); */

    /* allocate protection domain */
    if ((ib_com_hndl.pd = ibv_alloc_pd(ib_com_hndl.ctx)) == NULL) {
        fprintf(stderr,
            "[ERROR] Could not allocate protection domain "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(EXIT_FAILURE);
    }
    return 0;
}

/**
 * \brief Connects IB peers. Responder side
 * 
 * Initializes local communication buffer and connects it with the remote buffer
 * Exchanges QP information
 */
int ib_connect_responder(void *memreg, int mr_id, oob_t *oob)
{
    ib_com_hndl.loc_com_buf.recv_buf = memreg;
    /* initialize loc comm buf and connect to remote */
    ib_init_com_hndl(mr_id);

    /* exchange QP information */
    oob_receive(oob, &ib_com_hndl.rem_com_buf.qp_info, sizeof(ib_qp_info_t));
    oob_send(oob, &ib_com_hndl.loc_com_buf.qp_info, sizeof(ib_qp_info_t));

    ib_com_hndl.rem_com_buf.recv_buf = (uint8_t*)ib_com_hndl.rem_com_buf.qp_info.addr;

    ib_con_com_buf();
    return 0;
}

/**
 * \brief Connects IB peers. Requester side
 * 
 * initializes local communication buffer and connects it with the remote buffer
 * exchanges QP information
 */
int ib_connect_requester(void *memreg, int mr_id, char *responder_address, oob_t *oob)
{
    ib_com_hndl.loc_com_buf.recv_buf = memreg;
    /* initialize loc comm buf and connect to remote */
    ib_init_com_hndl(mr_id);

    /* exchange QP information */
    oob_send(oob, &ib_com_hndl.loc_com_buf.qp_info, sizeof(ib_qp_info_t));
    oob_receive(oob, &ib_com_hndl.rem_com_buf.qp_info, sizeof(ib_qp_info_t));

    ib_com_hndl.rem_com_buf.recv_buf = (uint8_t*)ib_com_hndl.rem_com_buf.qp_info.addr;

    ib_con_com_buf();
    return 0;
}

/**
 * \brief Initializes InfiniBand infastructure
 * 
 * Finds and opens IB device and corresponding port.
 * Gets port attributes and allocates protection domain
 */
void ib_free_memreg(void* memreg, int mr_id, bool gpumemreg)
{
    /* free memory regions*/ 
    if (ibv_dereg_mr(mrs[mr_id]) < 0) {
        fprintf(stderr,
                "ERROR: Could not de-register  "
            "segment "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    if(gpumemreg){
        cudaFree(memreg);
    }else{
        free(memreg);
    }
}


/**
 * \brief Transmission clean up
 * 
 * Destroys: QP, completion queue and completion event channel
 */
void ib_cleanup(void)
{
    /* destroy qp */
    if (ibv_destroy_qp(ib_com_hndl.qp) < 0) {
        fprintf(stderr,
                "ERROR: Could not destroy QP "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* destroy completion queues */
    if (ibv_destroy_cq(ib_com_hndl.cq) < 0) {
        fprintf(stderr,
                "ERROR: Could not destroy CQ"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* destroy the completion event channel */
  if (ibv_destroy_comp_channel(ib_com_hndl.comp_chan) < 0) {
        fprintf(stderr,
                "ERROR: Could not destroy completion event channel"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
}

/**
 * \brief Full clean up
 * 
 * Destroys protection domain and closes device context
 */
void ib_final_cleanup(void) 
{
    /* free protection domain */
    if (ibv_dealloc_pd(ib_com_hndl.pd) < 0) {
        fprintf(stderr,
            "ERROR: Unable to de-allocate PD "
            "on device"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* close device context */
    if (ibv_close_device(ib_com_hndl.ctx) < 0) {
        fprintf(stderr,
            "ERROR: Unable to close device context "
            "on device"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
}

/**
 * \brief Prepares list of dublicate send wrs
 * 
 * For sendlist operation. Links all iterations of wrs together so only one ibv_post_send is needed
 */
int ib_prepare_send_list(void *memptr, int mr_id, size_t length, bool fromgpumem, int iterations)
{
    for (unsigned int i = 0; i < iterations; i++){
    ib_create_send_wr(memptr, length, mr_id, fromgpumem, ib_com_hndl.send_wr);
    }
}