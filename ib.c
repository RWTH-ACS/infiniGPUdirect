/*
 * Copyright 2014 Simon Pickartz,
 *           2020 Niklas Eiling
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
#define CQ_ENTRIES      (1)
#define IB_PP_WRITE_WR_ID   (2)
#define IB_PP_RECV_WR_ID    (1)
#define IB_PP_SEND_WR_ID    (0)
#define IB_PP_MTU       (IBV_MTU_2048)
#define MAX_DEST_RD_ATOMIC  (1)
#define MIN_RNR_TIMER       (1)
#define MAX_SEND_WR         (8192)  // TODO: should be
                    // com_hndl.dev_attr_ex.orig_attr.max_qp_wr
                    // fix for mlx_5 adapter
#define MAX_INLINE_DATA (0)
#define MAX_RECV_WR (1)
#define MAX_SEND_SGE    (1)
#define MAX_RECV_SGE    (1)



/*
 * Helper data types
 */
typedef struct ib_pp_qp_info {
    uint32_t qpn;
    uint16_t lid;
    uint16_t psn;
    uint32_t key;
    uint64_t addr;
} ib_pp_qp_info_t;

typedef struct ib_pp_com_buf {
    uint8_t *send_buf;
    uint8_t *recv_buf;
    ib_pp_qp_info_t qp_info;
    volatile char *new_msg;
    volatile char *send_flag;
} ib_pp_com_buf_t;


typedef struct ib_pp_com_hndl {
    struct ibv_context      *ctx;       /* device context */
    struct ibv_device_attr_ex   dev_attr_ex;    /* extended device attributes */
    struct ibv_port_attr        port_attr;  /* port attributes */
    struct ibv_pd           *pd;        /* protection domain */
    struct ibv_mr           *mr;        /* memory region */
    struct ibv_cq           *cq;        /* completion queue */
    struct ibv_qp           *qp;        /* queue pair */
    struct ibv_comp_channel     *comp_chan;     /* completion event channel */
    struct ibv_send_wr      *send_wr;   /* data send list */
    ib_pp_com_buf_t         loc_com_buf;
    ib_pp_com_buf_t         rem_com_buf;
    uint8_t             used_port;  /* port of the IB device */
    uint32_t            buf_size;   /* size of the buffer */
} ib_pp_com_hndl_t;
/*
 * Global variables
 */
uint8_t my_mask, rem_mask;

static ib_pp_com_hndl_t ib_pp_com_hndl;
static struct sockaddr_in ib_pp_server;
static int com_sock = 0;
static int listen_sock = 0;
static int device_id = 0;


static struct ibv_mr *mrs[32];
//static size_t mr_len = 0;


static oob_t oob;


/*
 * Helper functions
 */

/* synchronize client and server in case of one_sided */
void
ib_pp_barrier(int mr_id, int32_t server)
{
    if (server) {
        struct ibv_sge sg_list = {
            .addr   = 0,
            .length = 0,
            .lkey   = mrs[mr_id]->lkey
        };
        struct ibv_recv_wr recv_wr = {
            .wr_id      = IB_PP_RECV_WR_ID,
            .sg_list    = &sg_list,
            .num_sge    = 1,
        };
        struct ibv_recv_wr *bad_wr;

        if (ibv_post_recv(ib_pp_com_hndl.qp, &recv_wr, &bad_wr) < 0) {
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
            .wr_id      = IB_PP_SEND_WR_ID,
            .sg_list    = &sg_list,
            .num_sge    = 1,
            .opcode     = IBV_WR_SEND,
            .send_flags = IBV_SEND_SIGNALED,
        };
        struct ibv_send_wr *bad_wr;

        if (ibv_post_send(ib_pp_com_hndl.qp, &send_wr, &bad_wr) < 0) {
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
        if ((ne = ibv_poll_cq(ib_pp_com_hndl.cq, 1, &wc)) < 0) {
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
//do we need this? possible
size_t ib_register_memreg(void** mem_address, size_t memsize, int mr_id)
{
    /* allocate memory and register it with the protection domain */
//    int res;
    if (mem_address == NULL) return 0;

    if ((mrs[mr_id] = ibv_reg_mr(ib_pp_com_hndl.pd,
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

size_t ib_allocate_memreg(void **mem_address, size_t memsize, int mr_id, bool gpumemreg)
{
    /* allocate memory and register it with the protection domain */
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

    if ((mrs[mr_id] = ibv_reg_mr(ib_pp_com_hndl.pd,
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


/* initialize communication buffer for data transfer */
void
ib_pp_init_com_hndl(int mr_id)
{
    /* create completion event channel */
    if ((ib_pp_com_hndl.comp_chan =
        ibv_create_comp_channel(ib_pp_com_hndl.ctx)) == NULL) {
        fprintf(stderr,
            "[ERROR] Could not create the completion channel "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(EXIT_FAILURE);
    }


    /* create the completion queue */
    if ((ib_pp_com_hndl.cq = ibv_create_cq(ib_pp_com_hndl.ctx,
                               CQ_ENTRIES,
                               NULL,        /* TODO: check cq_context */
                           ib_pp_com_hndl.comp_chan,
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
        .send_cq = ib_pp_com_hndl.cq,
        .recv_cq = ib_pp_com_hndl.cq,
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
    if ((ib_pp_com_hndl.qp = ibv_create_qp(ib_pp_com_hndl.pd,
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
        .port_num       = ib_pp_com_hndl.used_port,
        .qp_access_flags    = (IBV_ACCESS_REMOTE_WRITE)
    };
    if (ibv_modify_qp(ib_pp_com_hndl.qp,
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
    ib_pp_com_hndl.loc_com_buf.qp_info.qpn  = ib_pp_com_hndl.qp->qp_num;
    ib_pp_com_hndl.loc_com_buf.qp_info.psn  = lrand48() & 0xffffff;
    ib_pp_com_hndl.loc_com_buf.qp_info.key  = mrs[mr_id]->lkey;
    ib_pp_com_hndl.loc_com_buf.qp_info.addr = (uint64_t)ib_pp_com_hndl.loc_com_buf.recv_buf;
    ib_pp_com_hndl.loc_com_buf.qp_info.lid  = ib_pp_com_hndl.port_attr.lid;
}

/* connect to remote communication buffer */
void
ib_pp_con_com_buf()
{
    /* connect QPs */
    struct ibv_qp_attr qp_attr = {
        .qp_state       = IBV_QPS_RTR,
        .path_mtu       = IBV_MTU_2048,
        .dest_qp_num        = ib_pp_com_hndl.rem_com_buf.qp_info.qpn,
        .rq_psn         = ib_pp_com_hndl.rem_com_buf.qp_info.psn,
        .max_dest_rd_atomic = MAX_DEST_RD_ATOMIC,
        .min_rnr_timer      = MIN_RNR_TIMER,
        .ah_attr        = {
            .is_global  = 0,
            .sl         = 0,
            .src_path_bits  = 0,
            .dlid       = ib_pp_com_hndl.rem_com_buf.qp_info.lid,
            .port_num   = ib_pp_com_hndl.used_port,
        }
    };
    if (ibv_modify_qp(ib_pp_com_hndl.qp,
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
    qp_attr.sq_psn          = ib_pp_com_hndl.loc_com_buf.qp_info.psn;
    qp_attr.max_rd_atomic   = 1;
    if (ibv_modify_qp(ib_pp_com_hndl.qp, &qp_attr,
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
 * \brief Prepares the an 'ibv_send_wr'
 *
 * This function prepares an 'ibv_send_wr' structure that is prepared for the
 * transmission of a single memory page using the IBV_WR_RDMA_WRITE verb.
 */
static inline struct ibv_send_wr *
prepare_send_list_elem(void)
{
    /* create work request */
    struct ibv_send_wr *send_wr =  (struct ibv_send_wr*)calloc(1, sizeof(struct ibv_send_wr));
    struct ibv_sge *sge =  (struct ibv_sge*)calloc(1, sizeof(struct ibv_sge));

    /* basic work request configuration */
    send_wr->next       = NULL;
    send_wr->sg_list    = sge;
    send_wr->num_sge    = 1;

    return send_wr;
}

static inline
void cleanup_send_list(void)
{
    struct ibv_send_wr *cur_send_wr = ib_pp_com_hndl.send_wr;
    struct ibv_send_wr *tmp_send_wr = NULL;
    while (cur_send_wr != NULL) {
        free(cur_send_wr->sg_list);
        tmp_send_wr = cur_send_wr;
        cur_send_wr = cur_send_wr->next;
        free(tmp_send_wr);
    }
}

void
ib_pp_prepare_run(void *memreg, size_t length, int mr_id, bool gpumemreg)
{
    
    //memset(ib_pp_com_hndl.loc_com_buf.send_buf, 0x42, ib_pp_com_hndl.buf_size);
    static uint8_t one = 1;
    /* create work request */
    struct ibv_send_wr *send_wr = prepare_send_list_elem();

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

    send_wr->wr.rdma.rkey = ib_pp_com_hndl.rem_com_buf.qp_info.key;
    send_wr->wr.rdma.remote_addr    = (uintptr_t)ib_pp_com_hndl.rem_com_buf.recv_buf;


    send_wr->wr_id          = IB_PP_WRITE_WR_ID;
    send_wr->opcode			= IBV_WR_RDMA_WRITE_WITH_IMM;
	send_wr->send_flags		= IBV_SEND_SIGNALED | IBV_SEND_SOLICITED;
	send_wr->imm_data		= htonl(0x1);

    ib_pp_com_hndl.send_wr = send_wr;

}


/* send data */
void
ib_pp_msg_send(ib_pp_com_hndl_t *com_hndl)
{

    /* we have to call ibv_post_send() as long as 'send_list' contains elements  */
    struct ibv_wc wc;
    struct ibv_send_wr *remaining_send_wr = NULL;
    do {
        /* send data */
        remaining_send_wr = NULL;
        if (ibv_post_send(com_hndl->qp, com_hndl->send_wr, &remaining_send_wr) && (errno != ENOMEM)) {
            fprintf(stderr,
                "[ERROR] Could not post send - %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(EXIT_FAILURE);
        }

        /* wait for send WRs if CQ is full */
        int res = 0;
        do {
            if ((res = ibv_poll_cq(com_hndl->cq, 1, &wc)) < 0) {
                fprintf(stderr,
                    "[ERROR] Could not poll on CQ - %d (%s). Abort!\n",
                    errno,
                    strerror(errno));
                exit(EXIT_FAILURE);
            }
        } while (res < 1);

        if (wc.status != IBV_WC_SUCCESS) {
            fprintf(stderr,
                "###[ERROR] WR failed status %s (%d) for wr_id %lu ###\n",
                ibv_wc_status_str(wc.status),
                wc.status,
                wc.wr_id);

        }

        com_hndl->send_wr = remaining_send_wr;
    } while (remaining_send_wr);

    cleanup_send_list();
}

/* recv data */
void
ib_pp_msg_recv(ib_pp_com_hndl_t *com_hndl, uint32_t length, int mr_id)
{
    /* request notification on the event channel */
	if (ibv_req_notify_cq(com_hndl->cq, 1) < 0) {
		fprintf(stderr,
			"[ERROR] Could request notify for completion queue "
			"- %d (%s). Abort!\n",
			errno,
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* post recv matching IBV_RDMA_WRITE_WITH_IMM */
	struct ibv_cq *ev_cq;
	void *ev_ctx;
	struct ibv_sge sg;
	struct ibv_recv_wr recv_wr;
	struct ibv_recv_wr *bad_wr;
	uint32_t recv_buf = 0;

	memset(&sg, 0, sizeof(sg));
	sg.addr	  = (uintptr_t)&recv_buf;
	sg.length = sizeof(recv_buf);
	sg.lkey	  = mrs[mr_id]->lkey;

	memset(&recv_wr, 0, sizeof(recv_wr));
	recv_wr.wr_id      = 0;
	recv_wr.sg_list    = &sg;
	recv_wr.num_sge    = 1;

	if (ibv_post_recv(com_hndl->qp, &recv_wr, &bad_wr) < 0) {
	        fprintf(stderr,
			"[ERROR] Could post recv - %d (%s). Abort!\n",
			errno,
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* wait for requested event */
	if (ibv_get_cq_event(com_hndl->comp_chan, &ev_cq, &ev_ctx) < 0) {
	        fprintf(stderr,
			"[ERROR] Could get event from completion channel "
			"- %d (%s). Abort!\n",
			errno,
			strerror(errno));
		exit(EXIT_FAILURE);
	}

	/* acknowledge the event */
	ibv_ack_cq_events(com_hndl->cq, 1);

}

int ib_init(int _device_id)
{
    device_id = _device_id;
    /* initialize com_hndl */
    memset(&ib_pp_com_hndl, 0, sizeof(ib_pp_com_hndl));

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
        if ((ib_pp_com_hndl.ctx = ibv_open_device(device_list[cur_dev])) == NULL) {
            fprintf(stderr,
                "[ERROR] Could not open the device context "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(EXIT_FAILURE);
        }

        /* determine port count via normal device query (necessary for mlx_5) */
        if (ibv_query_device(ib_pp_com_hndl.ctx, &ib_pp_com_hndl.dev_attr_ex.orig_attr) < 0) {
            fprintf(stderr,
                "[ERROR] Could not query normal device attributes "
                "- %d (%s). Abort!\n",
                errno,
                strerror(errno));
            exit(EXIT_FAILURE);
        }


        /* check all ports */
        size_t num_ports = ib_pp_com_hndl.dev_attr_ex.orig_attr.phys_port_cnt;
        for (size_t cur_port=0; cur_port<=num_ports; ++cur_port) {
            /* query current port */
            if (ibv_query_port(ib_pp_com_hndl.ctx, cur_port, &ib_pp_com_hndl.port_attr) < 0){
                fprintf(stderr,
                    "[ERROR] Could not query port %lu "
                    "- %d (%s). Abort!\n",
                    cur_port,
                    errno,
                    strerror(errno));
                exit(EXIT_FAILURE);
            }

            if (ib_pp_com_hndl.port_attr.state == IBV_PORT_ACTIVE) {
                active_port_found = 1;
                ib_pp_com_hndl.used_port = cur_port;
                break;
            }
        }

        /* close this device if no active port was found */
        if (!active_port_found) {
               if (ibv_close_device(ib_pp_com_hndl.ctx) < 0) {
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

    fprintf(stderr, "[INFO] Using device '%s' and port %u\n",
            ibv_get_device_name(device_list[cur_dev]),
            ib_pp_com_hndl.used_port);

    /* allocate protection domain */
    if ((ib_pp_com_hndl.pd = ibv_alloc_pd(ib_pp_com_hndl.ctx)) == NULL) {
        fprintf(stderr,
            "[ERROR] Could not allocate protection domain "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(EXIT_FAILURE);
    }
    return 0;
}

/* Both next functions should be solved outside this file and be removed from here */
int ib_init_oob_listener(uint16_t port)
{
    return oob_init_listener(&oob, port);
}
int ib_init_oob_sender(const char* address, uint16_t port)
{
    return oob_init_sender(&oob, address, port);
}

int ib_connect_server(void *memreg, int mr_id)
{
    ib_pp_com_hndl.loc_com_buf.recv_buf = memreg;
    /* initialize loc comm buf and connect to remote */
    ib_pp_init_com_hndl(mr_id);

    /* exchange QP information */
    oob_receive(&oob, &ib_pp_com_hndl.rem_com_buf.qp_info, sizeof(ib_pp_qp_info_t));
    oob_send(&oob, &ib_pp_com_hndl.loc_com_buf.qp_info, sizeof(ib_pp_qp_info_t));

    ib_pp_com_hndl.rem_com_buf.recv_buf = (uint8_t*)ib_pp_com_hndl.rem_com_buf.qp_info.addr;

    ib_pp_con_com_buf();
    return 0;
}

int ib_connect_client(void *memreg, int mr_id, char *server_address)
{
    ib_pp_com_hndl.loc_com_buf.recv_buf = memreg;
    /* initialize loc comm buf and connect to remote */
    ib_pp_init_com_hndl(mr_id);

    /* exchange QP information */
    oob_send(&oob, &ib_pp_com_hndl.loc_com_buf.qp_info, sizeof(ib_pp_qp_info_t));
    oob_receive(&oob, &ib_pp_com_hndl.rem_com_buf.qp_info, sizeof(ib_pp_qp_info_t));

    ib_pp_com_hndl.rem_com_buf.recv_buf = (uint8_t*)ib_pp_com_hndl.rem_com_buf.qp_info.addr;

    ib_pp_con_com_buf();
    return 0;
}

void ib_free_memreg(void* memreg, int mr_id, bool gpumemreg)
{
    /* free memory regions*/ 
//    printf("Deregistering memory ... \n");
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



void ib_cleanup(void)
{
    /* destroy qp */
//    printf("Destroying queue pair ... \n");
    if (ibv_destroy_qp(ib_pp_com_hndl.qp) < 0) {
        fprintf(stderr,
                "ERROR: Could not destroy QP "
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* destroy completion queues */
//    printf("Destroying completion queue ... \n");
    if (ibv_destroy_cq(ib_pp_com_hndl.cq) < 0) {
        fprintf(stderr,
                "ERROR: Could not destroy CQ"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* destroy the completion event channel */
  if (ibv_destroy_comp_channel(ib_pp_com_hndl.comp_chan) < 0) {
        fprintf(stderr,
                "ERROR: Could not destroy completion event channel"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
}

/*
 * Tear everything down
 */
void ib_final_cleanup(void) 
{
    /* free protection domain */
//    printf("Deallocating PD ... \n");
    if (ibv_dealloc_pd(ib_pp_com_hndl.pd) < 0) {
        fprintf(stderr,
            "ERROR: Unable to de-allocate PD "
            "on device"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }

    /* close device context */
//    printf("Closing device ... \n");
    if (ibv_close_device(ib_pp_com_hndl.ctx) < 0) {
        fprintf(stderr,
            "ERROR: Unable to close device context "
            "on device"
            "- %d (%s). Abort!\n",
            errno,
            strerror(errno));
        exit(errno);
    }
//    printf("Done!\n");
}

int ib_server_prepare(void *memptr, int mr_id, size_t length, bool togpumem)
{
    ib_connect_server(memptr, mr_id);
    ib_pp_prepare_run(memptr, length, mr_id, togpumem);
    return 0;
}

int ib_server_recv(void *memptr, int mr_id, size_t length, bool togpumem)
{
/*    printf("local address :  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, ADDR %p, KEY 0x%08x\n",
           ib_pp_com_hndl.loc_com_buf.qp_info.lid,
           ib_pp_com_hndl.loc_com_buf.qp_info.qpn,
           ib_pp_com_hndl.loc_com_buf.qp_info.psn,
           (void*)ib_pp_com_hndl.loc_com_buf.qp_info.addr,
           ib_pp_com_hndl.loc_com_buf.qp_info.key);
    printf("remote address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, ADDR %p, KEY 0x%08x\n",
           ib_pp_com_hndl.rem_com_buf.qp_info.lid,
           ib_pp_com_hndl.rem_com_buf.qp_info.qpn,
           ib_pp_com_hndl.rem_com_buf.qp_info.psn,
           (void*)ib_pp_com_hndl.rem_com_buf.qp_info.addr,
           ib_pp_com_hndl.rem_com_buf.qp_info.key); */
    ib_server_prepare(memptr, mr_id, length, togpumem);
    ib_pp_msg_recv(&ib_pp_com_hndl, length, mr_id);
    return 0;
}

int ib_client_prepare(void *memptr, int mr_id, size_t length, char *peer_node, bool fromgpumem)
{
    ib_connect_client(memptr, mr_id, peer_node);
    ib_pp_prepare_run(memptr, length, mr_id, fromgpumem);
    return 0;
}

int ib_client_send(void *memptr, int mr_id, size_t length, char *peer_node, bool fromgpumem)
{

/*    printf("local address :  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, ADDR %p, KEY 0x%08x\n",
           ib_pp_com_hndl.loc_com_buf.qp_info.lid,
           ib_pp_com_hndl.loc_com_buf.qp_info.qpn,
           ib_pp_com_hndl.loc_com_buf.qp_info.psn,
           (void*)ib_pp_com_hndl.loc_com_buf.qp_info.addr,
           ib_pp_com_hndl.loc_com_buf.qp_info.key);
    printf("remote address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, ADDR %p, KEY 0x%08x\n",
           ib_pp_com_hndl.rem_com_buf.qp_info.lid,
           ib_pp_com_hndl.rem_com_buf.qp_info.qpn,
           ib_pp_com_hndl.rem_com_buf.qp_info.psn,
           (void*)ib_pp_com_hndl.rem_com_buf.qp_info.addr,
           ib_pp_com_hndl.rem_com_buf.qp_info.key); */

    ib_client_prepare(memptr, mr_id, length, peer_node, fromgpumem);
    ib_pp_msg_send(&ib_pp_com_hndl);
}


