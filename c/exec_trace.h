#ifndef EXEC_TRACE_H
#define EXEC_TRACE_H

/*
 * Execution trace support for the patched Colibri c/colibri.c.
 *
 * This file deliberately uses the exact symbols referenced by the two patches:
 *   g_exec_trace
 *   g_exec_trace_load_ctx
 *   exec_trace_init()
 *   exec_trace_event()
 *   exec_trace_request_begin()/exec_trace_request_end()
 *   exec_trace_load_ctx_set()/exec_trace_load_ctx_clear()
 *   exec_trace_load_event()
 *   EXEC_EV_...
 *
 * Events stay in RAM for the whole request.  Output happens only from
 * exec_trace_request_end(), so trace writes do not compete with model reads.
 */

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <time.h>

#ifndef EXEC_TRACE_MAX_EVENTS
#define EXEC_TRACE_MAX_EVENTS 2000000u
#endif

typedef enum ExecTraceEventCode {
    EXEC_EV_INVALID = 0,

    EXEC_EV_REQUEST_BEGIN,
    EXEC_EV_PROMPT_BEGIN,
    EXEC_EV_GENERATION_BEGIN,
    EXEC_EV_REQUEST_END,

    EXEC_EV_TOKEN_BEGIN,
    EXEC_EV_TOKEN_END,
    EXEC_EV_LAYER_BEGIN,
    EXEC_EV_LAYER_END,

    EXEC_EV_ROUTER_BEGIN,
    EXEC_EV_ROUTER_END,
    EXEC_EV_EXPERT_SELECTED,

    EXEC_EV_PILOT_BEGIN,
    EXEC_EV_PILOT_PREDICTED,
    EXEC_EV_PILOT_END,
    EXEC_EV_PILOT_LOAD_REQUEST,

    EXEC_EV_PILOT_IO_BEGIN,
    EXEC_EV_PILOT_IO_END,
    EXEC_EV_PILOT_EXPERT_READY,

    EXEC_EV_DEMAND_LOAD_REQUEST,
    EXEC_EV_SPEC_LOAD_REQUEST,
    EXEC_EV_IO_QUEUED,
    EXEC_EV_IO_SUBMIT,
    EXEC_EV_IO_COMPLETE,
    EXEC_EV_EXPERT_READY,

    EXEC_EV_LOAD_CANCELLED,
    EXEC_EV_LOAD_SKIPPED_RESIDENT,
    EXEC_EV_QUEUE_FULL,

    EXEC_EV_PIPE_ENQUEUE_DEMAND,
    EXEC_EV_PIPE_ENQUEUE_PILOT,
    EXEC_EV_PIPE_PROMOTE_PILOT,
    EXEC_EV_PIPE_CANCEL_PILOT,
    EXEC_EV_PIPE_DISCARD_PILOT,
    EXEC_EV_PIPE_WAIT_BEGIN,
    EXEC_EV_PIPE_WAIT_END,

    EXEC_EV_WAIT_BEGIN,
    EXEC_EV_WAIT_END,

    EXEC_EV_EXPERT_COMPUTATION_BEGIN,
    EXEC_EV_EXPERT_COMPUTATION_END,

    EXEC_EV_COUNT
} ExecTraceEventCode;

typedef struct ExecTraceEvent {
    uint32_t timestamp_us;
    uint32_t request;
    uint32_t token;
    uint16_t event_code;
    int16_t layer;
    int16_t expert;
    int16_t disk;
} ExecTraceEvent;

typedef struct ExecTraceState {
    ExecTraceEvent *events;
    uint32_t capacity;
    _Atomic uint32_t count;
    _Atomic uint32_t overflow;
    _Atomic int request_active;

    uint64_t epoch_ns;
    uint32_t request;
    int enabled;

    char output_path[1024];
} ExecTraceState;

typedef struct ExecTraceLoadCtx {
    uint32_t token;
    int layer;
    int expert;
    int disk;
    int active;
} ExecTraceLoadCtx;

static ExecTraceState g_exec_trace = {0};
static _Thread_local ExecTraceLoadCtx g_exec_trace_load_ctx = {0};

static inline uint64_t exec_trace_clock_ns(void){
    struct timespec ts;
#ifdef CLOCK_MONOTONIC_RAW
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline uint32_t exec_trace_now_us(void){
    uint64_t epoch = g_exec_trace.epoch_ns;
    if(!epoch) return 0;
    return (uint32_t)((exec_trace_clock_ns() - epoch) / 1000ull);
}

static inline const char *exec_trace_event_name(uint16_t code){
    switch((ExecTraceEventCode)code){
    case EXEC_EV_REQUEST_BEGIN: return "REQUEST_BEGIN";
    case EXEC_EV_PROMPT_BEGIN: return "PROMPT_BEGIN";
    case EXEC_EV_GENERATION_BEGIN: return "GENERATION_BEGIN";
    case EXEC_EV_REQUEST_END: return "REQUEST_END";
    case EXEC_EV_TOKEN_BEGIN: return "TOKEN_BEGIN";
    case EXEC_EV_TOKEN_END: return "TOKEN_END";
    case EXEC_EV_LAYER_BEGIN: return "LAYER_BEGIN";
    case EXEC_EV_LAYER_END: return "LAYER_END";
    case EXEC_EV_ROUTER_BEGIN: return "ROUTER_BEGIN";
    case EXEC_EV_ROUTER_END: return "ROUTER_END";
    case EXEC_EV_EXPERT_SELECTED: return "EXPERT_SELECTED";
    case EXEC_EV_PILOT_BEGIN: return "PILOT_BEGIN";
    case EXEC_EV_PILOT_PREDICTED: return "PILOT_PREDICTED";
    case EXEC_EV_PILOT_END: return "PILOT_END";
    case EXEC_EV_PILOT_LOAD_REQUEST: return "PILOT_LOAD_REQUEST";
    case EXEC_EV_PILOT_IO_BEGIN: return "PILOT_IO_BEGIN";
    case EXEC_EV_PILOT_IO_END: return "PILOT_IO_END";
    case EXEC_EV_PILOT_EXPERT_READY: return "PILOT_EXPERT_READY";
    case EXEC_EV_DEMAND_LOAD_REQUEST: return "DEMAND_LOAD_REQUEST";
    case EXEC_EV_SPEC_LOAD_REQUEST: return "SPEC_LOAD_REQUEST";
    case EXEC_EV_IO_QUEUED: return "IO_QUEUED";
    case EXEC_EV_IO_SUBMIT: return "IO_SUBMIT";
    case EXEC_EV_IO_COMPLETE: return "IO_COMPLETE";
    case EXEC_EV_EXPERT_READY: return "EXPERT_READY";
    case EXEC_EV_LOAD_CANCELLED: return "LOAD_CANCELLED";
    case EXEC_EV_LOAD_SKIPPED_RESIDENT: return "LOAD_SKIPPED_RESIDENT";
    case EXEC_EV_QUEUE_FULL: return "QUEUE_FULL";
    case EXEC_EV_PIPE_ENQUEUE_DEMAND: return "PIPE_ENQUEUE_DEMAND";
    case EXEC_EV_PIPE_ENQUEUE_PILOT: return "PIPE_ENQUEUE_PILOT";
    case EXEC_EV_PIPE_PROMOTE_PILOT: return "PIPE_PROMOTE_PILOT";
    case EXEC_EV_PIPE_CANCEL_PILOT: return "PIPE_CANCEL_PILOT";
    case EXEC_EV_PIPE_DISCARD_PILOT: return "PIPE_DISCARD_PILOT";
    case EXEC_EV_PIPE_WAIT_BEGIN: return "PIPE_WAIT_BEGIN";
    case EXEC_EV_PIPE_WAIT_END: return "PIPE_WAIT_END";
    case EXEC_EV_WAIT_BEGIN: return "WAIT_BEGIN";
    case EXEC_EV_WAIT_END: return "WAIT_END";
    case EXEC_EV_EXPERT_COMPUTATION_BEGIN: return "EXPERT_COMPUTATION_BEGIN";
    case EXEC_EV_EXPERT_COMPUTATION_END: return "EXPERT_COMPUTATION_END";
    default: return "UNKNOWN";
    }
}

static inline void exec_trace_event(uint16_t event_code, uint32_t token,
                                    int layer, int expert, int disk){
    if(!g_exec_trace.enabled ||
       !atomic_load_explicit(&g_exec_trace.request_active,memory_order_relaxed)) return;
    uint32_t slot=atomic_fetch_add_explicit(&g_exec_trace.count,1,memory_order_relaxed);
    if(slot>=g_exec_trace.capacity){
        //atomic_store_explicit(&g_exec_trace.overflow,1,memory_order_relaxed);
        return;
    }
    ExecTraceEvent *e=&g_exec_trace.events[slot];
    e->timestamp_us=exec_trace_now_us();
    e->request=g_exec_trace.request;
    e->token=token;
    e->event_code=event_code;
    e->layer=(int16_t)layer;
    e->expert=(int16_t)expert;
    e->disk=(int16_t)disk;
#ifdef PIPE_DEBUG
    fprintf(stderr, "[EXEC_TRACE] ts=%u req=%u tok=%u ev=%s layer=%d expert=%d disk=%d\n",
            e->timestamp_us, e->request, e->token,
            exec_trace_event_name(event_code), (int)e->layer, (int)e->expert, (int)e->disk);
#endif
}

static inline void exec_trace_load_ctx_set(uint32_t token,int layer,int expert,int disk){
    g_exec_trace_load_ctx.token=token;
    g_exec_trace_load_ctx.layer=layer;
    g_exec_trace_load_ctx.expert=expert;
    g_exec_trace_load_ctx.disk=disk;
    g_exec_trace_load_ctx.active=1;
}

static inline void exec_trace_load_ctx_clear(void){
    g_exec_trace_load_ctx.active=0;
    g_exec_trace_load_ctx.token=0;
    g_exec_trace_load_ctx.layer=-1;
    g_exec_trace_load_ctx.expert=-1;
    g_exec_trace_load_ctx.disk=-1;
}

static inline void exec_trace_load_event(uint16_t event_code){
    if(g_exec_trace_load_ctx.active)
        exec_trace_event(event_code,g_exec_trace_load_ctx.token,
                         g_exec_trace_load_ctx.layer,g_exec_trace_load_ctx.expert,
                         g_exec_trace_load_ctx.disk);
}

static inline int exec_trace_cmp_event(const void *a,const void *b){
    const ExecTraceEvent *ea=(const ExecTraceEvent *)a;
    const ExecTraceEvent *eb=(const ExecTraceEvent *)b;
    if(ea->timestamp_us<eb->timestamp_us) return -1;
    if(ea->timestamp_us>eb->timestamp_us) return 1;
    return 0;
}

static inline void exec_trace_write_request(void){
    uint32_t n=atomic_load_explicit(&g_exec_trace.count,memory_order_relaxed);
    if(n>g_exec_trace.capacity) n=g_exec_trace.capacity;
    if(!n) return;
    qsort(g_exec_trace.events,n,sizeof(*g_exec_trace.events),exec_trace_cmp_event);

    char filename[1200];
    const char *base=g_exec_trace.output_path[0]?g_exec_trace.output_path:"exec_trace.csv";
    const char *dot=strrchr(base,'.');
    if(dot && !strcmp(dot,".csv")){
        size_t prefix=(size_t)(dot-base);
        snprintf(filename,sizeof(filename),"%.*s.request_%u.csv",(int)prefix,base,g_exec_trace.request);
    }else snprintf(filename,sizeof(filename),"%s.request_%u.csv",base,g_exec_trace.request);

    FILE *f=fopen(filename,"w");
    if(!f) return;
    fprintf(f,"timestamp_us,event_code,request,token,layer,expert,disk\n");
    for(uint32_t i=0;i<n;i++){
        const ExecTraceEvent *e=&g_exec_trace.events[i];
        fprintf(f,"%u,%s,%u,%u,%d,%d,%d\n",e->timestamp_us,
                exec_trace_event_name(e->event_code),e->request,e->token,
                (int)e->layer,(int)e->expert,(int)e->disk);
    }
    fclose(f);
}

static inline void exec_trace_init(void){
    if(g_exec_trace.events || g_exec_trace.enabled) return;
    const char *env=getenv("EXEC_TRACE");
    if(!env || !*env || !strcmp(env,"0")) return;
    g_exec_trace.capacity=EXEC_TRACE_MAX_EVENTS;
    g_exec_trace.events=(ExecTraceEvent *)malloc((size_t)g_exec_trace.capacity*sizeof(*g_exec_trace.events));
    if(!g_exec_trace.events){ perror("EXEC_TRACE malloc"); return; }
    const char *path=getenv("EXEC_TRACE_FILE");
    if(path && *path) snprintf(g_exec_trace.output_path,sizeof(g_exec_trace.output_path),"%s",path);
    else snprintf(g_exec_trace.output_path,sizeof(g_exec_trace.output_path),"exec_trace.csv");
    g_exec_trace.enabled=1;
}

static inline void exec_trace_request_begin(void){
    if(!g_exec_trace.enabled) return;
    atomic_store_explicit(&g_exec_trace.count,0,memory_order_relaxed);
    atomic_store_explicit(&g_exec_trace.overflow,0,memory_order_relaxed);
    g_exec_trace.request++;
    g_exec_trace.epoch_ns=exec_trace_clock_ns();
    atomic_store_explicit(&g_exec_trace.request_active,1,memory_order_relaxed);
    exec_trace_event(EXEC_EV_REQUEST_BEGIN,0,-1,-1,-1);
    exec_trace_event(EXEC_EV_PROMPT_BEGIN,0,-1,-1,-1);
}

static inline void exec_trace_request_end(void){
    if(!g_exec_trace.enabled ||
       !atomic_load_explicit(&g_exec_trace.request_active,memory_order_relaxed)) return;
    exec_trace_event(EXEC_EV_REQUEST_END,0,-1,-1,-1);
    atomic_store_explicit(&g_exec_trace.request_active,0,memory_order_relaxed);
    exec_trace_write_request();
}

#endif /* EXEC_TRACE_H */
