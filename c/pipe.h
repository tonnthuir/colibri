/* ============================ PIPE: load ‖ matmul ============================
 * Overlap NVMe expert-weight loads with expert matmul. A small persistent pool
 * of I/O worker pthreads runs the misses' pread (expert_load) into distinct
 * ws[] slabs and sets a per-slot `ready` flag; the MAIN thread walks the block's
 * experts in order, waiting on ready[q] only for the expert it needs right now,
 * and does all matmul_qt on itself (matmul_qt parallelises internally via OpenMP
 * and checks !omp_in_parallel() for GPU dispatch — so it must stay off the omp
 * team and off these I/O threads).
 *
 * Cross-generation safety is provided by a single generation-tagged, lock-free
 * cursor `cur = (gen<<8) | index`. The main thread is the sole writer of `gen`
 * (monotonic bump, so no ABA); workers grab jobs by CAS-advancing the low 8-bit
 * index. THE INVARIANT: a worker reads eids[i]/layer only AFTER its winning CAS,
 * and that CAS's comparand carries the generation — so if `cur`'s gen advanced
 * (a new batch was published), the CAS fails and the worker re-reads, seeing the
 * new generation. A straggler preempted anywhere (wake gap, post-cursor) can
 * therefore NEVER grab a wrong-generation job or read torn batch state: its
 * first act is a gen-checked CAS. dispatch publishes all batch state with
 * relaxed stores and then RELEASE-stores `cur`; each worker ACQUIRE-loads `cur`,
 * so the ready[] reset + eids[]/njobs/layer are visible before any worker acts.
 *
 * PipePool's batch arrays are retained for compatibility (Metal, URING),
 * but the worker pool now reads from PipeQueue instead. Each PipeJob is a persistent
 * object created at dispatch and freed after completion.
 *
 * The per-expert pipe_queue_wait(job) in the matmul loop makes every grabbed job
 * complete before the block ends. The mutex/condvar exist ONLY to park/wake idle workers,
 * never for correctness. Gated behind PIPE=1; OFF => the original blocking-load +
 * serial-matmul path runs byte-identically. */
static int g_pipe=0;      /* PIPE=1: async expert-load pipeline. Default ON for Windows
                           * (parsed in main: getenv("PIPE")?:1 on _WIN32, :0 elsewhere).
                           * Keeps expert pread off the forward-pass thread so loads overlap
                           * the matmul. PIPE=0 opts back into the blocking serial path. */
static int g_pipe_nw=8;   /* PIPE_WORKERS=n: I/O worker threads (disk-parallel reads) */
static int g_uring=0;     /* URING=1: Linux io_uring load/completion backend; implies PIPE */
static int g_pipe_block=0;/* COLI_PIPE_BLOCK=1: pipe_wait blocca su una condvar invece dello
                           * spin sched_yield (default OFF = spin byte-identico). EN: a yield
                           * storm on the main thread fights the OpenMP team for cycles during
                           * multi-ms loads; the condvar wake costs ~5us against reads that
                           * cost 0.5-3ms (#159). Pthread pool only: the URING backend has no
                           * waiter spin to replace. */
/* PIPE_WORKERS>0 esplicito nell'env implica PIPE=1: dimensionare il pool
 * dichiara l'intento di usarlo (una campagna intera l'ha impostato con la
 * pipe spenta senza accorgersene). EN: fires ONLY when PIPE is unset in the
 * env AND the platform default left the pipe off (on _WIN32 it already
 * defaults to 1) AND PIPE_WORKERS parses positive — the internal default of
 * 8 does not count, PIPE_WORKERS=0/empty does not, and an explicit PIPE=0
 * always wins. */
static int pipe_workers_imply_pipe(const char *pipe_env, const char *pw_env, int pipe_now){
    return !pipe_now && !pipe_env && pw_env && atoi(pw_env)>0;
}
typedef struct {
    _Atomic uint64_t cur;                         /* (gen<<8)|index; gen main-only, index 0..njobs (≤64) */
    _Atomic int njobs;                            /* current batch job count */
    _Atomic int eids[64];                         /* current batch expert ids */
    _Atomic uint32_t tokens[64];                  /* originating token for trace */
    _Atomic int layer;                            /* current batch layer */
    _Atomic int ready[64];                        /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;        /* ONLY for parking/waking idle workers */
    pthread_cond_t cv_done;                       /* COLI_PIPE_BLOCK: signals ready[] transitions */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;


/* Job kind: what triggered this load */
typedef enum {
    PIPE_JOB_DEMAND,     /* current-layer miss from routing */
    PIPE_JOB_PILOT,      /* speculative future-layer prediction */
    PIPE_JOB_CANCELLED   /* pilot job that has been cancelled */
} PipeJobKind;

/* Job lifecycle states. Splits "the load is over" from "the slot is
 * free again": completing expert_load() stops at READY, and only the consumer
 * that used the loaded expert moves the job through CONSUMING to RELEASED.
 *
 *   QUEUED -> RUNNING [owns WS] -> READY [owns WS] -> CONSUMING [owns WS] -> RELEASED [slot FREE]
 *
 */
typedef enum {
    PIPE_JOB_QUEUED,     /* enqueued, not yet grabbed by a worker */
    PIPE_JOB_RUNNING,    /* worker has claimed it, I/O in progress */
    PIPE_JOB_READY,      /* load complete, data in ws buffer, WS slot still owned */
    PIPE_JOB_CONSUMING,  /* a consumer claimed the READY expert, WS slot still owned */
    PIPE_JOB_RELEASED    /* consumer finished: the WS slot has been handed back */
} PipeJobState;

/* The states in which a job's I/O is still outstanding — the only states a
 * consumer has to wait out. Everything else is terminal, with or without a slot
 * left to resolve. */
static inline int pipe_job_state_inflight(PipeJobState s){
    return s == PIPE_JOB_QUEUED || s == PIPE_JOB_RUNNING;
}
/* The states in which the loaded expert is still owned by the job and therefore
 * readable by its consumer: READY (I/O done, nobody claimed it yet) or CONSUMING
 * (claimed, not released yet). */
static inline int pipe_job_state_loaded(PipeJobState s){
    return s == PIPE_JOB_READY || s == PIPE_JOB_CONSUMING;
}

/* ---- WS-slot ownership (Step 1) --------------------------------------
 * There are 64 physical m->ws[] slots. A slot is named by a generation-
 * qualified token:  token = (generation << 6) | slot.
 * The authoritative generation lives ONLY in PipeQueue.ws_tab[slot].gen and is
 * mutated only under PipeQueue.mx; PipeJob.ws_tok is a copy kept so a consumer
 * can resolve its pointer after the wait. Generations start at 1, so gen==0 is
 * never a valid token: a calloc/memset-zeroed handle therefore decodes to
 * something provably invalid instead of silently aliasing slot 0. */
typedef uint64_t WsToken;
#define WS_SLOT_BITS 6
#define WS_NSLOTS    (1<<WS_SLOT_BITS)
#define PIPE_DEMAND_RESERVED_SLOTS 24
#define WS_NONE      ((WsToken)~0ull)
#define WS_TOKEN(gen,slot)  (((WsToken)(gen) << WS_SLOT_BITS) | (WsToken)(slot))
#define WS_SLOT(t)  ((int)((t) & ((1ull<<WS_SLOT_BITS)-1ull)))
#define WS_GEN(t)   ((uint64_t)((t) >> WS_SLOT_BITS))
static inline int ws_tok_valid(WsToken t){ return t!=WS_NONE && WS_GEN(t)!=0 && WS_SLOT(t)<WS_NSLOTS; }

/* Persistent job object — created at dispatch, freed on completion */
typedef struct PipeJob {
    int layer;                          /* target layer (also serves as priority) */
    int eid;                            /* expert id */
    _Atomic PipeJobKind kind;           /* DEMAND, PILOT, or CANCELLED */
    WsToken ws_tok;                     /* owned WS slot token: WS_NONE until claimed,
                                         * kept across READY, cleared again on release */
    _Atomic PipeJobState state;         /* lifecycle state machine */
    uint32_t token;                     /* trace token */
    int job_id;                         /* unique monotonically increasing id */
    Model *pool_m;                      /* owning model — needed by worker after dequeue */
    int  dst_cache_slot;                /* predetermined cache slot for promotion; -1 = no promotion */
} PipeJob;

/* ---- compute-sequence entry ----
 * One entry per deduplicated routed expert in the current block, in routing
 * order (same order as uniq[]).  Each entry retains everything the execution
 * path needs — no reconstruction from the queue later. */
typedef struct {
    int  expert_id;                     /* expert id (also the index into uniq[]) */
    int  cached;                        /* 1 = already resident (pin/ecache), 0 = needs I/O */
    int  needs_io;                      /* 1 = this expert was a MISS in this block. Set from
                                         * qof[] at resolution, so it stays true for a PIPE=0
                                         * miss (loaded synchronously) and for a miss whose
                                         * enqueue was rejected as a duplicate. */
    PipeJob *job;                       /* PipeJob for I/O-bound experts; NULL when cached,
                                         * and also NULL for a PIPE=0 miss */
    ESlot *eslot;                       /* slot the expert is ALREADY in: a pin/ecache hit, or
                                         * the positional ws[] slot of a PIPE=0 load. NULL while
                                         * I/O is pending — under PIPE the slot is named by
                                         * job->ws_tok, which does not exist until a claim. */
    int  consumed;                      /* 1 = CPU has already executed this expert */
    int  ref;                           /* 1 = the consumer holds the in-flight eslot ref that
                                         * pipe_seq_find_ready() took. moe_seq_finish() drops it.
                                         * Always paired, including the entries that are skipped
                                         * without computing. */
    WsToken scratch;                    /* slot borrowed by the pipe_scratch_load() fallback for
                                         * this entry; WS_NONE normally. Released by moe_seq_finish(). */
    int  nr;                            /* number of tokens (positions) that use this expert */
    int  rows[64];                      /* token positions (row indices into x[]) that use this expert */
    float rw[64];                       /* routing weight for each position in rows[] */
    int  dst_cache_slot;                /* predetermined cache slot for promotion; -1 = no promotion */
} PipeSeqEntry;

/* Per-layer FIFO buckets. layer == priority: lower layer numbers served first.
 * Max layers = 256 (matches g_pilot_inflight[256] and g_cur_moe_layer range).
 * A job stays in its bucket until a worker CAS-grabs the head.
 * Cancellation is logical: the job stays in the list but is skipped on grab. */
#define PIPE_MAX_LAYERS 256

typedef struct PipeJobNode {
    PipeJob *job;
    struct PipeJobNode *next;
} PipeJobNode;

typedef struct {
    PipeJobNode *head;
    PipeJobNode *tail;
    int count;                          /* jobs in this bucket */
} PipeLayerBucket;

/* One m->ws[] slot's ownership record. Named type so the allocator can hold a
 * pointer to it; see PipeQueue.ws_tab. */
typedef struct {
    uint64_t gen;                       /* ownership incarnation, starts at 1 */
    uint8_t  owned;                     /* 0 = free, 1 = owned */
    uint8_t  kind;                      /* PipeJobKind of the current owner */
    int      owner_job;                 /* job_id of the current owner, -1 when free */
} WsSlotEnt;

/* Unified priority queue — replaces PipePool's batch arrays for worker scheduling */
typedef struct {
    PipeLayerBucket buckets[PIPE_MAX_LAYERS];
    _Atomic int total_jobs;             /* total queued jobs (for metrics) */

    /* Lookup table: (layer, eid) → PipeJob* for promotion/dedup */
    PipeJob **lookup;                   /* allocated to size: n_layers * n_experts */
    int lookup_layers;                  /* model->c.n_layers + 1 */
    int lookup_experts;                 /* model->c.n_experts */

    /* Mutex for queue mutations (enqueue, cancel, promote, lookup) */
    pthread_mutex_t mx;

    /* Condition variable to wake idle workers */
    pthread_cond_t cv;

    /* Condition variable for READY-state notifications.
     * Workers signal this when a job transitions to READY (I/O complete,
     * data available for the CPU consumer).
     * The CPU consumer waits on this when scanning the compute sequence
     * and no expert is currently READY — standard condvar pattern:
     *   lock(mx); while (!predicate) pthread_cond_wait(cv_ready, mx); ...
     * This replaces cv_done with a semantically correct name. */
    pthread_cond_t cv_ready;

    /* Monotonic job ID counter */
    int next_job_id;

    /* Worker notification: how many workers are currently idle */
    _Atomic int idle_workers;

    /* ---- 64 WS-slot resources (Step 1) ----
     * ws_tab[s].gen is the single authoritative generation for slot s; state
     * is FREE/OWNED; ws_free is the number of FREE slots. All three are
     * mutated only with mx held, so claim predicates and the worker park
     * predicate can be tested against them without a second lock. */
    WsSlotEnt ws_tab[WS_NSLOTS];
    int ws_free;                        /* FREE slots, guarded by mx */
    int inited;                         /* set at the end of pipe_queue_init */
    unsigned long ws_acquire_n, ws_release_n, ws_release_bad_n;
    int ws_peak_used;                   /* max 64-ws_free observed */
} PipeQueue;

static PipeQueue g_pipe_queue;

/* Current layer context — set by moe() before dispatch, read by workers */
static _Atomic int g_cur_pipe_layer = 0;

/* ---- PipeQueue inline helpers ---- */

static inline PipeJob *pipe_queue_lookup(int layer, int eid) {
    if (layer < 0 || eid < 0) return NULL;
    if (layer >= g_pipe_queue.lookup_layers || eid >= g_pipe_queue.lookup_experts) return NULL;
    return g_pipe_queue.lookup[(size_t)layer * g_pipe_queue.lookup_experts + eid];
}

static inline void pipe_queue_set_lookup(int layer, int eid, PipeJob *job) {
    if (layer < 0 || eid < 0) return;
    if (layer >= g_pipe_queue.lookup_layers || eid >= g_pipe_queue.lookup_experts) return;
    g_pipe_queue.lookup[(size_t)layer * g_pipe_queue.lookup_experts + eid] = job;
}

static inline void pipe_queue_clear_lookup(int layer, int eid) {
    if (layer < 0 || eid < 0) return;
    if (layer >= g_pipe_queue.lookup_layers || eid >= g_pipe_queue.lookup_experts) return;
    g_pipe_queue.lookup[(layer * g_pipe_queue.lookup_experts + eid)] = NULL;
}

/* ---- WS-slot allocator ---------------------------------------
 * pipe_ws_acquire MUST be called with g_pipe_queue.mx held (claim does so), so
 * that "a slot was minted" and "a job entered RUNNING_IO" are one atomic
 * decision. pipe_ws_release takes mx itself. Neither ever blocks. */

/* The invariants are checked, not just documented.
 * The properties checked: every slot is owned OR free (never both, never
 * neither — the counter must match the table), and no generation is ever 0,
 * which is what stops a zero-initialised handle aliasing slot 0. */
static int ws_tab_consistent(void){
    int free_n = 0;
    for(int s = 0; s < WS_NSLOTS; s++){
        const WsSlotEnt *t = &g_pipe_queue.ws_tab[s];
        if(t->gen == 0) return 0;
        if(!t->owned) free_n++;
    }
    return free_n == g_pipe_queue.ws_free;
}
#define WS_CHECK() assert(ws_tab_consistent())

/* Take one FREE slot on behalf of `owner_job` (pass -1 for a transient
 * consumer-side scratch borrow, which has no PipeJob). Returns 1 and writes
 * *out_tok on success, or 0 with *out_tok==WS_NONE when all 64 slots are
 * owned. Pilot jobs are restricted to slots >= PIPE_DEMAND_RESERVED_SLOTS;
 * demand jobs may acquire any free slot. */
static int pipe_ws_acquire(PipeJobKind kind, int owner_job, WsToken *out_tok){
    /* Pilot jobs may only acquire non-reserved slots.
     * Demand jobs may acquire any available slot. */
    int s_start = (kind == PIPE_JOB_PILOT) ? PIPE_DEMAND_RESERVED_SLOTS : 0;
    for(int s=s_start;s<WS_NSLOTS;s++){
        WsSlotEnt *t = &g_pipe_queue.ws_tab[s];
        if(t->owned) continue;
        t->owned = 1;
        t->kind  = (uint8_t)kind;
        t->owner_job = owner_job;
        g_pipe_queue.ws_free--;
        g_pipe_queue.ws_acquire_n++;
        int used = WS_NSLOTS - g_pipe_queue.ws_free;
        if(used > g_pipe_queue.ws_peak_used) g_pipe_queue.ws_peak_used = used;
        WsToken tok = WS_TOKEN(t->gen, s);
        assert(ws_tok_valid(tok));        /* requirement: one owner, and a mintable token */
        WS_CHECK();
        if(out_tok) *out_tok = tok;
#ifdef PIPE_WS_DEBUG
        fprintf(stderr, "[PIPE_WS] acquire owner=%d slot=%d gen=%llu free=%d\n",
                owner_job, s, (unsigned long long)t->gen, g_pipe_queue.ws_free);
#endif
        return 1;
    }
    if(out_tok) *out_tok = WS_NONE;
    return 0;
}

/* Give a slot back. Validates the generation against the authoritative table
 * entry first, so a stale or duplicated token CANNOT free a slot somebody else
 * owns: the call is rejected and reported instead. On success the generation is
 * bumped, so every token that referred to the previous incarnation becomes
 * invalid. Workers are woken because slot capacity, not just queue depth, can
 * unblock a queued job. Returns 0 on success, -1 if the token was rejected. */
/* Release with g_pipe_queue.mx ALREADY held (claim's rollback path). */
static int pipe_ws_release_locked(WsToken tok){
    if(!ws_tok_valid(tok)){
        g_pipe_queue.ws_release_bad_n++;
        fprintf(stderr, "[PIPE_WS] REJECTED release of non-token %llu\n",
                (unsigned long long)tok);
        return -1;
    }
    int s = WS_SLOT(tok);
    WsSlotEnt *t = &g_pipe_queue.ws_tab[s];
    if(!t->owned || t->gen != WS_GEN(tok)){
        g_pipe_queue.ws_release_bad_n++;
        fprintf(stderr, "[PIPE_WS] REJECTED release slot=%d tok-gen=%llu table-gen=%llu owned=%d (double release or stale token)\n",
                s, (unsigned long long)WS_GEN(tok), (unsigned long long)t->gen, (int)t->owned);
        return -1;
    }
#ifdef PIPE_WS_DEBUG
    int owner = t->owner_job;
#endif
    t->owned = 0;
    t->kind = 0;
    t->owner_job = -1;
    /* Bump the incarnation. 58 generation bits cannot wrap in practice; if it
     * ever did we skip 0 (which means "no token") rather than alias silently. */
    if(++t->gen >= (1ull<<58)){
        fprintf(stderr, "[PIPE_WS] slot=%d generation reached 2^58, wrapping to 1\n", s);
        t->gen = 1;
    }
    g_pipe_queue.ws_free++;
    g_pipe_queue.ws_release_n++;
    WS_CHECK();                           /* requirement: freed exactly once, counter agrees */
    /* A released slot is work for the parked workers even if nothing was
     * enqueued, so signal the worker park cv while mx is still held. */
    pthread_cond_broadcast(&g_pipe_queue.cv);
#ifdef PIPE_WS_DEBUG
    fprintf(stderr, "[PIPE_WS] release job=%d slot=%d new-gen=%llu free=%d\n",
            owner, s, (unsigned long long)t->gen, g_pipe_queue.ws_free);
#endif
    return 0;
}

/* Release a slot a consumer is finished with. Takes the queue mutex itself. */
static int pipe_ws_release(WsToken tok){
    pthread_mutex_lock(&g_pipe_queue.mx);
    int rc = pipe_ws_release_locked(tok);
    pthread_mutex_unlock(&g_pipe_queue.mx);
    return rc;
}

/* requirement: "every acquired slot is eventually released" has to be
 * observable in a real run, not only inside the unit test, and ws_release_bad_n
 * was previously counted but never shown. Registered with atexit() by
 * pipe_queue_init(), so a benchmark or a serve prints the accounting when it
 * exits normally. Silent while the allocator was never touched (PIPE=0 runs). */
static void pipe_ws_dump_stats(void){
    unsigned long a = g_pipe_queue.ws_acquire_n, r = g_pipe_queue.ws_release_n;
    unsigned long bad = g_pipe_queue.ws_release_bad_n;
    if(!a && !r && !bad) return;
    fprintf(stderr, "[PIPE_WS] slots: acquire=%lu release=%lu rejected=%lu peak_used=%d\n",
            a, r, bad, g_pipe_queue.ws_peak_used);
    if(a != r)
        fprintf(stderr, "[PIPE_WS] WARNING: %lu slot%s acquired but %lu released — %d still owned at exit\n",
                a, a==1?"":"s", r, WS_NSLOTS - g_pipe_queue.ws_free);
}
static inline int pipe_ws_free(void){ return g_pipe_queue.ws_free; }

/* Forward declarations for functions called by pipe_enqueue_demand below */
static int pipe_queue_enqueue(PipeJob *job);
static PipeJob *pipe_enqueue_demand(Model *m, int layer, int eid, uint32_t token);

/* Enqueue a single demand job. The parameters describe the expert the queue has
 * to fetch and nothing else: there is no positional argument that could be read
 * as a predetermined m->ws[] slot — WS allocation belongs to the queue.
 * Allocates and initializes a PipeJob, then enqueues it into the PipeQueue.
 * Returns PipeJob* on success (caller owns the pointer; stores in qjob[]).
 * Returns NULL on failure (allocation failed or duplicate for same layer/eid). */
static PipeJob *pipe_enqueue_demand(Model *m, int layer, int eid, uint32_t token) {
    PipeJob *job = (PipeJob *)calloc(1, sizeof(PipeJob));
    if (!job) {
        fprintf(stderr, "[PIPE] PipeJob allocation failed\n");
        return NULL;
    }

    job->layer = layer;
    job->eid = eid;
    atomic_store_explicit(&job->kind, PIPE_JOB_DEMAND, memory_order_release);
    job->pool_m = m;
    job->token = token;
    /* A queued job owns no WS slot The worker acquires the slot when
     * it claims the job (pipe_queue_claim -> pipe_ws_acquire) */
    job->ws_tok = WS_NONE;
    job->dst_cache_slot = -1;         /* -1 = no promotion target (set later by promotions calc) */
    atomic_store_explicit(&job->state, PIPE_JOB_QUEUED, memory_order_relaxed);

    /* next_job_id must be assigned under the queue mutex to prevent races. */
    pthread_mutex_lock(&g_pipe_queue.mx);
    job->job_id = g_pipe_queue.next_job_id++;
    pthread_mutex_unlock(&g_pipe_queue.mx);

    int rc = pipe_queue_enqueue(job);
    if (rc != 0) {
        /* Duplicate — lookup table already has an entry for (layer, eid).
         * Free the newly allocated job; the existing entry remains. */
        free(job);
        return NULL;
    }

    return job;  /* Caller owns the pointer; store in qjob[j]. */
}

/* Non-blocking probe of a job's completion status. Returns 1 if READY, 0 otherwise. */
static inline int pipe_job_ready(PipeJob *job) {
    if (!job) return 0;
    PipeJobState s = atomic_load_explicit(&job->state, memory_order_acquire);
    return (s == PIPE_JOB_READY) ? 1 : 0;
}

/* Wait for a specific job to complete and RETURN the workspace slot it loaded,
 * resolved from the job's WS token. The router does not know
 * which m->ws[] entry an expert landed in (the I/O worker picks it when it
 * claims the job), so resolving here is the only way a consumer can learn it —
 * and returning the pointer is what turns a forgotten resolution into a
 * compile error instead of a dereference of the wrong expert.
 * Returns NULL when the job was cancelled or discarded: the caller then owns no
 * slot and must not read one.
 * Waiting is NOT consumption. This resolves the slot of a job whose I/O
 * is over — READY, or CONSUMING because this same consumer already claimed it
 * (per-expert waits may be called multiple times) — but it never releases
 * anything. pipe_job_release() is the only operation that hands a job's slot back.
 * Job lifetime is managed per-expert: each consumer releases its own job after use.
 * (Batch-wide guaranteed drain has been retired) */
static ESlot *pipe_wait_job(Model *m, PipeJob *job) {
    if (!job) return NULL;

    int spins = 0;
    while (1) {
        PipeJobState s = atomic_load_explicit(&job->state, memory_order_acquire);
        if (pipe_job_state_loaded(s)){
#ifdef PIPE_WS_DEBUG
            if (spins > 10) fprintf(stderr, "[PIPE_WS_DEBUG] WAIT DONE job=%d L%d E%d state=%d spins=%d\n",
                    job->job_id, job->layer, job->eid, (int)s, spins);
#endif
            WsToken t = job->ws_tok;
            if(!ws_tok_valid(t)) return NULL;
            return &m->ws[WS_SLOT(t)];
        }
        if (s == PIPE_JOB_RELEASED) {
            /* Terminal, and the slot is already back in the pool: there is nothing
             * left to resolve. No wait site runs after the block's release site
             * today; answering NULL keeps a wait placed there from spinning on a
             * state that can never change again. */
            return NULL;
        }
        /* s is QUEUED or RUNNING */
#ifdef PIPE_WS_DEBUG
        if (spins == 0) fprintf(stderr, "[PIPE_WS_DEBUG] WAIT START job=%d L%d E%d state=%d ws_tok=%llu ws_free=%d total=%u\n",
                job->job_id, job->layer, job->eid, (int)s,
                (unsigned long long)job->ws_tok, g_pipe_queue.ws_free,
                (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire));
#endif
        spins++;
        if (g_pipe_block) {
            /* Condvar wait on the ready condition. The predicate is "still in flight"
             * rather than "not READY", because a completed job can also
             * be found CONSUMING or RELEASED here, and a waiter that slept through
             * either transition must wake rather than hang.
             * Using cv_ready ensures the worker's READY-mark + signal are atomic with
             * respect to the CPU consumer's check+wait, following the standard
             * condvar pattern: lock → while(!pred) wait → perform transition → unlock. */
            pthread_mutex_lock(&g_pipe_queue.mx);
            PipeJobState w = atomic_load_explicit(&job->state, memory_order_acquire);
            while (pipe_job_state_inflight(w)) {
                pthread_cond_wait(&g_pipe_queue.cv_ready, &g_pipe_queue.mx);
                w = atomic_load_explicit(&job->state, memory_order_acquire);
            }
            WsToken t = job->ws_tok;
            pthread_mutex_unlock(&g_pipe_queue.mx);
            if(!pipe_job_state_loaded(w)) return NULL;
            return ws_tok_valid(t) ? &m->ws[WS_SLOT(t)] : NULL;
        }
        /* Spin with yield */
        sched_yield();
    }
}

/* ---- The explicit consumption and release operations --------------
 * A job's WS slot now spans the whole time its contents matter, not just the
 * read: expert_load() completing only makes the job READY, so no later claim can
 * re-issue the slot a consumer is still reading. The consumer closes the
 * lifecycle itself — claim with pipe_job_consume(), give it back with
 * pipe_job_release(). */

/* Claim a READY job for the calling consumer: READY -> CONSUMING, WS slot still
 * owned by the job, and return the expert it loaded.
 * The CAS makes the claim single-winner without the queue mutex: the transition
 * is one word of the job, and neither a bucket nor the slot table depends on
 * it — pipe_ws_release() stays the only place a slot becomes FREE.
 * Deliberately idempotent: per-expert paths may call this more than once
 * (Metal drain, per-expert wait), and a job already CONSUMING belongs to the
 * consumer that is asking, so the same slot comes back every time.
 * Batch-wide guaranteed drain has been retired (Step 10); each consumer
 * releases its own job after use. A job that resolves to no slot
 * (cancelled, discarded, already released) yields NULL exactly as
 * pipe_wait_job() does. */
static ESlot *pipe_job_consume(Model *m, PipeJob *job) {
    ESlot *e = pipe_wait_job(m, job);
    if (!e) return NULL;
#ifdef PIPE_WS_DEBUG
    fprintf(stderr, "[PIPE_WS_DEBUG] CONSUME job=%d L%d E%d slot=%d state=%d ws_free=%d\n",
            job->job_id, job->layer, job->eid, WS_SLOT(job->ws_tok),
            (int)atomic_load_explicit(&job->state, memory_order_relaxed),
            g_pipe_queue.ws_free);
#endif
    PipeJobState expected = PIPE_JOB_READY;
    atomic_compare_exchange_strong_explicit(&job->state, &expected, PIPE_JOB_CONSUMING,
                                            memory_order_acq_rel, memory_order_acquire);
    return e;
}

/* Hand a job's WS slot back: the only operation that ends a job's ownership of
 * a slot (the allocator entry points remain for the claim rollback and
 * for consumer scratch loans, which have no PipeJob).
 * ws_tok is taken before the release, so a slot is handed back exactly once no
 * matter how often this is called, and the job lands in the terminal RELEASED
 * state, where it can never resolve to a slot again. The state is checked rather
 * than trusted: a job still in flight would mean giving back a buffer a worker
 * is writing, which is the bug this step exists to prevent. The slot goes back
 * either way — the caller is the block's release site, and withholding capacity
 * there would strand every later claim. */
static void pipe_job_release(PipeJob *job) {
    if (!job) return;
    WsToken tok = job->ws_tok;
    job->ws_tok = WS_NONE;
    if(!ws_tok_valid(tok)) return;        /* never owned a slot: nothing to hand back */
    PipeJobState s = atomic_load_explicit(&job->state, memory_order_acquire);
    if(!pipe_job_state_loaded(s))
        fprintf(stderr, "[PIPE] job %d (L%d,E%d) released from state %d, its I/O was never awaited\n",
                job->job_id, job->layer, job->eid, (int)s);
    atomic_store_explicit(&job->state, PIPE_JOB_RELEASED, memory_order_release);
    pipe_ws_release(tok);
}

/* Consumer-side fallback load. A demand wait that resolves to
 * no slot — the job was cancelled while queued, was discarded, or was never
 * enqueued because (layer,eid) was already queued fell through to a blocking load.
 * The old positional ws[] ordinal can belong to any job now, so the consumer
 * borrows a slot from the allocator and hands it back at the end of the block (see
 * the release site in moe()).
 * Exhausted capacity is a WAIT, not an error; only if no slot appears within
 * WS_SCRATCH_WAIT_S does the caller get NULL, and it must then skip that expert
 * with a warning — never kill the process. */
#define WS_SCRATCH_WAIT_S 5.0
/* Take a slot, waiting up to WS_SCRATCH_WAIT_S for capacity to come back (a
 * release broadcasts on the same mutex it allocates under). Returns 0 only on
 * timeout. The caller must not hold the queue mutex. */
static int pipe_ws_acquire_wait(PipeJobKind kind, int owner_job, WsToken *out_tok){
    *out_tok = WS_NONE;
    double t0 = now_s();
    int warned = 0;
    for(;;){
        pthread_mutex_lock(&g_pipe_queue.mx);
        int got = pipe_ws_acquire(kind, owner_job, out_tok);
        pthread_mutex_unlock(&g_pipe_queue.mx);
        if(got) return 1;
        if(!warned){ warned = 1;
            fprintf(stderr, "[PIPE_WS] pool empty; waiting up to %.0fs for a slot\n", WS_SCRATCH_WAIT_S); }
        if(now_s()-t0 > WS_SCRATCH_WAIT_S){
            fprintf(stderr, "[PIPE_WS] still no slot after %.0fs\n", WS_SCRATCH_WAIT_S);
            return 0;
        }
        sched_yield();
    }
}

/* The fall-through itself: load (layer,eid) synchronously into a slot this
 * consumer owns. *out_tok is the loan, released by the block's release site. */
static ESlot *pipe_scratch_load(Model *m, int layer, int eid, WsToken *out_tok){
    WsToken tok;
    if(!pipe_ws_acquire_wait(PIPE_JOB_DEMAND, -1, &tok)) return NULL;
    ESlot *e = &m->ws[WS_SLOT(tok)];
    expert_load(m, layer, eid, e, 1, 1);   /* demand=1: the moe miss path, as the old fall-through used */
    *out_tok = tok;
    return e;
}

/* Hand back a consumer scratch slot. WS_NONE ("nothing borrowed") is a no-op. */
static void pipe_scratch_release(WsToken tok){
    if(ws_tok_valid(tok)) pipe_ws_release(tok);
}

/* Consumer-side wrapper: wait for a demand job and bind its loaded slot. When
 * the job yields none, falls back to pipe_scratch_load() and reports the
 * borrowed token through *scratch (WS_NONE when the job's own slot was used,
 * so the caller releases exactly what was borrowed). Returns NULL only if even
 * the fallback could not obtain a slot; the caller must then skip the expert. */
static ESlot *pipe_wait_slot(Model *m, PipeJob *job, int layer, int eid, WsToken *scratch){
    /* A consumer that asks for the slot takes ownership of the loaded
     * expert with it (READY -> CONSUMING); the slot comes back only through
     * pipe_job_release() at the block's release site. */
    ESlot *e = pipe_job_consume(m, job);
    *scratch = WS_NONE;
    if(e) return e;
    return pipe_scratch_load(m, layer, eid, scratch);
}

/* ---- Queue operations (declarations) ---- */
static void pipe_queue_init(Model *m);
/* pipe_queue_enqueue forward decl is above pipe_enqueue_demand */
static PipeJob *pipe_queue_claim(Model *m, int current_layer);
static void pipe_queue_cancel(PipeJob *job);
static void pipe_queue_promote(PipeJob *job, PipeJobKind new_kind);
static void pipe_queue_wait(PipeJob *job);

/* ---- Opportunistic READY selection from the compute sequence ----
 * Scans the active layer's compute sequence (under g_pipe_queue.mx) for the
 * first unconsumed entry whose expert is READY.  Cached entries are always
 * immediately ready.  When a READY I/O entry is found its job is transitioned
 * READY → CONSUMING before the function returns.  If no entry is ready the
 * caller waits on cv_ready and rescans (handles spurious wakeups via the
 * standard while-loop pattern).  Returns NULL only when the sequence is
 * empty (seq_n == 0). */
static PipeSeqEntry *pipe_seq_find_ready(Model *m, PipeSeqEntry *seq, int seq_n);

/* Claim a job from the persistent priority queue.
 * Scans from current_layer forward (modular wrap), limited by CLAIM_SCAN_DEPTH.
 * Lower layer numbers = higher priority, served first. */
#define CLAIM_SCAN_DEPTH 5

/* Bucket list surgery. The bucket is a singly-linked FIFO with a tail pointer
 * that pipe_queue_enqueue() appends through, so ANY removal has to keep tail
 * honest: the walk in pipe_queue_claim() can reach a middle or tail node (a
 * logically cancelled node ahead of it), and freeing a node that tail still
 * points at turns the next enqueue into a write through freed memory.
 * `link` is the address that currently holds node (bucket->head, or
 * &prev->next); `prev` is NULL exactly for the head. bucket_relink() puts a
 * node back at the very address it was removed from — the claim's CAS-loss
 * rollback — and restores tail from node->next, which the unlink never clears. */
static void bucket_unlink(PipeLayerBucket *b, PipeJobNode **link,
                          PipeJobNode *node, PipeJobNode *prev) {
    *link = node->next;
    if (node == b->tail) b->tail = prev;   /* prev==NULL also covers the last node */
    b->count--;
}

static void bucket_relink(PipeLayerBucket *b, PipeJobNode **link, PipeJobNode *node) {
    *link = node;
    if (node->next == NULL) b->tail = node;
    b->count++;
}

static PipeJob *pipe_queue_claim(Model *m, int current_layer) {
    /* Single-shot: try to claim one job from the priority queue.
     * Returns NULL if no job is available. Retry logic is in the caller
     * (the worker thread) which parks on the condvar when NULL is returned. */
#ifdef PIPE_DEBUG
  fprintf(stderr, "[PIPE_DEBUG] pipe_worker: trying to claim job, current_layer=%d, total_jobs=%u\n", current_layer, (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_relaxed));
#endif
    pthread_mutex_lock(&g_pipe_queue.mx);

    PipeJob *job = NULL;
    WsToken tok = WS_NONE;

    /* Slots are a claim-time resource: no slot, no claim, and no mutation of
     * the queue at all. A worker that returns NULL here parks on the cv that
     * pipe_ws_release() broadcasts, so queued work cannot starve while a
     * consumer sits in pipe_wait_job(). */
    if (g_pipe_queue.ws_free > 0) {
      for (int offset = 0; offset < CLAIM_SCAN_DEPTH && job == NULL; offset++) {
	/* This is lookup_layers - 1 because the model does not include the MTP layer in its count, so we need to go to n_layers + 1 */
        int p = (current_layer + offset) % (g_pipe_queue.lookup_layers - 1);
#ifdef PIPE_DEBUG
        fprintf(stderr, "[PIPE_DEBUG] pipe_worker: trying to claim job, current_layer=%d, offset=%d, job_layer=%d\n", current_layer, offset, p);
#endif
        PipeLayerBucket *bucket = &g_pipe_queue.buckets[p];
        PipeJobNode **link = &bucket->head;
        PipeJobNode *node = bucket->head;
        PipeJobNode *prev = NULL;           /* node before node, NULL for the head */

        while (node != NULL) {
            PipeJob *candidate = node->job;

            /* Peek before unlinking. Anything that is not QUEUED or is CANCELLED
	       is stepped over; its node stays in the list. */
            if ((atomic_load_explicit(&candidate->state, memory_order_acquire) != PIPE_JOB_QUEUED) ||
		(atomic_load_explicit(&candidate->kind, memory_order_acquire) == PIPE_JOB_CANCELLED)) {
                    
                prev = node;
                link = &node->next;
                node = node->next;
                continue;
            }

            /* Mint the WS slot BEFORE unlinking, so the only way to give it
             * back is the rollback below. */
            if (!pipe_ws_acquire(atomic_load_explicit(&candidate->kind, memory_order_acquire), candidate->job_id, &tok)) {
#ifdef PIPE_WS_DEBUG
                fprintf(stderr, "[PIPE_WS_DEBUG] CLAIM FAIL: job=%d L%d E%d kind=%d ws_free=%d total_jobs=%u\n",
                        candidate->job_id, candidate->layer, candidate->eid,
                        atomic_load_explicit(&candidate->kind, memory_order_acquire), g_pipe_queue.ws_free,
                        (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire));
                /* Dump slot occupancy */
                for (int __i = 0; __i < WS_NSLOTS; __i++) {
                    WsSlotEnt *__e = &g_pipe_queue.ws_tab[__i];
                    if (__e->owned)
		        fprintf(stderr, "  slot %d: owned gen=%llu kind=%d owner=%d\n", __i, (unsigned long long)__e->gen, atomic_load_explicit(&__e->kind, memory_order_acquire), __e->owner_job);
                }
#endif
                break;
            }

            bucket_unlink(bucket, link, node, prev);
            atomic_fetch_sub(&g_pipe_queue.total_jobs, 1);

            PipeJobState expected = PIPE_JOB_QUEUED;
            if (atomic_compare_exchange_strong_explicit(
                    &candidate->state,
                    &expected,
                    PIPE_JOB_RUNNING,
                    memory_order_acq_rel,
                    memory_order_acquire)) {
                job = candidate;
                job->ws_tok = tok;      /* the worker's right to write ws[slot] */
                free(node);

                /* Remove from the dedup lookup while the queue mutex is still
                 * held: a later request for this expert must not get a pointer
                 * to a job that is now owned by a worker. */
                pipe_queue_clear_lookup(job->layer, job->eid);
                break;
            }

            /* Lost the CAS (a concurrent logical cancel). Restore the node at
             * the exact position it was unlinked from and hand the slot back;
             * the job object stays owned by the queue. */
            pipe_ws_release_locked(tok);
            tok = WS_NONE;
            bucket_relink(bucket, link, node);
            atomic_fetch_add(&g_pipe_queue.total_jobs, 1);
            prev = node;
            link = &node->next;
            node = node->next;
        }
      }
    }

    if (job == NULL) {
#ifdef PIPE_DEBUG
        fprintf(stderr, "[PIPE_DEBUG] pipe_worker: NO JOB FOUND, unlocking and parking\n");
#endif
        pthread_mutex_unlock(&g_pipe_queue.mx);
        return NULL;
    }

#ifdef PIPE_DEBUG
    fprintf(stderr, "[PIPE_DEBUG] pipe_worker: CLAIMED job layer=%d eid=%d state=%d\n", job->layer, job->eid, (int)job->state);
#endif
    pthread_mutex_unlock(&g_pipe_queue.mx);
    return job;
}

/* Persistent worker — reads from PipeQueue instead of PipePool batch arrays */
static void *pipe_worker(void *arg) {
    Model *m = (Model *)arg;
#ifdef PIPE_DEBUG
    fprintf(stderr, "[PIPE_DEBUG] pipe_worker: THREAD START m=%p\n", (void*)m);
#endif
    for (;;) {
        /* Read current layer from global — set by moe() before dispatch */
        int current_layer = (int)atomic_load_explicit(&g_cur_pipe_layer, memory_order_acquire);
        PipeJob *job = pipe_queue_claim(m, current_layer);
        if (!job) {
            /* No jobs: park on cv */
#ifdef PIPE_WS_DEBUG
            fprintf(stderr, "[PIPE_WS_DEBUG] WORKER PARK: total_jobs=%u ws_free=%d idle=%d\n",
                    (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire),
                    g_pipe_queue.ws_free,
                    (int)atomic_load_explicit(&g_pipe_queue.idle_workers, memory_order_relaxed));
#endif
#ifdef PIPE_DEBUG
            fprintf(stderr, "[PIPE_DEBUG] pipe_worker: no job available, parking (total_jobs=%u, idle_workers=%d)\n",
                    (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire),
                    (int)atomic_load_explicit(&g_pipe_queue.idle_workers, memory_order_relaxed));
#endif
            pthread_mutex_lock(&g_pipe_queue.mx);
            atomic_fetch_add(&g_pipe_queue.idle_workers, 1);
            /* Park only when there is nothing runnable at all. A queued job
             * whose WS slot is still held by a consumer is NOT runnable, so
             * the slot count is part of the wake predicate; pipe_ws_release()
             * broadcasts this cv when capacity comes back. */
            while (atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire) == 0
                   || g_pipe_queue.ws_free == 0) {
                pthread_cond_wait(&g_pipe_queue.cv, &g_pipe_queue.mx);
            }
#ifdef PIPE_WS_DEBUG
            fprintf(stderr, "[PIPE_WS_DEBUG] WOKEN: total_jobs=%u ws_free=%d\n",
                    (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire),
                    g_pipe_queue.ws_free);
#endif
#ifdef PIPE_DEBUG
            fprintf(stderr, "[PIPE_DEBUG] pipe_worker: woken up, total_jobs=%u\n",
                    (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire));
#endif
            atomic_fetch_sub(&g_pipe_queue.idle_workers, 1);
            pthread_mutex_unlock(&g_pipe_queue.mx);
            continue;
        }

        /* Execute the load OUTSIDE the queue lock.
         * The WS slot was minted by pipe_queue_claim() under the queue mutex,
         * so a RUNNING job always owns one; the pointer may legitimately be
         * used without the lock, the token may not be re-derived. */
        int L = job->layer;
        int eid = job->eid;
        WsToken wtok = job->ws_tok;
        uint32_t tok = job->token;         /* trace token, NOT a slot handle */

        /* requirement: no job enters I/O without a slot. pipe_queue_claim() mints the
         * token and the RUNNING transition together, so this aborts only on a bug;
         * the refusal below is what keeps it true in an NDEBUG build. */
        assert(ws_tok_valid(wtok));
        if (!ws_tok_valid(wtok)) {
            /* Cannot happen while claim owns both the CAS and the acquire.
             * Reaching I/O without a slot would scribble on a slot somebody
             * else owns, so give up on claiming and try again */
            fprintf(stderr, "[PIPE] job %d (L%d,E%d) claimed with no WS slot\n",
                    job->job_id, L, eid);
            continue;
        }
        int ws = WS_SLOT(wtok);

#ifdef PIPE_DEBUG
        fprintf(stderr, "[PIPE_DEBUG] pipe_worker: executing load layer=%d eid=%d ws=%d\n", L, eid, ws);
#endif
        exec_trace_load_ctx_set(tok, L, eid, expert_route(L, eid));
        exec_trace_load_event(EXEC_EV_IO_SUBMIT);
        expert_load(m, L, eid, &m->ws[ws], 1, 1);

        /* Mark complete. I/O completion makes the job READY and NOTHING
         * else — the WS slot stays owned by the job. The consumer that computes on
         * this expert claims it (pipe_job_consume) and releases it
         * (pipe_job_release), so a loaded buffer cannot be re-issued to the next
         * claim while anybody can still read it.
         *
         * State transition + signal are one mutex-protected event.
         * This eliminates any theoretical window where the state could become
         * visible before the signal reaches the CPU consumer. */
        pthread_mutex_lock(&g_pipe_queue.mx);
        atomic_store_explicit(&job->state, PIPE_JOB_READY, memory_order_release);
	exec_trace_load_event(EXEC_EV_EXPERT_READY);
        exec_trace_load_ctx_clear();
#ifdef PIPE_DEBUG
        fprintf(stderr, "[PIPE_DEBUG] pipe_worker: load complete for layer=%d eid=%d\n", L, eid);
#endif
        /* Wake the CPU consumer and pipe_wait() caller: an expert is now READY. */
	pthread_cond_broadcast(&g_pipe_queue.cv_ready);
	pthread_mutex_unlock(&g_pipe_queue.mx);

        /*                                                                                                                                                                                                 
         * Do NOT free(job).                                                                                                                                                                               
         *                                                                                                                                                                                                 
         * The dispatching moe() call retains PipeJob* in qjob[] and may still                                                                                                                             
         * call pipe_queue_wait(job).  The consumer owns the job lifetime.                                                                                                                                 
         */                                                                                                                                                                                                
    }
    return NULL;
}

/* Initialize the persistent priority queue. Called once from pipe_init(). */
static void pipe_queue_init(Model *m) {
    pthread_mutex_init(&g_pipe_queue.mx, NULL);
    pthread_cond_init(&g_pipe_queue.cv, NULL);
    pthread_cond_init(&g_pipe_queue.cv_ready, NULL);
    atomic_store(&g_pipe_queue.total_jobs, 0);
    atomic_store(&g_pipe_queue.idle_workers, 0);
    g_pipe_queue.next_job_id = 0;

    /* WS-slot ownership table. Generations start at 1 so that gen==0 is never
     * a mintable token, which is what makes a zero-initialised handle (calloc,
     * memset of Model, a PipeJob that skipped its initialisation) decode to
     * something invalid rather than to hot slot 0. */
    for (int i = 0; i < WS_NSLOTS; i++) {
        g_pipe_queue.ws_tab[i].gen = 1;
        g_pipe_queue.ws_tab[i].owned = 0;
        g_pipe_queue.ws_tab[i].kind = 0;
        g_pipe_queue.ws_tab[i].owner_job = -1;
    }
    g_pipe_queue.ws_free = WS_NSLOTS;
    g_pipe_queue.ws_acquire_n = g_pipe_queue.ws_release_n = 0;
    g_pipe_queue.ws_release_bad_n = 0;
    g_pipe_queue.ws_peak_used = 0;
    /* One registration per process: pipe_queue_init() runs once (pipe_queue_ensure()
     * guards the on-demand path), and the flag is still 0 here. */
    if(!g_pipe_queue.inited) atexit(pipe_ws_dump_stats);
    g_pipe_queue.inited = 1;

    /* Zero all buckets */
    for (int i = 0; i < PIPE_MAX_LAYERS; i++) {
        g_pipe_queue.buckets[i].head = NULL;
        g_pipe_queue.buckets[i].tail = NULL;
        g_pipe_queue.buckets[i].count = 0;
    }

    /* Allocate lookup table */
    g_pipe_queue.lookup_layers = (int)m->c.n_layers + 2;
    g_pipe_queue.lookup_experts = (int)m->c.n_experts;
    size_t lookup_size = (size_t)g_pipe_queue.lookup_layers * g_pipe_queue.lookup_experts * sizeof(PipeJob *);
    g_pipe_queue.lookup = (PipeJob **)calloc(lookup_size, 1);
    if (!g_pipe_queue.lookup) {
        fprintf(stderr, "[PIPE] lookup table allocation failed\n");
        exit(1);
    }
}

/* On-demand initialisation for code that needs the WS-slot table but may run in
 * a session where pipe_init() never did: a PIPE=0 build, or a pipe build whose
 * first block has no misses. Without it the table reads ws_free==0 and the
 * device-lost recovery path below could not borrow a scratch slot. Deliberately
 * never re-initialises — pipe_queue_init() resets the table, which would strip
 * ownership from slots a worker already holds. Same "first caller wins" shape
 * as the `if(!g_pp.started) pipe_init(m)` guard in moe(). */
static void pipe_queue_ensure(Model *m){
    if(!g_pipe_queue.inited) pipe_queue_init(m);
}

/* Enqueue a job into the persistent queue. Returns 0 on success, -1 on duplicate. */
static int pipe_queue_enqueue(PipeJob *job) {
#ifdef PIPE_DEBUG
    fprintf(stderr, "[PIPE_DEBUG] pipe_queue_enqueue: adding layer=%d eid=%d (job_id=%d)\n", job->layer, job->eid, job->job_id);
#endif
    pthread_mutex_lock(&g_pipe_queue.mx);

    /*                                                                                                                                                                                                     
     * Lookup and insertion must be one critical section.  Otherwise two                                                                                                                                   
     * dispatchers can both observe NULL and enqueue the same (layer,eid).                                                                                                                                 
     */                                                                                                                                                                                                    
    PipeJob *existing = pipe_queue_lookup(job->layer, job->eid);                                                                                                                                           
    if (existing != NULL) {                                                                                                                                                                                
#ifdef PIPE_DEBUG
        fprintf(stderr,
                "[PIPE_DEBUG] pipe_queue_enqueue: DUPLICATE for "
                "layer=%d eid=%d, rejecting\n",
                job->layer, job->eid);
#endif
        pthread_mutex_unlock(&g_pipe_queue.mx);                                                                                                                                                            
        return -1;                                                                                                                                                                                         
    }                                                                                                                                                                                                      
                                                                                                                                                                                                           
    PipeJobNode *node = (PipeJobNode *)malloc(sizeof(PipeJobNode));
    if (!node) {                                                                                                                                                                                           
        pthread_mutex_unlock(&g_pipe_queue.mx);                                                                                                                                                            
        fprintf(stderr, "[PIPE] PipeJobNode allocation failed\n");                                                                                                                                         
        return -1;                                                                                                                                                                                         
    }                                                                                                                                                                                                      
                                                                                                                                                                                                           
    node->job = job;
    node->next = NULL;

    PipeLayerBucket *bucket = &g_pipe_queue.buckets[job->layer];
    if (bucket->head == NULL) {
        bucket->head = node;
        bucket->tail = node;
    } else {
        bucket->tail->next = node;
        bucket->tail = node;
    }
    bucket->count++;

    atomic_fetch_add(&g_pipe_queue.total_jobs, 1);

    /* Insert into lookup table */
    pipe_queue_set_lookup(job->layer, job->eid, job);
    /* A queued job is only runnable while a WS slot is free; the predicate at
     * the worker park site tests both, so this one broadcast covers both. */

    pthread_cond_signal(&g_pipe_queue.cv);
    pthread_mutex_unlock(&g_pipe_queue.mx);

#ifdef PIPE_DEBUG
    fprintf(stderr, "[PIPE_DEBUG] pipe_queue_enqueue: enqueued OK, total_jobs=%u\n", (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_relaxed));
#endif
    return 0;
}

/* Logical cancellation. Called when a pilot prediction turns out to be wrong. */
static void pipe_queue_cancel(PipeJob *job) {
    atomic_store_explicit(&job->kind, PIPE_JOB_CANCELLED, memory_order_release);
}

/* Promote a job's kind (e.g., PILOT → DEMAND). Only changes kind, never layer. */
static void pipe_queue_promote(PipeJob *job, PipeJobKind new_kind) {
    pthread_mutex_lock(&g_pipe_queue.mx);

    PipeJobKind old_kind = atomic_load_explicit(&job->kind, memory_order_acquire);
    atomic_store_explicit(&job->kind, new_kind, memory_order_release);

    if (old_kind == PIPE_JOB_PILOT && new_kind == PIPE_JOB_DEMAND) {
        /* Wake all workers — demand should outrank any running pilot */
        pthread_cond_broadcast(&g_pipe_queue.cv);
    }

    pthread_mutex_unlock(&g_pipe_queue.mx);
}

/* Wait for a specific job to complete. Replaces the old pipe_wait(int q). */
static void pipe_queue_wait(PipeJob *job) {
    while (1) {
        PipeJobState s = atomic_load_explicit(&job->state, memory_order_acquire);
        PipeJobKind k = atomic_load_explicit(&job->kind, memory_order_acquire);
        /* CONSUMING and RELEASED are past the load too. Listing them keeps a
         * wait placed after a consumption from spinning on a state that will never
         * go back to READY. */
        if (pipe_job_state_loaded(s) || s == PIPE_JOB_RELEASED) return;
	/* A cancelled job will never transition out of queued, so there is no benefit to waiting
	 * Perhaps this should print out an error message, because waiting on a cancelled job is wrong */
	if (s == PIPE_JOB_QUEUED && k == PIPE_JOB_CANCELLED) return;
        /* QUEUED or RUNNING */
        sched_yield();
    }
}

/* ---- Opportunistic READY selection from the compute sequence ---- */

/* Pre-compute cache promotion targets for all I/O-bound sequence entries.
 * Called after I/O jobs are enqueued but before execution starts.
 * Capacity slots filled first (reverse order), then LRU victims.
 * Each I/O-bound expert gets a predetermined dst_cache_slot established before
 * execution requires them (architecture invariant).
 * missk[] maps miss ordinal → index into uniq[]/qjob[].
 * Results are stored in both seq[].dst_cache_slot and job->dst_cache_slot
 * so both the sequence-driven path and GPU paths can read them directly.
 * Returns: void (results stored in seq[].dst_cache_slot and job->dst_cache_slot). */
static void pipe_seq_compute_promotions(Model *m, int layer,
                                        PipeSeqEntry *seq, int seq_n) {

    /* Mark EVERY entry as having no promotion target FIRST, before any early
     * return below. seq[] is a stack array in moe() and the per-expert finish
     * site reads dst_cache_slot for every entry it claimed, so a value left
     * uninitialised on the promo_count<=0 / !Sl paths would be used as a live
     * cache index (a bogus swap into a random ecache slot). */
    for (int i = 0; i < seq_n; i++) seq[i].dst_cache_slot = -1;

    /* Misses are counted from the sequence itself, not from qjob[]/missk[]: an
     * entry needs a cache home iff it was a miss (needs_io), whether or not it
     * owns a PipeJob. That keeps the PIPE=0 reference path promoting too, which
     * it stopped doing when the block-wide FASE D swap was removed. */
    int nmiss = 0;
    for (int i = 0; i < seq_n; i++) if (seq[i].needs_io) nmiss++;

    /* Determine promotion count: cap at cache capacity */
    int promo_count = nmiss < m->ecap ? nmiss : m->ecap;
    if (promo_count <= 0) return;

    ESlot *Sl = m->ecache[layer];
    if (!Sl) return;

    /* Pre-compute targets in reverse order so the last experts in missk[]
     * get the capacity slots first. Exactly `promo_count` iterations.
     *
     * NOTE: ws_tok is NOT checked here.  At promotion-calculation time
     * the job may still be QUEUED and has no WS slot yet.  The WS token only
     * matters later, when the actual data is copied/swapped into the destination.
     *
     * A temporary cursor tracks capacity-slot assignment so successive promotions
     * get distinct slots.  The real m->ecn[layer] is NOT modified —
     * promotion decisions are predetermined for the layer.
     */
    /* Architecture invariant: "a routed expert selected for the final cache will
     * not be evicted during that layer". An expert that is ALREADY resident and
     * still pending in this compute sequence must therefore never be picked as a
     * victim for a later expert of the same layer. eslot_lru_victim() skips slots
     * that carry an in-flight reference, so hold one on every resident sequence
     * expert for the duration of the selection and drop it again before returning.
     * Without this, the swap below can empty the slot a later sequence entry is
     * about to matmul on (se->eslot would name a different expert). */
    ESlot *res[WS_NSLOTS]; int nres = 0;
    for (int i = 0; i < seq_n; i++)
        if (!seq[i].needs_io && seq[i].eslot) {
            eslot_acquire(seq[i].eslot);
            res[nres++] = seq[i].eslot;
        }

    int cur = m->ecn[layer];            /* temporary capacity-slot cursor */
    /* Record of LRU slots assigned
     * Claim each cache slot as busy when it is claimed for replacement
     * Then release after claiming because they aren't actually in use */
    int assigned[promo_count];     
    for (int i = 0; i < promo_count; i++) {
      assigned[i]=-1;
    }

    /* Reverse ROUTING order, so the last misses in the block get the free
     * capacity slots first and the earlier ones take LRU victims — the same
     * assignment order the block-wide FASE D swap used. */
    int a = 0;
    for (int i = seq_n - 1; i >= 0 && a < promo_count; i--) {
        if (!seq[i].needs_io) continue;
        /* no ws_tok test here — the job may still be QUEUED and own no
         * slot; the slot is resolved at the finish site. */

        int cache_slot;
        if (cur < m->ecap) {
            /* New capacity slot — advance temp cursor (Issue 4) */
            cache_slot = cur++;
        } else {
            int victim = eslot_lru_victim(Sl, m->ecn[layer], m->ecap);
            if (victim < 0) break;
            assigned[a] = victim;
	    eslot_acquire(&Sl[victim]);
            cache_slot = victim;
        }

        /* Record the pre-computed target for the sequence entry, and also on the
         * job itself so Metal/Vulkan paths can read it directly. */
        seq[i].dst_cache_slot = cache_slot;
        if (seq[i].job) seq[i].job->dst_cache_slot = cache_slot;
        a++;
    }
    for (int i = 0; i < nres; i++) eslot_release(res[i]);
    for (int i = 0; i < promo_count; i++) {
        if (assigned[i]>=0) {
	    eslot_release(&Sl[assigned[i]]);
	}
    }

}

/* Scan the compute sequence for the first unconsumed entry whose expert is READY.
 * Held under g_pipe_queue.mx throughout.
 *   - Cached entries (job==NULL) are immediately ready — return them outright.
 *   - I/O entries need their PipeJob state to be PIPE_JOB_READY.
 *   - A cached entry requires no state transition; a READY I/O entry is
 *     transitioned to CONSUMING (claim) before returning.
 * If no entry is ready: wait on cv_ready, rescan (handles spurious wakeups).
 * Returns the claimed PipeSeqEntry, or NULL when seq_n == 0. */
#ifdef PIPE_WS_DEBUG
static const char *pipe_job_state_name(int s) {
    switch (s) {
    case PIPE_JOB_QUEUED:    return "QUEUED";
    case PIPE_JOB_RUNNING:   return "RUNNING";
    case PIPE_JOB_READY:     return "READY";
    case PIPE_JOB_CONSUMING: return "CONSUMING";
    case PIPE_JOB_RELEASED:  return "RELEASED";
    default:                 return "?";
    }
}
#endif

static PipeSeqEntry *pipe_seq_find_ready(Model *m, PipeSeqEntry *seq, int seq_n) {
    if (seq_n <= 0) return NULL;

    while (1) {
        int unconsumed = 0;
        for (int i = 0; i < seq_n; i++) {
            PipeSeqEntry *e = &seq[i];

            /* Skip already-executed entries */
            if (e->consumed) continue;

	    /* Skip cancelled jobs */
	    if ((e->job) && (atomic_load_explicit(&e->job->kind, memory_order_acquire) == PIPE_JOB_CANCELLED)) continue;
	    
            unconsumed = 1;

            /* No job means nothing has to be waited for: either the expert is
             * already resident (pin/ecache), or it is a PIPE=0 miss whose load
             * completed synchronously into e->eslot. Take the in-flight ref that
             * keeps LRU/rss_guard off it while the caller computes; the matching
             * release is moe_seq_finish(), keyed on e->ref, and runs even when the
             * caller skips the expert. */
            if (!e->job) {
                if (e->eslot && !e->ref) { eslot_acquire(e->eslot); e->ref = 1; }
                return e;
            }

            /* I/O-bound expert: check if its job is READY. */
            if (e->job) {
	      PipeJobState s = atomic_load_explicit(&e->job->state,
                                                      memory_order_acquire);
                /* A job a DIFFERENT path already resolved (Metal drain)
                 * can never come back to READY. Marking
                 * the entry consumed here is what stops this scan from sleeping on
                 * cv_ready forever for work that will never arrive. */
                if (s == PIPE_JOB_RELEASED) {
                    e->consumed = 1;
                    continue;
                }
                if (s == PIPE_JOB_READY) {
                    /* Claim: READY → CONSUMING under the queue mutex.
                     * Direct state transition under mutex (no CAS).
                     * In-flight ref count acquired to prevent LRU replacement
                     * of this slot while the CPU is about to consume it.
                     * Matching eslot_release() in moe() after computation. */
                    atomic_store_explicit(&e->job->state, PIPE_JOB_CONSUMING,
                                          memory_order_release);
                    WsToken ct = e->job->ws_tok;
                    if (ws_tok_valid(ct)) { eslot_acquire(&m->ws[WS_SLOT(ct)]); e->ref = 1; }
                    e->scratch = WS_NONE;
#ifdef PIPE_WS_DEBUG
                    fprintf(stderr, "[PIPE_WS_DEBUG] SEQ CLAIM job=%d L%d E%d slot=%d ws_free=%d\n",
                            e->job->job_id, e->job->layer, e->job->eid,
                            ws_tok_valid(ct) ? WS_SLOT(ct) : -1, g_pipe_queue.ws_free);
#endif
                    return e;
                }
            }
        }

        /* All entries consumed — no more work for this block. */
        if (!unconsumed) return NULL;

        /* Only a PipeJob is ever woken by an I/O completion. If no remaining
         * entry is waiting on one, this cv has no possible signaler and cv_ready
         * would never fire: a PIPE=0 build (no workers, maybe no pipe_init() at
         * all), or entries left in a state a worker will never move again. Return
         * and let the caller finish the block instead of sleeping forever on a
         * condvar nobody owns. */
        int awaiting_worker = 0;
        for (int i = 0; i < seq_n; i++)
            if (!seq[i].consumed && seq[i].job) { awaiting_worker = 1; break; }
        if (!awaiting_worker) return NULL;

#ifdef PIPE_WS_DEBUG
        /* Everything a hang report needed and did not have: which experts are
         * still outstanding, in which job state, and what the slot pool looks
         * like while we are about to sleep on cv_ready. */
        fprintf(stderr, "[PIPE_WS_DEBUG] SEQ WAIT: ws_free=%d total=%u idle=%d |",
                g_pipe_queue.ws_free,
                (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_acquire),
                (int)atomic_load_explicit(&g_pipe_queue.idle_workers, memory_order_relaxed));
        for (int i = 0; i < seq_n; i++)
            if (!seq[i].consumed)
                fprintf(stderr, " E%d[%s]", seq[i].expert_id,
                        seq[i].job ? pipe_job_state_name((int)atomic_load_explicit(
                                         &seq[i].job->state, memory_order_relaxed))
                                   : "nojob");
        fprintf(stderr, "\n");
#endif

        /* Nothing ready: wait on cv_ready for an I/O completion.  The
         * standard condvar pattern — hold mx, test predicate, wait, rescan. */
        pthread_cond_wait(&g_pipe_queue.cv_ready, &g_pipe_queue.mx);
    }
}

static void pipe_init(Model *m){
    if(g_pp.started) return;
#ifdef __linux__
    if(g_uring){
        if(uring_batch_init(&g_ub_pipe)){ perror("URING=1 io_uring_setup"); exit(1); }
        /* Even with no pthread workers the queue object must be initialised:
         * moe() still calls pipe_enqueue_demand(), which locks g_pipe_queue.mx
         * and reads the WS-slot table. Leaving it zero-initialised made the
         * queue "work" only because a zeroed pthread_mutex_t happens to be an
         * unlocked mutex on glibc, and would read as ZERO slot capacity. */
        pipe_queue_init(m);
        g_pp.m=m; g_pp.started=1; return;
    }
#endif
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16) g_pp.nw=16; if(g_pp.nw<1) g_pp.nw=1;
#ifdef PIPE_DEBUG
    fprintf(stderr, "[PIPE_DEBUG] pipe_init: creating %d worker threads (m=%p)\n", g_pp.nw, (void*)m);
#endif
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    pthread_cond_init(&g_pp.cv_done,NULL);  /* fallback for pipe_wait() and Metal compat */
    /* The persistent priority queue (and with it the WS-slot table, whose
     * zero value means "no capacity" rather than "64 free") MUST exist before
     * the first worker can run. It used to be initialised after the
     * pthread_create loop, so a worker scheduled in that window tested an
     * uninitialised mutex and an empty slot table. */
    pipe_queue_init(m);
    for(int i=0;i<g_pp.nw;i++) {
        pthread_create(&g_pp.th[i],NULL,pipe_worker,m);
#ifdef PIPE_DEBUG
        fprintf(stderr, "[PIPE_DEBUG] pipe_init: created worker thread %d\n", i);
#endif
    }
    g_pp.started=1;
}
/* Dispatch enqueues PipeJob objects into the persistent priority queue.
 * The batch-oriented PipePool arrays are no longer used for scheduling — workers
 * read from PipeQueue instead. `jobs` is the array of pre-allocated PipeJob* from
 * moe() (NULL for URING path). */
static void pipe_dispatch(Model *m, int layer, const int *eids,
                          const uint32_t *tokens, int njobs,
                          PipeJob *const *jobs) {
#ifdef __linux__
    if (g_uring) {
        uring_batch_reset(&g_ub_pipe);
        for (int q = 0; q < njobs; q++) {
            int li = uring_load_add(&g_ub_pipe, m, layer, eids[q], &m->ws[q], 1, tokens[q]);
            if (li != q) { fprintf(stderr, "URING: expert batch overflow\n"); exit(1); }
        }
        if (uring_submit_batch(&g_ub_pipe)) { perror("URING: submit"); exit(1); }
        return;
    }
#endif
    if (!g_pp.started) pipe_init(m);

#ifdef PIPE_DEBUG
    fprintf(stderr, "[PIPE_DEBUG] pipe_dispatch: layer=%d njobs=%d g_uring=%d\n", layer, njobs, g_uring);
#endif
    /* Set current layer context for workers */
    atomic_store_explicit(&g_cur_pipe_layer, layer, memory_order_release);

    for (int q = 0; q < njobs; q++) {
        PipeJob *job = jobs ? jobs[q] : NULL;
        if (!job) {
            /* Fallback: create job inline (shouldn't happen with updated moe()) */
            job = (PipeJob *)calloc(1, sizeof(PipeJob));                                                                                                                                                   
            if (!job) {                                                                                                                                                                                    
                fprintf(stderr, "[PIPE] PipeJob allocation failed\n");                                                                                                                                     
                exit(1);                                                                                                                                                                                   
            }                                                                                                                                                                                              
                                                                                                                                                                                                           
            job->layer = layer;
            job->eid = eids[q];
	    atomic_store_explicit(&job->kind, PIPE_JOB_DEMAND, memory_order_release);
            job->ws_tok = WS_NONE;      /* the worker acquires on claim */
            job->token = tokens[q];
            job->dst_cache_slot = -1;     /* -1 = no promotion target */
            atomic_store_explicit(&job->state, PIPE_JOB_QUEUED, memory_order_relaxed);
            pthread_mutex_lock(&g_pipe_queue.mx);                                                                                                                                                          
            job->job_id = g_pipe_queue.next_job_id++;                                                                                                                                                      
            pthread_mutex_unlock(&g_pipe_queue.mx);                                                                                                                                                        
        }
        /*                                                                                                                                                                                                 
         * Demand jobs must always receive their trace token explicitly.                                                                                                                                   
         * Do not leave a malloc-created token field uninitialized.                                                                                                                                        
         */                                                                                                                                                                                                
        job->token = tokens[q];                                                                                                                                                                            

        int enq_rc = pipe_queue_enqueue(job);
        if (enq_rc != 0) {
            fprintf(stderr,                                                                                                                                                                                
                    "[PIPE] duplicate job for (%d,%d), skipping\n",                                                                                                                                        
                    layer, eids[q]);                                                                                                                                                                       
                                                                                                                                                                                                           
            /*                                                                                                                                                                                             
             * For fallback-created jobs, dispatch owns the object and can free                                                                                                                            
             * it immediately.  For jobs supplied by moe(), the caller owns the                                                                                                                            
             * object and needs its qjob[] entry fixed separately.                                                                                                                                         
             */                                                                                                                                                                                            
            if (!jobs)                                                                                                                                                                                     
                free(job);                                                                                                                                                                                 
        }
    }
#ifdef PIPE_DEBUG
    fprintf(stderr, "[PIPE_DEBUG] pipe_dispatch: done dispatching, total_jobs=%u\n", (unsigned)atomic_load_explicit(&g_pipe_queue.total_jobs, memory_order_relaxed));
#endif
}

/* Non-blocking probe of a pipe job's completion status.
 * Kept for Metal path compatibility — returns 1 if job is READY, 0 if QUEUED/RUNNING.
 * Under URING, completion happens inside finalize, so report not-ready. */
static inline int pipe_ready(int q) {
    (void)q;
#ifdef __linux__
    if (g_uring) return 0;
#endif
    /* pipe_ready is not used with the persistent queue.
     * The Metal path uses pipe_queue_wait instead. Return 0 as safe default. */
    return 0;
}

/* Per-slot completion wait — replaced by pipe_queue_wait(PipeJob*) for Phase 2.
 * Kept for URING path compatibility and as a fallback. */
static inline void pipe_wait(int q) {
    (void)q;
#ifdef __linux__
    if (g_uring) {
        if (uring_finalize_load(&g_ub_pipe, q, 1)) { perror("URING: expert load"); exit(1); }
        return;
    }
#endif
    /* This path should not be reached for PIPE=1. Use pipe_queue_wait(job) instead.
     * Fallback spin — should only trigger if URING=0 and the old batch path is still used. */
    if (g_pipe_block) {
        pthread_mutex_lock(&g_pp.mx);
        pthread_cond_wait(&g_pp.cv_done, &g_pp.mx);
        pthread_mutex_unlock(&g_pp.mx);
        return;
    }
    sched_yield();
}
