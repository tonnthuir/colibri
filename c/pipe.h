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
 * The per-expert pipe_wait(ready[q]) in the matmul loop makes every grabbed job
 * complete before the block ends, so no grab outlives its generation — which is
 * why the old `active` counter AND the end-of-block drain barrier are gone (both
 * were redundant with those per-slot waits + the gen-tagged cursor). The mutex/
 * condvar exist ONLY to park/wake idle workers, never for correctness. Gated
 * behind PIPE=1; OFF => the original blocking-load + serial-matmul path runs
 * byte-identically. */
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
    _Atomic int layer;                            /* current batch layer */
    _Atomic int ready[64];                        /* per-slot load-done flag */
    pthread_mutex_t mx; pthread_cond_t cv;        /* ONLY for parking/waking idle workers */
    pthread_cond_t cv_done;                       /* COLI_PIPE_BLOCK: signals ready[] transitions */
    Model *m;
    pthread_t th[16]; int nw; int started;
} PipePool;
static PipePool g_pp;

static void *pipe_worker(void *arg){
    (void)arg; PipePool *p=&g_pp; uint64_t seen=0;
    for(;;){
        pthread_mutex_lock(&p->mx);
        while((atomic_load_explicit(&p->cur,memory_order_relaxed)>>8)==seen)
            pthread_cond_wait(&p->cv,&p->mx);
        pthread_mutex_unlock(&p->mx);
        for(;;){
            uint64_t c=atomic_load_explicit(&p->cur,memory_order_acquire);
            seen=c>>8;
            uint32_t i=(uint32_t)(c & 0xFF);
            if(i >= (uint32_t)atomic_load_explicit(&p->njobs,memory_order_relaxed))
                break;                                /* batch drained → re-park */
            if(atomic_compare_exchange_weak_explicit(&p->cur,&c,c+1,
                    memory_order_acq_rel,memory_order_relaxed)){
                int L  =atomic_load_explicit(&p->layer,memory_order_relaxed);
                int eid=atomic_load_explicit(&p->eids[i],memory_order_relaxed); /* AFTER winning CAS */
                expert_load(p->m,L,eid,&p->m->ws[i],1,1);  /* needed-now load: fatal on I/O error (matches serial path); demand=1: this IS moe()'s own miss path */
                atomic_store_explicit(&p->ready[i],1,memory_order_release);
                if(g_pipe_block){                     /* wake a main thread parked in pipe_wait */
                    pthread_mutex_lock(&p->mx);
                    pthread_cond_broadcast(&p->cv_done);
                    pthread_mutex_unlock(&p->mx);
                }
            }
            /* CAS failed → another worker advanced index (or gen advanced): re-loop */
        }
    }
    return NULL;
}
static void pipe_init(Model *m){
    if(g_pp.started) return;
#ifdef __linux__
    if(g_uring){
        if(uring_batch_init(&g_ub_pipe)){ perror("URING=1 io_uring_setup"); exit(1); }
        g_pp.m=m; g_pp.started=1; return;
    }
#endif
    g_pp.m=m; g_pp.nw=g_pipe_nw; if(g_pp.nw>16) g_pp.nw=16; if(g_pp.nw<1) g_pp.nw=1;
    atomic_store(&g_pp.cur,0); atomic_store(&g_pp.njobs,0);
    pthread_mutex_init(&g_pp.mx,NULL); pthread_cond_init(&g_pp.cv,NULL);
    pthread_cond_init(&g_pp.cv_done,NULL);
    for(int i=0;i<g_pp.nw;i++) pthread_create(&g_pp.th[i],NULL,pipe_worker,NULL);
    g_pp.started=1;
}
/* enqueue `njobs` loads (slots ws[0..njobs)); returns immediately, workers run ahead.
 * Order is load-bearing: write all batch state RELAXED, then RELEASE-store cur to
 * publish it, then wake parked workers. */
static void pipe_dispatch(Model *m,int layer,const int *eids,int njobs){
#ifdef __linux__
    if(g_uring){
        uring_batch_reset(&g_ub_pipe);
        for(int q=0;q<njobs;q++){
            int li=uring_load_add(&g_ub_pipe,m,layer,eids[q],&m->ws[q],1);
            if(li!=q){ fprintf(stderr,"URING: expert batch overflow\n"); exit(1); }
        }
        if(uring_submit_batch(&g_ub_pipe)){ perror("URING: submit"); exit(1); }
        return;
    }
#endif
    g_pp.m=m;
    atomic_store_explicit(&g_pp.njobs,njobs,memory_order_relaxed);
    atomic_store_explicit(&g_pp.layer,layer,memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.eids[q],eids[q],memory_order_relaxed);
    for(int q=0;q<njobs;q++) atomic_store_explicit(&g_pp.ready[q],0,memory_order_relaxed); /* reset BEFORE publish */
    uint64_t g=(atomic_load_explicit(&g_pp.cur,memory_order_relaxed)>>8)+1;
    atomic_store_explicit(&g_pp.cur,(g<<8),memory_order_release);                          /* PUBLISH */
    pthread_mutex_lock(&g_pp.mx); pthread_cond_broadcast(&g_pp.cv); pthread_mutex_unlock(&g_pp.mx);
}
/* Non-blocking probe of a pipe slot's load-done flag — an ORDERING hint only (the
 * VK block computes ready experts first): callers still pipe_wait() before touching
 * the slab. Under URING completion happens inside finalize, there is no flag to
 * peek — report not-ready and let the wait do the work. */
static inline int pipe_ready(int q){
#ifdef __linux__
    if(g_uring) return 0;
#endif
    return atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)!=0;
}
static inline void pipe_wait(int q){
#ifdef __linux__
    if(g_uring){
        if(uring_finalize_load(&g_ub_pipe,q,1)){ perror("URING: expert load"); exit(1); }
        return;
    }
#endif
    if(g_pipe_block){
        /* Fast path senza lock; poi ri-verifica SOTTO il lock prima di ogni
         * wait. EN: the worker stores ready (release) BEFORE it takes mx to
         * broadcast, so a set flag can never be missed (no lost wakeup). */
        if(atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) return;
        pthread_mutex_lock(&g_pp.mx);
        while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire))
            pthread_cond_wait(&g_pp.cv_done,&g_pp.mx);
        pthread_mutex_unlock(&g_pp.mx);
        return;
    }
    while(!atomic_load_explicit(&g_pp.ready[q],memory_order_acquire)) sched_yield();
}
