#!/bin/bash
# run_smoke.sh -- the smoke instances of the four verification problems, with the
# few companions needed to read them (control, other coupling mode, one refinement).
# Serial, one core, niceness 10.  Usage:
#     bash run_smoke.sh [group ...]      groups: p1 p2 p2x p3 p3s p4 p4o probes   (default: all)
# Every case is a directory under smoke/; the measures are in <case>/result.json and
# the solver output in <case>/log.*.  smoke_report.py turns them into RUNLOG.md.
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${SMOKE_OUT:-$HERE/smoke}"
LOG="$OUT/run_smoke.log"
CONORMAL_LIB="${CONORMAL_LIB:-$HERE/../derivation/coNormalInterface/lib/libcoNormalInterface.so}"
mkdir -p "$OUT"
GROUPS_=("$@"); [ ${#GROUPS_[@]} -eq 0 ] && GROUPS_=(p1 p2 p2x p3 p3s p4 p4o probes)
want() { for g in "${GROUPS_[@]}"; do [ "$g" = "$1" ] && return 0; done; return 1; }

run() {            # run <name> <problem> key=value ...
    local name=$1 prob=$2; shift 2
    rm -rf "$OUT/$name"
    python3 "$HERE/gencase.py" new "$prob" "$OUT/$name" "$@" > /dev/null || { echo "$name: generator failed" | tee -a "$LOG"; return; }
    local t0=$(date +%s.%N)
    nice -n 10 openfoam2606 -c "$OUT/$name/Allrun" > "$OUT/$name/log.Allrun" 2>&1
    local rc=$? t1=$(date +%s.%N)
    printf '%s  %-34s rc=%d  wall=%6.1fs  %s\n' "$(date +%H:%M:%S)" "$name" $rc "$(echo "$t1 - $t0" | bc)" \
        "$(tail -n 1 "$OUT/$name/log.Allrun")" | tee -a "$LOG"
}
mono() {           # mono <name> <ncorr>   single-region solidFoam reference of an existing case
    local name=$1
    rm -rf "$OUT/${name}_mono"
    python3 "$HERE/monoref.py" new "$OUT/$name" ncorr=$2 > /dev/null
    local t0=$(date +%s.%N)
    nice -n 10 openfoam2606 -c "$OUT/${name}_mono/Allrun" > "$OUT/${name}_mono/log.Allrun" 2>&1
    local rc=$? t1=$(date +%s.%N)
    printf '%s  %-34s rc=%d  wall=%6.1fs  %s\n' "$(date +%H:%M:%S)" "${name}_mono" $rc "$(echo "$t1 - $t0" | bc)" \
        "$(tail -n 1 "$OUT/${name}_mono/log.Allrun")" | tee -a "$LOG"
}

echo "=== run_smoke.sh $(date -Is) groups: ${GROUPS_[*]} load: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$LOG"

# ---------------------------------------------------------------- P1 oblique patch test
if want p1; then
    P1="k1=28 k2=0.8 k3=0.8 ki=1 La=0.01 Li=0.01 H=0.01 T0=320 A=1000 B=500"
    run p1_dir_t30_n16      patch $P1 theta=30 nxa=16 nxi=16 ny=16 nsteps=400 coupling=directional     # primary instance
    run p1_dir_t30_n8       patch $P1 theta=30 nxa=8  nxi=8  ny=8  nsteps=200 coupling=directional
    run p1_dir_t30_n32      patch $P1 theta=30 nxa=32 nxi=32 ny=32 nsteps=800 coupling=directional
    run p1_sca_t30_n16      patch $P1 theta=30 nxa=16 nxi=16 ny=16 nsteps=400 coupling=scalar
    run p1_dir_t0_n16       patch $P1 theta=0  nxa=16 nxi=16 ny=16 nsteps=400 coupling=directional     # aligned control
    run p1_dir_t30_n16_rot  patch $P1 theta=30 nxa=16 nxi=16 ny=16 nsteps=400 coupling=directional phi=20
    run p1_shear_dir_t30    patch $P1 theta=30 nxa=16 nxi=16 ny=4  nsteps=400 coupling=directional ybc=gradient
    run p1_shear_sca_t30    patch $P1 theta=30 nxa=16 nxi=16 ny=4  nsteps=400 coupling=scalar      ybc=gradient
    if [ -f "$CONORMAL_LIB" ]; then
        run p1_con_t30_n16  patch $P1 theta=30 nxa=16 nxi=16 ny=16 nsteps=400 coupling=conormal conormal_lib="$CONORMAL_LIB"
        run p1_shear_con_t30 patch $P1 theta=30 nxa=16 nxi=16 ny=4 nsteps=400 coupling=conormal ybc=gradient conormal_lib="$CONORMAL_LIB"
    else
        echo "co-normal library not found at $CONORMAL_LIB: corrected-coupling runs skipped" | tee -a "$LOG"
    fi
    mono p1_dir_t30_n8  8
    mono p1_dir_t30_n16 8
    mono p1_dir_t30_n32 8
    mono p1_dir_t0_n16  8
fi

# ---------------------------------------------------------------- P2 annulus
if want p2; then
    P2="r1=0.004 r2=0.009 r3=0.015 ko=0.15 T1=350 T3=300"
    run p2_dir_A_L1   annulus $P2 kr=0.8 kth=0.8 kz=28 nra=10 nri=12 nth=16 nsteps=600  coupling=directional  # primary instance
    run p2_dir_A_L2   annulus $P2 kr=0.8 kth=0.8 kz=28 nra=20 nri=24 nth=32 nsteps=1200 coupling=directional
    run p2_dir_A_L3   annulus $P2 kr=0.8 kth=0.8 kz=28 nra=40 nri=48 nth=64 nsteps=2400 coupling=directional
    run p2_sca_A_L1   annulus $P2 kr=0.8 kth=0.8 kz=28 nra=10 nri=12 nth=16 nsteps=600  coupling=scalar
    run p2_sca_A_L2   annulus $P2 kr=0.8 kth=0.8 kz=28 nra=20 nri=24 nth=32 nsteps=1200 coupling=scalar
    run p2_dir_B_ring annulus $P2 kr=0.8 kth=28 kz=28 sector=360 nra=10 nri=12 nth=16 nsteps=600 coupling=directional
    run p2_dir_C_trap annulus $P2 kr=0.8 kth=28 kz=28 sector=90  nra=10 nri=12 nth=16 nsteps=600 coupling=directional
    run p2_dir_C_trap_nth64 annulus $P2 kr=0.8 kth=28 kz=28 sector=90 nra=10 nri=12 nth=64 nsteps=600 coupling=directional
fi

# ---------------------------------------------------------------- P2 variant C with the scalar option (expected to fail)
if want p2x; then
    run p2_sca_C_trap annulus r1=0.004 r2=0.009 r3=0.015 ko=0.15 T1=350 T3=300 kr=0.8 kth=28 kz=28 sector=90 nra=10 nri=12 nth=16 nsteps=600 coupling=scalar
fi

# ---------------------------------------------------------------- P3 transient two-layer
if want p3; then
    P3="k1=28 k2=0.8 k3=0.8 theta=60 ki=0.15 La=0.004 Li=0.006 rhoA=2000 cpA=1000 rhoI=1000 cpI=1500 xbc=NN ic=step Ta0=350 Ti0=300 nxa=32 nxi=48 nwrite=4"
    run p3_dir_dt04      slab $P3 dt=0.4 nsteps=100 coupling=directional
    run p3_dir_dt02      slab $P3 dt=0.2 nsteps=200 coupling=directional                 # primary instance
    run p3_dir_dt01      slab $P3 dt=0.1 nsteps=400 coupling=directional
    run p3_dir_dt02_o3   slab $P3 dt=0.2 nsteps=200 coupling=directional nouter=3
    run p3_dir_dt02_rev  slab $P3 dt=0.2 nsteps=200 coupling=directional order="aniso iso"
    run p3_sca_dt02      slab $P3 dt=0.2 nsteps=200 coupling=scalar
    run p3_sca_dt01      slab $P3 dt=0.1 nsteps=400 coupling=scalar
    mono p3_dir_dt02 30
fi

# ---------------------------------------------------------------- P3 with a tangential gradient (sheared, transient)
if want p3s; then
    P3S="k1=28 k2=0.8 k3=0.8 theta=30 ki=0.15 La=0.004 Li=0.006 H=0.002 rhoA=2000 cpA=1000 rhoI=1000 cpI=1500 xbc=DD ic=uniform Ta0=300 TL=350 TR=300 B=2000 nxa=32 nxi=48 ny=4 nwrite=4 dt=0.2 nsteps=200"
    run p3s_dir_dt02     slab $P3S coupling=directional
    if [ -f "$CONORMAL_LIB" ]; then
        run p3s_con_dt02 slab $P3S coupling=conormal conormal_lib="$CONORMAL_LIB"
        run p3s_con_dt01 slab ${P3S/dt=0.2 nsteps=200/dt=0.1 nsteps=400} coupling=conormal conormal_lib="$CONORMAL_LIB"
    fi
fi

# ---------------------------------------------------------------- P4 aligned slab, flux ratio
if want p4; then
    P4="ki=0.15 La=0.004 Li=0.006 rhoA=2000 cpA=1000 rhoI=1000 cpI=1500 xbc=NN ic=step Ta0=350 Ti0=300 nwrite=2"
    run p4_sca_t90        slab $P4 k1=28 k2=0.8 k3=0.8 theta=90 nxa=16 nxi=24 dt=0.05 nsteps=400 coupling=scalar   # primary instance
    run p4_sca_t90_fine   slab $P4 k1=28 k2=0.8 k3=0.8 theta=90 nxa=32 nxi=48 dt=0.025 nsteps=800 coupling=scalar
    run p4_sca_t0         slab $P4 k1=28 k2=0.8 k3=0.8 theta=0  nxa=16 nxi=24 dt=0.05 nsteps=400 coupling=scalar
    run p4_sca_kkk        slab $P4 k1=2  k2=2   k3=2   theta=0  nxa=16 nxi=24 dt=0.05 nsteps=400 coupling=scalar
    run p4_dir_kkk        slab $P4 k1=2  k2=2   k3=2   theta=0  nxa=16 nxi=24 dt=0.05 nsteps=400 coupling=directional
    run p4_fun_t90_kc5    slab $P4 k1=28 k2=0.8 k3=0.8 theta=90 nxa=16 nxi=24 dt=0.05 nsteps=400 coupling=function kc=5
    run p4_dir_t90        slab $P4 k1=28 k2=0.8 k3=0.8 theta=90 nxa=16 nxi=24 dt=0.05 nsteps=400 coupling=directional
fi

# ---------------------------------------------------------------- P4 with four coupling passes (staggering term removed)
if want p4o; then
    P4="ki=0.15 La=0.004 Li=0.006 rhoA=2000 cpA=1000 rhoI=1000 cpI=1500 xbc=NN ic=step Ta0=350 Ti0=300 nwrite=2 nxa=16 nxi=24 dt=0.05 nsteps=400 nouter=4"
    run p4_sca_t90_o4     slab $P4 k1=28 k2=0.8 k3=0.8 theta=90 coupling=scalar
    run p4_sca_t0_o4      slab $P4 k1=28 k2=0.8 k3=0.8 theta=0  coupling=scalar
    run p4_sca_kkk_o4     slab $P4 k1=2  k2=2   k3=2   theta=0  coupling=scalar
    run p4_fun_t90_kc5_o4 slab $P4 k1=28 k2=0.8 k3=0.8 theta=90 coupling=function kc=5
    run p4_dir_t90_o4     slab $P4 k1=28 k2=0.8 k3=0.8 theta=90 coupling=directional
fi

# ---------------------------------------------------------------- probes (not part of the four problems)
if want probes; then
    PX="k1=28 k2=0.8 k3=0.8 theta=90 ki=0.15 La=0.004 Li=0.006 xbc=NN ic=step Ta0=350 Ti0=300 nxa=16 nxi=24 dt=0.5 nsteps=200 nwrite=4 implicit=yes"
    run x_impl_dir_eqcp   slab $PX rhoA=2000 cpA=1000 rhoI=1500 cpI=1000 coupling=directional
    run x_impl_dir_cp15   slab $PX rhoA=2000 cpA=1000 rhoI=1000 cpI=1500 coupling=directional
    run x_impl_dir_cp15_o10 slab $PX rhoA=2000 cpA=1000 rhoI=1000 cpI=1500 coupling=directional nouter=10
    run x_impl_sca_eqcp   slab $PX rhoA=2000 cpA=1000 rhoI=1500 cpI=1000 coupling=scalar
    run x_expl_dir_eqcp   slab ${PX/implicit=yes/implicit=no} rhoA=2000 cpA=1000 rhoI=1500 cpI=1000 coupling=directional
fi
echo "=== finished $(date -Is)" | tee -a "$LOG"
