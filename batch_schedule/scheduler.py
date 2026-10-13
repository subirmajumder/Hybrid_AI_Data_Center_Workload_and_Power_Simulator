# -*- coding: utf-8 -*-
"""
Event-driven GPU job scheduler simulator.
@author: Subir Majumder

GPU scheduler with:
    * Two scheduling policies: FCFS with backfilling (``FCFS_BKFL``) and
      smallest-work-first (``SWF``, sorted by GPUs then runtime).
    * Variable GPU capacity over time, modelled as a piecewise-constant
      function.
    * Job segmentation into fixed-size chunks with strict dependencies:
 
        - segment_1 arrives at the job's submit time.
        - segment_{i>1} is released only when segment_{i-1} finishes.

Time base:
- All times (jobs and capacity) are expressed as integer seconds since a
    base timestamp.
- Base --first-day (ISO-8601).

"""

from pathlib import Path
import bisect
import csv
import heapq
import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from operator import attrgetter

import pandas as pd

logger = logging.getLogger(__name__)

##################################################
# Helpers

def parse_datetime_to_unix_int(ts: str) -> int:
    """ISO-8601 string -> Unix seconds as an int."""
    return int(pd.to_datetime(ts).value // 1_000_000_000)


##################################################
# Models

@dataclass
class JobSeg:
    """One job segment."""

    id: str              # e.g. "123_1"
    parent_id: int       # original id_job
    arrival: int         # release time (seconds since base)
    gpus: int
    runtime: int         # actual execution time for this segment (seconds)
    timelimit: int       # scheduler timelimit (seconds) for this segment

    start_time: Optional[int] = None
    end_time: Optional[int] = None
    status: str = "OK"   # OK, UNSCHEDULABLE_GPU, UNSCHEDULABLE_FEAS, CANCELLED_CHAIN
    reserved_start: Optional[int] = None  # reservation start for head job


##################################################
# CSV loaders

def load_jobs(
    csv_path: str,
    segment_duration_sec: int,
    base_ts: int,
    normalise: bool = True,
) -> Tuple[List[JobSeg], List[JobSeg], Dict[str, str]]:
    """
    Load jobs from a CSV file and split them into dependent segments.
    The CSV must contain the columns ``id_job``, ``time_submit``,
    ``gpu_alloc``, ``timelimit`` (in minutes) and ``runtime`` (in seconds).
    Jobs with a non-positive runtime are skipped.
    
    Each job is split into ``ceil(runtime / segment_duration_sec)`` segments.

    Only the first segment of a job gets its real arrival time; later segments 
    are released by the simulator when their predecessor finishes.
    
    Returns:
        A tuple ``(all_segs, root_segs, succ_map)`` where:
    
        * ``all_segs`` is every segment of every job.
        * ``root_segs`` holds the first segment of each job, sorted by
          arrival time.
        * ``succ_map`` maps a segment id to the id of its successor.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        return [], [], {}

    submit_times = [parse_datetime_to_unix_int(r["time_submit"]) for r in rows]

    if normalise:
        base = base_ts
    else:
        base = 0

    all_segs: List[JobSeg] = []
    root_segs: List[JobSeg] = []
    succ_map: Dict[str, str] = {}

    for r, raw_ts in zip(rows, submit_times):
        parent_id = int(r["id_job"])
        arrival0 = int(raw_ts - base)
        gpus = int(r["gpu_alloc"])

        # NOTE: timelimit is actually in minutes in the CSV
        job_timelimit_sec = int(float(r["timelimit"]) * 60)
        total_runtime_sec = int(float(r["runtime"]))

        if total_runtime_sec <= 0:
            continue

        num_segments = int(math.ceil(total_runtime_sec / segment_duration_sec))
        prev_id: Optional[str] = None

        for i in range(num_segments):
            remaining_runtime   = total_runtime_sec  - i * segment_duration_sec
            remaining_timelimit = job_timelimit_sec - i * segment_duration_sec

            seg_runtime   = int(min(segment_duration_sec, remaining_runtime))
            seg_timelimit = int(min(segment_duration_sec, remaining_timelimit))

            # discard zero-runtime segments.
            if seg_runtime <= 0:
                continue

            seg_id = f"{parent_id}_{i + 1}"
            # successors released on predecessor finish; this initial 0 is just a placeholder
            seg_arrival = arrival0 if i == 0 else 0

            seg = JobSeg(
                id=seg_id,
                parent_id=parent_id,
                arrival=seg_arrival,
                gpus=gpus,
                runtime=seg_runtime,
                timelimit=seg_timelimit,
            )

            all_segs.append(seg)
            if i == 0:
                root_segs.append(seg)
            if prev_id is not None:
                succ_map[prev_id] = seg_id
            prev_id = seg_id

    root_segs.sort(key=lambda j: j.arrival)
    return all_segs, root_segs, succ_map


def load_capacity(
    csv_path: str,
    base_ts: int,
    normalise: bool = True,
) -> Tuple[List[int], List[int]]:
    """
    Load available GPU capacity timeline from a CSV file.
    
    The CSV must have one column whose name starts with ``time`` and one
    whose name starts with ``gpu`` (case-insensitive for ``time``). Each row
    gives the capacity that takes effect at that time and holds until the
    next row.

    Returns:
        A tuple ``(change_times, caps)`` of equal-length lists, sorted by
        time. Both are empty if the CSV has no rows.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], []

    tkey = next(k for k in rows[0] if k.lower().startswith("time"))
    gkey = next(k for k in rows[0] if k.lower().startswith("gpu"))

    raw_times = [parse_datetime_to_unix_int(r[tkey]) for r in rows]

    if normalise:
        base = base_ts
    else:
        base = 0

    times = [int(t - base) for t in raw_times]
    caps = [int(r[gkey]) for r in rows]

    pairs = sorted(zip(times, caps))
    times = [t for t, _ in pairs]
    caps = [c for _, c in pairs]

    return times, caps


##################################################
# Capacity timeline -> GOOD_INTERVALS

def build_good_intervals(
    change_times: List[int],
    caps: List[int],
) -> Tuple[
    Dict[int, List[Tuple[int, int]]],
    Dict[int, List[int]],
    Dict[int, List[int]],
]:
    """
    Precompute, for each GPU count ``g``, the intervals where capacity >= ``g``.
 
    The last interval of a given ``g`` is open-ended (end = ``math.inf``) if
    capacity is still at least ``g`` after the final change point.
    
    Returns:
        A tuple ``(good, starts, suffix_max_len)``:
 
        * ``good[g]``: list of ``(a, b)`` intervals where capacity >= ``g``.
        * ``starts[g]``: the interval start times ``[a0, a1, ...]``, for
          binary search.
        * ``suffix_max_len[g]``: list where entry ``j`` is the maximum
          interval length among intervals ``j`` onward
          (``max_{k >= j} (b_k - a_k)``; may be ``inf``).
    """
    max_cap = max(caps) if caps else 0
    good: Dict[int, List[Tuple[int, int]]] = {}
    starts: Dict[int, List[int]] = {}
    suffix_max_len: Dict[int, List[int]] = {}

    for g in range(1, max_cap + 1):
        intervals: List[Tuple[int, int]] = []
        block_start: Optional[int] = None
        for idx, t in enumerate(change_times):
            cap_here = caps[idx]
            if cap_here >= g:
                if block_start is None:
                    block_start = t
            else:
                if block_start is not None:
                    intervals.append((block_start, t))
                    block_start = None
        if block_start is not None:
            intervals.append((block_start, math.inf))

        good[g] = intervals
        starts[g] = [a for a, _ in intervals]

        # build suffix max lengths
        L = [0] * len(intervals)
        cur_max = 0
        for j in range(len(intervals) - 1, -1, -1):
            a, b = intervals[j]
            length = b - a  # may be inf
            if math.isinf(length):
                cur_max = math.inf
            else:
                cur_max = max(cur_max, length)
            L[j] = cur_max
        suffix_max_len[g] = L

    return good, starts, suffix_max_len


def update_capacity(
    time: int,
    change_times: List[int],
    caps: List[int],
    cap_idx: int,
) -> Tuple[int, int]:
    """
    Moves ``cap_idx`` forward by at most one step, so that
    ``change_times[cap_idx] <= time < change_times[cap_idx + 1]`` holds
    provided no more than one change point was crossed.
    """
    n = len(change_times)
    if cap_idx + 1 < n and change_times[cap_idx + 1] <= time:
        cap_idx += 1
    return cap_idx, caps[cap_idx]

##################################################
# Shape-level checks using GOOD_INTERVALS (for backfill + head)

def shape_can_run_from_time(
    start: int,
    gpus: int,
    dur: int,
    used_at_start: int,
    good_intervals: Dict[int, List[Tuple[int, int]]],
    good_starts: Dict[int, List[int]],
    max_cap: int,
) -> bool:
    """
    Check whether a job "shape" fits starting at a given time.
 
    A shape is a ``(gpus, dur)`` pair. The check assumes other jobs hold
    ``used_at_start`` GPUs constantly over the whole window, so the job fits
    only if capacity stays at or above ``used_at_start + gpus`` from
    ``start`` to ``start + dur``.
    """
    if dur <= 0:
        return True

    required = used_at_start + gpus
    if required <= 0:
        return True
    if required > max_cap:
        return False

    intervals = good_intervals.get(required)
    starts = good_starts.get(required)
    if not intervals or not starts:
        return False

    end_time = start + dur

    idx = bisect.bisect_right(starts, start) - 1
    if idx < 0:
        return False

    a, b = intervals[idx]
    if not (a <= start < b):
        return False

    return end_time <= b


def shape_can_run_now(
    now: int,
    gpus: int,
    dur: int,
    used_now: int,
    good_intervals: Dict[int, List[Tuple[int, int]]],
    good_starts: Dict[int, List[int]],
    max_cap: int,
) -> bool:
    """Check whether a job shape fits starting at the current time.

    """
    return shape_can_run_from_time(
        start=now,
        gpus=gpus,
        dur=dur,
        used_at_start=used_now,
        good_intervals=good_intervals,
        good_starts=good_starts,
        max_cap=max_cap,
    )

##################################################
# Head reservation: scan over job end times

def earliest_capacity_start(
    t: int,
    dur: int,
    required: int,
    good_intervals: Dict[int, List[Tuple[int, int]]],
    good_starts: Dict[int, List[int]],
    suffix_max_len: Dict[int, List[int]],
) -> float:
    """
    Find the earliest time the raw capacity alone can host a job.
 
    Searches for the earliest time ``>= t`` at which ``required`` GPUs are
    available continuously for ``dur`` seconds, ignoring any future
    releases by running jobs. ``suffix_max_len[required]`` is used to rule
    out quickly the cases where no remaining interval is long enough.
    """
    if dur <= 0:
        return t

    intervals = good_intervals.get(required)
    starts = good_starts.get(required)
    suffix = suffix_max_len.get(required)
    if not intervals or not starts or not suffix:
        return math.inf

    end_needed = t + dur

    # Can we start at t
    idx = bisect.bisect_right(starts, t)

    if idx > 0:
        a, b = intervals[idx - 1]
        if a <= t < b and end_needed <= b:
            return t

    if idx >= len(intervals):
        return math.inf

    # Quick duration-feasibility check:
    if not math.isinf(suffix[idx]) and suffix[idx] < dur:
        return math.inf

    # Otherwise, scan later intervals until one is long enough
    for j in range(idx, len(intervals)):
        a, b = intervals[j]
        cand = max(a, t)
        if cand + dur <= b:
            return cand

    return math.inf


def head_reservation_time(
    now: int,
    head: JobSeg,
    running: List[Tuple[int, str]],          # (end_time, seg_id)
    job_map: Dict[str, JobSeg],
    used_now: int,
    good_intervals: Dict[int, List[Tuple[int, int]]],
    good_starts: Dict[int, List[int]],
    suffix_max_len: Dict[int, List[int]],
    max_cap: int,
) -> float:
    """
    Compute the earliest time the head-of-queue job can start.
 
    Finds the earliest ``R >= now`` such that the head job can run for its
    full time limit. Candidate times are ``now`` and the end times of
    running segments; at each one, GPUs held by finished segments are
    released and :func:`earliest_capacity_start` is queried for the
    remaining requirement.
    
    """
    if head.timelimit <= 0:
        return now

    # Build sorted list of future finishes: (end_time, gpus)
    events: List[Tuple[int, int]] = []
    for end_t, seg_id in running:
        if end_t > now:
            seg = job_map[seg_id]
            events.append((end_t, seg.gpus))

    events.sort()  # sort once by end_time

    used = used_now
    t = now
    idx = 0  # pointer into events

    while True:
        while idx < len(events) and events[idx][0] == t:
            _, freed = events[idx]
            used -= freed
            idx += 1

        if used < 0: 
            used = 0
            
        required = used + head.gpus

        if required <= 0:
            cap_start = t
        elif required > max_cap:
            cap_start = math.inf
        else:
            cap_start = earliest_capacity_start(
                t=t,
                dur=head.timelimit,
                required=required,
                good_intervals=good_intervals,
                good_starts=good_starts,
                suffix_max_len=suffix_max_len,
            )

        if idx >= len(events):
            return cap_start

        next_finish = events[idx][0]

        if cap_start < next_finish:
            return cap_start

        t = next_finish


##################################################
# Core simulation

def cancel_chain(start_seg: JobSeg, job_map: Dict[str, JobSeg], succ_map: Dict[str, str], reason: str):
    """Mark this segment and all successors as cancelled/unschedulable."""
    sid = start_seg.id
    while True:
        seg = job_map[sid]
        if seg.status == "OK":
            seg.status = reason
        nxt = succ_map.get(sid)
        if nxt is None:
            break
        sid = nxt


def simulate_schedule(
    all_segs: List[JobSeg],
    root_segs: List[JobSeg],
    succ_map: Dict[str, str],
    cap_times: List[int],
    caps: List[int],
    good_intervals: Dict[int, List[Tuple[int, int]]],
    good_starts: Dict[int, List[int]],
    suffix_max_len: Dict[int, List[int]],
    scheduler: str = "FCFS_BKFL",
    progress=None,
):
    """
    Run the event-driven GPU scheduling simulation.
 
    Segments are modified in place: ``start_time``, ``end_time``,
    ``status`` and (for non-root segments) ``arrival`` are filled in.
 
    Two policies are supported:
 
    * ``"FCFS_BKFL"``: first-come-first-served with backfilling.
 
        1. Start as many queue heads as can start now.
        2. For the first head that cannot start now, compute its
           reservation time once. If it can never run, the head and all
           its successors are cancelled.
        3. Backfill pass: a single pass over the other queued segments,
           starting those that fit now and finish before the reservation.
           Shapes that already failed are used to prune dominated ones.
 
    * ``"SWF"``: queue sorted by ``(gpus, runtime)``; starts every segment
      that fits, scanning from the smallest.
 
    The simulation jumps between events: segment finishes, root arrivals,
    capacity changes and head reservations. Successor segments are released
    when their predecessor finishes.
    """
    if not all_segs or not cap_times:
        return

    job_map: Dict[str, JobSeg] = {s.id: s for s in all_segs}
    max_cap = max(caps)

    # Pre-mark segments that can never fit by GPU count
    for seg in all_segs:
        if seg.gpus > max_cap:
            seg.status = "UNSCHEDULABLE_GPU"

    # Simulation state
    time = 0
    cap_idx = 0
    capacity = caps[0]
    used_gpus = 0

    queue: List[JobSeg] = []
    running: List[Tuple[int, str]] = []  # heap of (end_time, seg_id)

    # Root segments: only OK ones; should already be sorted by arrival
    root_segs = [s for s in root_segs if s.status == "OK"]
    root_idx = 0
    
    # Queue state
    queue_dirty = True

    def release_root_arrivals():
        nonlocal root_idx
        while root_idx < len(root_segs) and root_segs[root_idx].arrival <= time:
            seg = root_segs[root_idx]
            if seg.status == "OK":
                queue.append(seg)
                queue_dirty = True
            root_idx += 1

    def start_segment(seg: JobSeg):
        nonlocal used_gpus, time
        # runtime should be > 0 by construction; this is defensive
        if seg.runtime <= 0:
            seg.start_time = time
            seg.end_time = time
            return
        seg.start_time = time
        seg.end_time = time + seg.runtime
        used_gpus += seg.gpus
        heapq.heappush(running, (seg.end_time, seg.id))
        
        # logger.info(
        # "START t=%d seg=%s parent=%d gpus=%d used=%d cap=%d free=%d end=%d",
        # time, seg.id, seg.parent_id, seg.gpus, used_gpus, capacity, capacity - used_gpus, seg.end_time)

    while True:
        # External arrivals at current time
        release_root_arrivals()

        if scheduler == "SWF":
            if queue_dirty:
                queue[:] = [seg for seg in queue if seg.status == "OK"]
                queue.sort(key=attrgetter("gpus", "runtime"))
                queue_dirty = False

            # pack as many jobs as possible now, starting from the smallest
            idx = 0
            while idx < len(queue):
                capacity = caps[cap_idx]
                free = capacity - used_gpus
                if free <= 0:
                    break

                seg = queue[idx]
                g = seg.gpus
                d = seg.timelimit

                if g > free:
                    idx += 1
                    continue

                if not shape_can_run_now(
                    now=time,
                    gpus=g,
                    dur=d,
                    used_now=used_gpus,
                    good_intervals=good_intervals,
                    good_starts=good_starts,
                    max_cap=max_cap,
                ):
                    idx += 1
                    continue

                queue.pop(idx)
                start_segment(seg)
                # do not increment idx; list shrank

        elif scheduler == "FCFS_BKFL":
            
            while queue:
            # HEAD PASS: start as many heads as can start now
                capacity = caps[cap_idx]
                head = queue[0]
    
                # Drop broken head
                if head.status != "OK":
                    queue.pop(0)
                    continue
    
                # Compute reservation ONCE for this head
                if head.reserved_start is None:
                    R = head_reservation_time(
                        now=time,
                        head=head,
                        running=running,          
                        job_map=job_map,
                        used_now=used_gpus,
                        good_intervals=good_intervals,
                        good_starts=good_starts,
                        suffix_max_len=suffix_max_len,
                        max_cap=max_cap,
                    )
    
                    if math.isinf(R):
                        # Head can NEVER be scheduled -> mark and drop whole chain
                        head.status = "UNSCHEDULABLE_FEAS"
                        cancel_chain(head, job_map, succ_map, "CANCELLED_CHAIN")
                        queue.pop(0)
                        # Try next head at same time
                        continue
    
                    head.reserved_start = int(R)
    
                # At this point, head has a finite reservation
                R = head.reserved_start  # type: ignore[assignment]
    
                if R == time:
                    # Head can and should start now
                    queue.pop(0)
                    start_segment(head)
                    # used_gpus changed; try next head at same time
                    continue
    
                # This is the first head that cannot start now (R > time).
                # Stop HEAD PASS; we'll use R for backfill.
                break
    
            # BACKFILL PASS: single pass over non-head jobs, if there is a waiting head
            if queue and queue[0].status == "OK" and queue[0].reserved_start is not None:
                head = queue[0]
                R = head.reserved_start
                if R > time:
                    head_wait = R - time
                    failed_shapes: List[Tuple[int, int]] = []
    
                    idx = 1  # skip head at index 0
                    while idx < len(queue):
                        capacity = caps[cap_idx]
                        free = capacity - used_gpus
                        if free <= 0:
                            break
    
                        seg = queue[idx]
                        if seg.status != "OK":
                            queue.pop(idx)
                            continue
    
                        g = seg.gpus
                        d = seg.timelimit
    
                        # Must finish before head's reservation time
                        if d > head_wait:
                            idx += 1
                            continue
    
                        # Quick free-GPU check
                        if g > free:
                            idx += 1
                            continue
    
                        # Monotone pruning: if dominated by a failed shape, skip
                        dominated = False
                        for gf, df in failed_shapes:
                            if g >= gf and d >= df:
                                dominated = True
                                break
                        if dominated:
                            idx += 1
                            continue
    
                        # Full feasibility check from 'time' with GOOD_INTERVALS
                        if not shape_can_run_now(
                            now=time,
                            gpus=g,
                            dur=d,
                            used_now=used_gpus,
                            good_intervals=good_intervals,
                            good_starts=good_starts,
                            max_cap=max_cap,
                        ):
                            failed_shapes.append((g, d))
                            idx += 1
                            continue
    
                        # Start this backfill job at current 'time'
                        queue.pop(idx)
                        start_segment(seg)
                        
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        # Determine next event time: finish / arrival / capacity change / head reservation
        next_finish = running[0][0] if running else math.inf
        next_arrival = root_segs[root_idx].arrival if root_idx < len(root_segs) else math.inf
        next_cap = cap_times[cap_idx + 1] if cap_idx + 1 < len(cap_times) else math.inf

        # Head reservation as an event (so we don't jump past the reserved start)
        if scheduler == "FCFS_BKFL" and queue and queue[0].status == "OK" and queue[0].reserved_start is not None:
            next_R = queue[0].reserved_start
        else:
            next_R = math.inf

        if (
            math.isinf(next_finish)
            and math.isinf(next_arrival)
            and math.isinf(next_cap)
            and math.isinf(next_R)
            and not queue
        ):
            break  # nothing more can happen

        next_time = min(next_finish, next_arrival, next_cap, next_R)
        if math.isinf(next_time):
            break
        
        # Update progress bar based on how much simulated time advanced
        if progress is not None:
            delta = next_time - time
            if delta > 0:
                # If our initial total was too small, grow it dynamically
                if next_time > progress.total:
                    progress.total = next_time
                    progress.refresh()
                progress.update(delta)

        # Advance time and capacity index
        time = next_time
        cap_idx, capacity = update_capacity(time, cap_times, caps, cap_idx)

        # Process finishes at this time
        while running and running[0][0] <= time:
            end_t, seg_id = heapq.heappop(running)
            seg = job_map[seg_id]
            if seg.end_time is None:
                seg.end_time = end_t
            used_gpus -= seg.gpus
            
            # logger.info(
            # "END   t=%d seg=%s parent=%d gpus=%d used=%d cap=%d free=%d",
            # time, seg.id, seg.parent_id, seg.gpus, used_gpus, capacity, capacity - used_gpus)

            # Release successor segment in dependency chain
            succ_id = succ_map.get(seg_id)
            if succ_id is not None:
                succ = job_map[succ_id]
                if succ.status == "OK" and succ.start_time is None:
                    succ.arrival = time
                    queue.append(succ)
                    queue_dirty = True
                    
    # Progress bar
    if progress is not None:
        if time > progress.total:
            progress.total = time
        if progress.n < time:
            progress.update(time - progress.n)
        progress.refresh()

##################################################
# Results writer

def write_results(
    all_segs: List[JobSeg],
    in_csv: str,
    out_csv: str,
    base_ts: int,
):
    """
    Write the scheduled segments to a CSV file.
    """
    with open(in_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        logger.warning("No input rows found when writing results.")
        return

    row_map = {int(r["id_job"]): r for r in rows}

    def to_ts(x: Optional[int]) -> pd.Timestamp:
        # seconds since base_ts -> naive timestamp (no timezone)
        return (
            pd.NaT
            if x is None
            else pd.to_datetime(base_ts + int(x), unit="s")
            )

    records: List[Dict[str, object]] = []
    for seg in all_segs:
        r = row_map.get(seg.parent_id)
        if r is None:
            continue

        rec: Dict[str, object] = {
            "id_job": seg.id,
            "parent_id": seg.parent_id,
            "time_submit": pd.to_datetime(r["time_submit"]),
            "segment_arrival": to_ts(seg.arrival),
            "gpu_alloc": r.get("gpu_alloc", ""),
            "start_time": to_ts(seg.start_time),
            "end_time": to_ts(seg.end_time),
            "status": seg.status,
            "cluster_id": r.get("cluster_id", ""),
            "timelimit": r.get("timelimit", ""),
            "runtime": r.get("runtime", ""),
        }
        records.append(rec)

    pd.DataFrame(records).to_csv(out_csv, index=False)
    logger.info("Results written to %s", out_csv)

##################################################
# Orchestration

def simulate(
    jobs_csv,
    cap_csv,
    output_csv,
    log_level = "INFO",
    first_day = None,
    scheduler = "FCFS_BKFL",
    segment_duration= 365*24*3600  # 365 day segments (seconds)
):
    """
    Run the full scheduling pipeline: load inputs, simulate, write results.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if first_day is None:
        raise ValueError("Please use --first-day.")

    base_ts = parse_datetime_to_unix_int(ts=first_day)

    ALL_SEGS, ROOT_SEGS, SUCC_MAP = load_jobs(csv_path=jobs_csv,
                                                segment_duration_sec=segment_duration,
                                                base_ts=base_ts, normalise=True,) 
    
    CAP_TIMES, CAPS = load_capacity(csv_path= cap_csv,
                                    base_ts=base_ts,
                                    normalise = True,)

    GOOD_INTERVALS, GOOD_STARTS, SUFFIX_MAX_LEN = build_good_intervals(change_times=CAP_TIMES, caps=CAPS)

    logger.info(
        "Loaded %d segments (%d roots); capacity timeline has %d change points; max cap %d", len(ALL_SEGS), len(ROOT_SEGS), len(CAP_TIMES), max(CAPS),)

    total_time_hint = CAP_TIMES[-1] if CAP_TIMES else 1
    with tqdm(total=total_time_hint, desc="Simulated time", unit="s") as pbar:

        simulate_schedule(all_segs=ALL_SEGS, 
                          root_segs=ROOT_SEGS,
                          succ_map=SUCC_MAP, 
                          cap_times=CAP_TIMES, 
                          caps=CAPS, 
                          good_intervals=GOOD_INTERVALS, 
                          good_starts=GOOD_STARTS, 
                          suffix_max_len=SUFFIX_MAX_LEN,
                          scheduler= scheduler,
                          progress=pbar,)
    
    write_results(all_segs=ALL_SEGS,
                        in_csv=jobs_csv,
                        out_csv=output_csv,
                        base_ts=base_ts,)

##################################################
# main

if __name__ == "__main__":

    # Defaults
    here = Path(__file__).resolve()
    root = here.parent.parent
    SIMULATION_START_DATE="2026-01-03"
    SEGMENT_DURATION = 2*24*3600
    
    DAILY_TIMESTAMPED_RESOURCE = root / "Temp" / "synthetic_second_arrivals_with_attrs.csv"
    CAPACITY_CSV = root / "Temp" / "gpu_capacity.csv"
    OUTPUT_SCHEDULED_JOBS_CSV = root / "Temp" / "scheduled_synthetic_jobs.csv"

    simulate(
        jobs_csv=DAILY_TIMESTAMPED_RESOURCE, 
        cap_csv=CAPACITY_CSV, 
        output_csv=OUTPUT_SCHEDULED_JOBS_CSV, 
        log_level="INFO", 
        first_day=SIMULATION_START_DATE,
        segment_duration= SEGMENT_DURATION,
        scheduler="SWF") 
