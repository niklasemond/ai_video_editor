"""Supervise one local process without changing macOS memory protections."""
import json
import pathlib
import signal
import subprocess
import sys
import time

import psutil


def reason(samples):
    if len(samples) >= 3 and all(s['pressure'] >= 4 for s in samples[-3:]):
        return 'Sustained critical macOS memory pressure'
    if len(samples) >= 7 and samples[-1]['swap'] - samples[-7]['swap'] > 2_000_000_000:
        return 'Swap grew by over 2 GB in one minute'
    if samples[-1]['available_disk'] < 30_000_000_000:
        return 'Available disk capacity fell below 30 GB reserve'
    return None


def sample():
    pressure = int(subprocess.check_output(['/usr/sbin/sysctl', '-n', 'kern.memorystatus_vm_pressure_level']))
    js = 'ObjC.import("Foundation"); var u=$.NSURL.fileURLWithPath("/System/Volumes/Data"); var v=Ref(); var e=Ref(); if(!u.getResourceValueForKeyError(v,$("NSURLVolumeAvailableCapacityForImportantUsageKey"),e)) throw Error("Capacity unavailable"); ObjC.unwrap(v[0]);'
    available = int(subprocess.check_output(['/usr/bin/osascript', '-l', 'JavaScript', '-e', js]))
    if available <= 0:
        raise RuntimeError('Cannot measure reclaimable disk capacity in this execution context')
    return dict(time=time.time(), pressure=pressure, swap=psutil.swap_memory().used,
                available_memory=psutil.virtual_memory().available, available_disk=available)


def process_usage(pid):
    """Include inference descendants when sandbox-exec remains a wrapper.

    Summed RSS may count shared pages more than once; system pressure/swap,
    rather than this diagnostic, govern termination.
    """
    root = psutil.Process(pid)
    try:
        members = [root] + root.children(recursive=True)
        scope = 'process_tree'
    except (psutil.AccessDenied, PermissionError):
        members = [root]
        scope = 'direct_child_only'
    rss, cpu = 0, 0.0
    for member in members:
        try:
            rss += member.memory_info().rss
            times = member.cpu_times()
            cpu += times.user + times.system
        except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
            pass
    return dict(rss=rss, cpu_seconds=cpu, process_count=len(members), usage_scope=scope)


def run(command, log):
    samples = [sample()]
    if reason(samples):
        raise RuntimeError(reason(samples))
    child = subprocess.Popen(command, start_new_session=True)
    try:
        with pathlib.Path(log).open('w') as stream:
            while child.poll() is None:
                s = sample()
                try:
                    s.update(process_usage(child.pid))
                except psutil.NoSuchProcess:
                    pass
                samples.append(s)
                stream.write(json.dumps(s) + '\n'); stream.flush()
                stop = reason(samples)
                if stop:
                    raise RuntimeError(stop)
                time.sleep(10)
        return child.wait()
    finally:
        if child.poll() is None:
            import os
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # The child can finish between poll() and killpg().
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()


def main():
    def cancel(signum, frame):
        # A second request must not interrupt the bounded child-group cleanup.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)
    try:
        return run(sys.argv[2:], sys.argv[1])
    except KeyboardInterrupt:
        print('Cancelled; inference process group terminated.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    sys.exit(main())
