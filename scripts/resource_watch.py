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
                    s['rss'] = psutil.Process(child.pid).memory_info().rss
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
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


if __name__ == '__main__':
    sys.exit(run(sys.argv[2:], sys.argv[1]))
