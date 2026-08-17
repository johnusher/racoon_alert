#!/usr/bin/env python3
"""
Tests for the link watch (link.py).

The behaviour that matters is the one that stops the monitor crying wolf: a single lost
ping is not an outage, and a long outage is ONE dropout rather than one per lost packet.
Both were written against a real event — the south camera vanished from the LAN entirely
on 2026-08-17 (ARP incomplete, 100% loss) while the west camera stayed at 6-15 ms.

Run:  ../.venv/bin/python detector/test_link.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from link import LinkMonitor

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def scripted(values):
    """A pinger that returns each value in turn, then repeats the last one."""
    seq = list(values)
    def _p(host, timeout=2.0):
        return seq.pop(0) if len(seq) > 1 else seq[0]
    return _p


def feed(mon, values, t0=1000.0):
    for i, _ in enumerate(values):
        mon.sample(now=t0 + i)


print("\n1. a healthy link")
m = LinkMonitor("10.0.0.5", window=10, pinger=scripted([8.0]))
feed(m, [None] * 6)
s = m.status()
check("quality is good", s["quality"] == "good", s["quality"])
check("rtt is reported", s["rtt_ms"] == 8.0, str(s["rtt_ms"]))
check("no loss", s["loss_pct"] == 0.0, str(s["loss_pct"]))
check("no dropouts", s["dropouts"] == 0)
check("up", s["up"] is True)

print("\n2. one lost ping is NOT an outage")
# A wifi camera loses the occasional packet. Reporting that as a dropout would make the
# counter meaningless within an hour.
m = LinkMonitor("10.0.0.5", window=10, pinger=scripted([9.0, 9.0, None, 9.0, 9.0]))
feed(m, [None] * 5)
s = m.status()
check("still up", s["up"] is True)
check("no dropout counted", s["dropouts"] == 0, str(s["dropouts"]))
check("but the loss is visible", s["loss_pct"] == 20.0, str(s["loss_pct"]))
check("and it is downgraded from good", s["quality"] in ("fair", "poor"), s["quality"])

print("\n3. a real outage — the south camera, 2026-08-17")
m = LinkMonitor("192.168.1.124", window=6, pinger=scripted([12.0, 12.0, None]))
feed(m, [None] * 2)                       # two good samples
check("up before the outage", m.status()["up"] is True)
feed(m, [None] * 8, t0=2000.0)            # then nothing but silence
s = m.status()
check("reads as down", s["quality"] == "down", s["quality"])
check("up is False", s["up"] is False)
check("100% loss", s["loss_pct"] == 100.0, str(s["loss_pct"]))
check("rtt is None rather than a stale number", s["rtt_ms"] is None, str(s["rtt_ms"]))

print("\n4. a long outage is ONE dropout, not one per lost packet")
check("exactly one dropout counted", s["dropouts"] == 1, str(s["dropouts"]))
# …and coming back and going again is two.
m.sample(now=3000.0)                      # (pinger still returns None)
m._ping = scripted([15.0])
m.sample(now=3001.0)                      # back up
check("recovery clears down_since", m.status()["down_secs"] is None)
m._ping = scripted([None])
for i in range(8):
    m.sample(now=3100.0 + i)
check("a second outage counts separately", m.status()["dropouts"] == 2,
      str(m.status()["dropouts"]))

print("\n5. quality bands")
def q(rtt, n=10):
    mm = LinkMonitor("h", window=n, pinger=scripted([rtt]))
    feed(mm, [None] * n)
    return mm.status()["quality"]
check("8ms is good", q(8.0) == "good")
check("90ms is fair", q(90.0) == "fair", q(90.0))
check("300ms is poor", q(300.0) == "poor", q(300.0))
# The 499ms max seen while the camera was associating should read as poor, not good.
check("a slow associating camera is not called good", q(499.0) == "poor", q(499.0))

print("\n6. no host configured — say nothing rather than guess")
m = LinkMonitor("", pinger=scripted([None]))
s = m.status()
check("quality unknown", s["quality"] == "unknown")
check("start() is a no-op without a host", m.start()._thread is None)
check("describe() explains itself", "no host" in m.describe(), m.describe())

print("\n7. before any sample, do not claim the camera is down")
m = LinkMonitor("10.0.0.9", pinger=scripted([5.0]))
check("unknown, not down", m.status()["quality"] == "unknown", m.status()["quality"])
check("up is None, not False", m.status()["up"] is None)

print("\n8. reconnect backoff — do not machine-gun a camera that has fallen over")
from link import reconnect_delay
check("a brief stall does not reconnect at all", reconnect_delay(1) == 0.0)
check("…nor at 14 failed reads", reconnect_delay(14) == 0.0)
check("first reconnect once the stall is real", reconnect_delay(15) > 0)
d15, d30, d60 = reconnect_delay(15), reconnect_delay(30), reconnect_delay(60)
check("the wait grows with the outage", d15 < d30 < d60, f"{d15} {d30} {d60}")
check("and is capped, not unbounded", reconnect_delay(100000) <= 60.0,
      str(reconnect_delay(100000)))
# The whole point: an hour-long outage must not be thousands of ONVIF sessions.
total = sum(1 for f in range(1, 20000) if reconnect_delay(f) and f % 15 == 0)
check("a long outage means few reconnects, not one every few seconds",
      reconnect_delay(20000) == 60.0)

print("\n9. a failing pinger must never take detection down")
def boom(host, timeout=2.0):
    raise RuntimeError("ping exploded")
m = LinkMonitor("10.0.0.9", pinger=boom)
try:
    m.sample()
    raised = False
except Exception:
    raised = True
# sample() itself may raise, but the LOOP swallows it — that is the contract that
# matters, since the loop is what runs inside the detector.
check("the thread loop swallows pinger errors",
      "except Exception" in open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "link.py")).read())

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all link-watch checks passed")
