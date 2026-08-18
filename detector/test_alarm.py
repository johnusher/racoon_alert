#!/usr/bin/env python3
"""
Tests for the local alarm (alarm.py).

Nothing here makes a sound: the thing that runs the commands is injected, so the tests
assert on what WOULD have been run. The behaviours that matter are the two that decide
whether a safety alarm is trustworthy — it must not machine-gun, and it must never be
able to take the detection loop down with it.

Run:  ../.venv/bin/python detector/test_alarm.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alarm import Alarm

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def spy():
    calls = []
    return calls, lambda cmd: calls.append(cmd)


def settle():
    time.sleep(0.25)                      # the noise happens on its own thread


print("\n1. it fires, and it makes both a noise and words")
calls, run = spy()
a = Alarm(repeat=2, runner=run)
a.available = True
check("fire() reports it fired", a.fire("Child at the gate", now=1000.0))
settle()
check("the sound is played `repeat` times", sum(c[0] == "afplay" for c in calls) == 2, str(calls))
check("and the words are spoken once", sum(c[0] == "say" for c in calls) == 1, str(calls))
check("the words are the ones passed in", any("Child at the gate" in c for c in calls), str(calls))

print("\n1b. the alarm speaks English, whatever language the camera greets in")
# talk.py greets a visitor at the gate with "Hallo." in Anna (German), which is right for
# Berlin. This is shouted across the room at the household and must not inherit it.
import json as _json, os as _os
_cfg = _json.load(open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.json")))
_alarm_voice = (_cfg.get("alarm") or {}).get("voice")
_greet_voice = (_cfg.get("talk") or {}).get("greet_voice")
import subprocess as _sp
_voices = _sp.run(["say", "-v", "?"], capture_output=True, text=True).stdout
_lang = {l.split()[0]: l for l in _voices.splitlines() if l.strip()}
check("the configured alarm voice is not the German greeting voice",
      _alarm_voice != _greet_voice, f"alarm={_alarm_voice} greet={_greet_voice}")
check("and it is an installed English voice",
      any(ln.startswith(_alarm_voice) and " en_" in ln for ln in _voices.splitlines()),
      f"{_alarm_voice}: " + next((ln for ln in _voices.splitlines()
                                  if ln.startswith(str(_alarm_voice))), "not installed"))

print("\n2. it will not machine-gun")
calls, run = spy()
a = Alarm(repeat=1, min_gap_secs=20.0, runner=run)
a.available = True
check("the first fires", a.fire("x", now=1000.0))
check("a second a second later does not", not a.fire("x", now=1001.0))
check("nor at 19s", not a.fire("x", now=1019.0))
check("but it does at 21s", a.fire("x", now=1021.0))

print("\n3. different alarms have their own clocks")
calls, run = spy()
a = Alarm(repeat=1, min_gap_secs=20.0, runner=run)
a.available = True
a.fire("gate", key="gate-opened", now=1000.0)
check("a different alarm is not blocked by the first",
      a.fire("child", key="child-at-gate", now=1001.0),
      "the gate opening must never swallow the child alarm")

print("\n4. switched off, or on a machine that cannot make noise")
calls, run = spy()
a = Alarm(enabled=False, runner=run)
a.available = True
check("disabled fires nothing", not a.fire("x", now=1000.0))
settle()
check("and runs nothing", calls == [], str(calls))
a2 = Alarm(runner=run)
a2.available = False
check("no afplay/say means no alarm", not a2.fire("x", now=1000.0))
check("and it says so", "off" in a2.describe(), a2.describe())

print("\n5. a broken sound player cannot take detection down")
def explode(cmd):
    raise OSError("no such thing as afplay")
a = Alarm(repeat=1, runner=explode)
a.available = True
fired = a.fire("x", now=1000.0)
settle()
check("fire() still returns cleanly", fired is True, "the thread swallowed the error")

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all alarm tests passed")
