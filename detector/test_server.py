#!/usr/bin/env python3
"""
Tests for the monitor server's two live switches — media recording and e-mail alerts.

They are worth a test because they are the only things in the system that can silently
stop it producing evidence: a switch that says ON while the detector believes OFF would
mean an event nobody has a clip or an e-mail for, and no way to tell from the page.

Runs a real server on an ephemeral port and talks to it over HTTP, because the parts
that can actually break (query parsing, CORS, the JSON contract the page reads) all
live in the handler rather than in the object.

Run:  ../.venv/bin/python detector/test_server.py
"""
import os, sys, json, tempfile, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import MonitorServer

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def get(path):
    with urllib.request.urlopen(f"{BASE}{path}", timeout=5) as r:
        return json.loads(r.read()), r.headers


tmp = tempfile.mkdtemp()
srv = MonitorServer(0, tmp)                       # port 0 = let the OS pick a free one
srv.start()
PORT = srv.httpd.server_address[1]
BASE = f"http://127.0.0.1:{PORT}"
flips = []
srv.on_switch = lambda name, on: flips.append((name, on))

print("1. the page can read both switches out of state.json")
st, hdrs = get("/state.json")
for k in ("auto_record", "email_alerts", "email_available"):
    check(f"state.json carries {k}", k in st)
check("CORS is open (the page is on another port)",
      hdrs.get("Access-Control-Allow-Origin") == "*")

print("\n2. recording switches off and on")
srv.email_available = True
srv.email_alerts = True
s, _ = get("/set?record=0")
check("record=0 turns it off", s["auto_record"] is False, str(s["auto_record"]))
check("…and state.json agrees", get("/state.json")[0]["auto_record"] is False)
check("…and the detector object agrees", srv.auto_record is False)
s, _ = get("/set?record=1")
check("record=1 turns it back on", s["auto_record"] is True)

print("\n3. setting is idempotent, not a toggle")
get("/set?record=0"); get("/set?record=0"); get("/set?record=0")
check("three identical offs leave it off", srv.auto_record is False)
get("/set?record=1")

print("\n4. e-mail cannot be armed when it is not configured")
srv.email_available = False
srv.email_alerts = False
s, _ = get("/set?email=1")
check("email=1 is refused with no e-mail configured", s["email_alerts"] is False,
      str(s["email_alerts"]))
check("…and the page is told why (email_available false)", s["email_available"] is False)
srv.email_available = True
s, _ = get("/set?email=1")
check("once configured, email=1 arms it", s["email_alerts"] is True)

print("\n5. both switches move together in one request")
s, _ = get("/set?record=0&email=0")
check("record and email both off", s["auto_record"] is False and s["email_alerts"] is False, str(s))

print("\n6. only real changes are announced to the detector")
flips.clear()
get("/set?record=0")                              # already off
check("a no-op change is not logged", flips == [], str(flips))
get("/set?record=1")
check("a real change is logged once", flips == [("auto_record", True)], str(flips))

print("\n7. junk input cannot wedge it")
for q in ("", "?", "?record=", "?record=banana", "?nonsense=1", "?record=0&record=1"):
    try:
        s, _ = get(f"/set{q}")
        ok = isinstance(s, dict) and "auto_record" in s
    except Exception as e:
        ok = False; s = e
    check(f"/set{q or '(bare)'} still answers sanely", ok, str(s)[:60])
# Only "0"/"false"/"off" mean off, so junk reads as ON. That is the safe direction for
# both switches — the failure mode is an extra clip or an extra e-mail, never a silently
# missing one. An omitted or empty value changes nothing at all.
check("junk reads as ON, which is the fail-safe direction",
      get("/set?record=banana")[0]["auto_record"] is True)
get("/set?record=0")
check("an omitted value leaves the switch alone",
      get("/set?nonsense=1")[0]["auto_record"] is False)
check("an empty value leaves the switch alone",
      get("/set?record=")[0]["auto_record"] is False)
get("/set?record=1")

print("\n8. an unknown switch name is ignored rather than setting an attribute")
check("set_switch rejects an unknown name", srv.set_switch("wipe_disk", True) is None)
check("…and no attribute appeared", not hasattr(srv, "wipe_disk"))

print("\n9. events with no media are representable (recording was off)")
srv.add_event("PERSON", None, "person:0.74")
check("a null snapshot survives into state.json",
      get("/state.json")[0]["events"][0]["snapshot"] is None)

srv.httpd.shutdown()
print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all monitor-server switch checks passed")
