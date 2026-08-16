#!/usr/bin/env python3
"""Email alerts for the h32 detector. Optional — only active if detector/secrets.json is set up."""
import os, json, ssl, smtplib, threading
from email.message import EmailMessage


class EmailNotifier:
    def __init__(self, secrets_path, email_cfg):
        self.cfg = email_cfg or {}
        self.smtp = {}
        self.enabled = False
        self.min_gap = self.cfg.get("min_gap_secs", 120)
        self._last = 0.0
        if self.cfg.get("enabled") and os.path.exists(secrets_path):
            try:
                self.smtp = json.load(open(secrets_path)).get("smtp", {})
                self.enabled = bool(self.smtp.get("host") and self.smtp.get("user") and self.cfg.get("to"))
            except Exception as e:
                print(f"[notify] secrets error: {e}")
        self.triggers = set(self.cfg.get("trigger_on", ["RACCOON", "ANIMAL", "PERSON"]))

    def maybe_alert(self, tag, detail, image_path):
        import time
        if not self.enabled or tag not in self.triggers:
            return
        if time.time() - self._last < self.min_gap:      # rate-limit emails
            return
        self._last = time.time()
        subject = f"[h32] {tag} detected"
        body = f"{tag} detected at {time.strftime('%Y-%m-%d %H:%M:%S')}.\nDetections: {detail}\nA clip is being recorded on the Mac (detector/events/)."
        threading.Thread(target=self._send, args=(subject, body, image_path), daemon=True).start()

    def _send(self, subject, body, image_path):
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = self.smtp.get("from", self.smtp["user"])
            msg["To"] = self.cfg["to"]
            msg.set_content(body)
            if image_path and os.path.exists(image_path):
                with open(image_path, "rb") as f:
                    msg.add_attachment(f.read(), maintype="image", subtype="jpeg",
                                       filename=os.path.basename(image_path))
            ctx = ssl.create_default_context()
            host, port = self.smtp["host"], int(self.smtp.get("port", 587))
            if port == 465:
                with smtplib.SMTP_SSL(host, port, context=ctx) as s:
                    s.login(self.smtp["user"], self.smtp["pass"]); s.send_message(msg)
            else:
                with smtplib.SMTP(host, port) as s:
                    s.starttls(context=ctx); s.login(self.smtp["user"], self.smtp["pass"]); s.send_message(msg)
            print(f"\n📧 alert emailed to {self.cfg['to']}")
        except Exception as e:
            print(f"\n[email failed: {e}]")


if __name__ == "__main__":
    # Verify email setup:  ../.venv/bin/python notify.py
    import sys
    base = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(base))
    import h32env                                          # alert address from local.env
    cfg = h32env.detector_config(os.path.join(base, "config.json"))
    n = EmailNotifier(os.path.join(base, "secrets.json"), {**cfg.get("email", {}), "enabled": True})
    if not n.enabled:
        print("Email not configured. Steps:\n"
              "  1. cp detector/secrets.json.example detector/secrets.json\n"
              "  2. fill in your SMTP host/user/pass (Gmail/Workspace: use a 16-char App Password)\n"
              "  3. set H32_EMAIL_TO in local.env (and email.enabled=true in detector/config.json\n"
              "     to arm real alerts)")
    else:
        print(f"Sending test email to {n.cfg['to']} via {n.smtp['host']}:{n.smtp.get('port',587)} …")
        n._send("[h32] test alert", "If you got this, h32 email alerts work. 🦝", None)
