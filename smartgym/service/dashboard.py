"""A zero-dependency review dashboard for match results.

The system's decisions must be *auditable*: a coach or referee has to be able to
see the score timeline, jump to any point and look at the proof image.  This
module serves exactly that from the standard library (``http.server``), so it
runs on the edge device or a laptop with no extra install:

    python -m smartgym.service.dashboard runs/badminton

It exposes:

* ``/``                      — score timeline with proof thumbnails
* ``/api/match``             — the full match JSON
* ``/api/points``            — the point list only
* ``/api/points/<n>``        — one point with its evidence
* ``/evidence/<file>``       — the proof images
* ``/export/timeline.csv``   — CSV download (spreadsheet friendly)
* ``/healthz``               — liveness probe for the IoT deployment

For a *live* stream, run the analyzer in ``tracking.mode: online`` and POST each
frame; the same JSON schema is served, so the same dashboard works.
"""

from __future__ import annotations

import json
import mimetypes
import os
import posixpath
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse

DEFAULT_MATCH_FILE = "match.json"

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartGym — match review</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
         background:#0e1116; color:#e6edf3; }
  header { padding:18px 24px; border-bottom:1px solid #21262d; display:flex; gap:24px;
           align-items:baseline; flex-wrap:wrap; background:#11161d; }
  h1 { font-size:18px; margin:0; letter-spacing:.3px; }
  .score { font-size:26px; font-weight:700; color:#7ee787; font-variant-numeric:tabular-nums; }
  .muted { color:#8b949e; font-size:13px; }
  main { padding:20px 24px 60px; max-width:1500px; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th, td { text-align:left; padding:9px 10px; border-bottom:1px solid #21262d; vertical-align:top; }
  th { color:#8b949e; font-weight:600; font-size:12px; text-transform:uppercase; letter-spacing:.5px; }
  tr:hover td { background:#161b22; }
  .pill { display:inline-block; padding:2px 8px; border-radius:99px; font-size:12px;
          background:#1f6feb22; color:#79c0ff; border:1px solid #1f6feb55; }
  .pill.warn { background:#bb800922; color:#e3b341; border-color:#bb800955; }
  .thumb { height:64px; border-radius:6px; border:1px solid #30363d; cursor:zoom-in; display:block; }
  a { color:#58a6ff; text-decoration:none; }
  a:hover { text-decoration:underline; }
  #lightbox { position:fixed; inset:0; background:#000000ee; display:none; align-items:center;
              justify-content:center; padding:24px; z-index:20; cursor:zoom-out; }
  #lightbox img { max-width:100%; max-height:100%; border-radius:8px; }
  .empty { padding:40px; text-align:center; color:#8b949e; }
</style>
</head>
<body>
<header>
  <h1>SmartGym <span class="muted" id="sport"></span></h1>
  <div class="score" id="score">—</div>
  <div class="muted" id="meta"></div>
</header>
<main>
  <div id="table"></div>
</main>
<div id="lightbox"><img id="lightbox-img" alt=""></div>
<script>
const $ = (id) => document.getElementById(id);
function esc(s){ return String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c])); }

async function load() {
  const m = await (await fetch("/api/match")).json();
  $("sport").textContent = (m.meta?.sport || "") + " · " + (m.meta?.players || "");
  $("score").textContent = m.final_score || "—";
  $("meta").textContent = [m.winner ? "winner: " + m.winner : "undecided",
    (m.meta?.video || ""), (m.meta?.processing_fps ? m.meta.processing_fps + " fps processing" : ""),
    (m.calibration?.method ? "calibration: " + m.calibration.method : "")].filter(Boolean).join("  ·  ");

  const rows = (m.points || []).map(p => {
    const ev = (p.events || []).map(e => e.type).filter(Boolean);
    const img = (p.evidence || []).find(f => f.includes("landing")) || (p.evidence || [])[0];
    return `<tr>
      <td>${p.index}</td>
      <td>${p.t_end?.toFixed?.(2) ?? p.t_end}</td>
      <td><b>${esc(p.winner)}</b></td>
      <td><span class="pill">${esc(p.cause)}</span></td>
      <td>${esc(p.score_after?.display || "")}</td>
      <td>${p.confidence?.toFixed?.(2) ?? ""}</td>
      <td>${p.needs_review ? '<span class="pill warn">review</span>' : ""}</td>
      <td class="muted">${esc([...new Set(ev)].join(", "))}</td>
      <td>${img ? `<img class="thumb" src="/evidence/${encodeURIComponent(img.split("/").pop())}" alt="">` : ""}</td>
    </tr>`;
  }).join("");
  $("table").innerHTML = rows
    ? `<table><thead><tr><th>#</th><th>t (s)</th><th>winner</th><th>cause</th><th>score after</th>
       <th>conf</th><th></th><th>events</th><th>proof</th></tr></thead><tbody>${rows}</tbody></table>`
    : '<div class="empty">No points scored yet. Analyse a clip first.</div>';
  document.querySelectorAll(".thumb").forEach(img => img.onclick = () => {
    $("lightbox-img").src = img.src; $("lightbox").style.display = "flex";
  });
}
$("lightbox").onclick = () => { $("lightbox").style.display = "none"; };
load();
</script>
</body>
</html>
"""


class MatchStore:
    """Loads a match JSON (and its evidence) from a run directory."""

    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        self.match_path = os.path.join(self.run_dir, DEFAULT_MATCH_FILE)
        self._lock = threading.Lock()
        self._cache: Optional[Tuple[float, Dict[str, Any]]] = None

    def match(self) -> Dict[str, Any]:
        try:
            mtime = os.path.getmtime(self.match_path)
        except OSError:
            return {"meta": {}, "final_score": "", "points": [], "timeline": [],
                    "note": f"no match.json in {self.run_dir}"}
        with self._lock:
            if self._cache and self._cache[0] == mtime:
                return self._cache[1]
            with open(self.match_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._cache = (mtime, data)
            return data

    def evidence_path(self, name: str) -> Optional[str]:
        safe = posixpath.basename(unquote(name))
        for sub in ("evidence", ""):
            candidate = os.path.join(self.run_dir, sub, safe) if sub else os.path.join(self.run_dir, safe)
            candidate = os.path.abspath(candidate)
            if not candidate.startswith(self.run_dir):
                continue
            if os.path.isfile(candidate):
                return candidate
        return None


class DashboardHandler(BaseHTTPRequestHandler):
    store: MatchStore = None          # injected by make_server
    server_version = "SmartGymDashboard/0.1"

    def log_message(self, fmt: str, *args) -> None:  # keep the console readable
        pass

    # -- helpers ------------------------------------------------------------
    def _send(self, code: int, body: bytes, ctype: str = "application/json",
              headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, code: int = 200) -> None:
        self._send(code, json.dumps(payload, indent=2).encode("utf-8"))

    # -- routes -------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            return self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
        if path == "/healthz":
            return self._json({"ok": True})
        if path == "/api/match":
            return self._json(self.store.match())
        if path == "/api/points":
            m = self.store.match()
            return self._json({"final_score": m.get("final_score"),
                               "winner": m.get("winner"),
                               "points": m.get("points", [])})
        if path.startswith("/api/points/"):
            try:
                idx = int(path.rsplit("/", 1)[1])
            except ValueError:
                return self._json({"error": "bad index"}, 400)
            for p in self.store.match().get("points", []):
                if p.get("index") == idx:
                    return self._json(p)
            return self._json({"error": "point not found"}, 404)
        if path.startswith("/evidence/"):
            name = path[len("/evidence/"):]
            full = self.store.evidence_path(name)
            if not full:
                return self._json({"error": "not found", "requested": name}, 404)
            ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
            with open(full, "rb") as fh:
                return self._send(200, fh.read(), ctype)
        if path == "/export/timeline.csv":
            csv_path = os.path.join(self.store.run_dir, "score_timeline.csv")
            if not os.path.isfile(csv_path):
                return self._json({"error": "no CSV exported"}, 404)
            with open(csv_path, "rb") as fh:
                return self._send(200, fh.read(), "text/csv",
                                  {"Content-Disposition": "attachment; filename=score_timeline.csv"})
        return self._json({"error": "not found", "path": path}, 404)

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()


def make_server(run_dir: str, host: str = "0.0.0.0", port: int = 8080
                ) -> ThreadingHTTPServer:
    handler = type("BoundDashboardHandler", (DashboardHandler,),
                   {"store": MatchStore(run_dir)})
    return ThreadingHTTPServer((host, port), handler)


def serve(run_dir: str, host: str = "0.0.0.0", port: int = 8080) -> None:
    store = MatchStore(run_dir)
    httpd = make_server(run_dir, host, port)
    print(f"SmartGym dashboard: http://{host}:{port}  (run dir: {store.run_dir})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Serve a SmartGym match review dashboard")
    parser.add_argument("run_dir", nargs="?", default="runs/badminton",
                        help="directory containing match.json and evidence/")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)
    serve(args.run_dir, args.host, args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
