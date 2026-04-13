#!/usr/bin/env python3
"""
Local HTTP server: serves index.html (with PREDICT_URL injected) and POST /predict
using output/7_rf_model.joblib. Run from the project root:

  python3 predict_server.py

Then open http://127.0.0.1:8765/ in your browser. The UI will call /predict on the
same origin — no edits to index.html are required on disk.

Expected joblib payload (dict):
  - scaler: sklearn StandardScaler (fitted on training features)
  - model: sklearn RandomForestClassifier
  - feature_columns: list of str, same order as X_final.csv

Create the file once, e.g. after training:
  import joblib
  joblib.dump({"scaler": scaler, "model": rf, "feature_columns": [...]}, "output/7_rf_model.joblib")
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "7_rf_model.joblib")
INDEX_PATH = os.path.join(ROOT, "index.html")
PREDICT_TOKEN = "__PREDICT_URL__"

SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}

# Lazy-loaded after first request
_artifacts: dict[str, Any] | None = None


def load_artifacts() -> dict[str, Any]:
    global _artifacts
    if _artifacts is not None:
        return _artifacts
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Save your trained scaler + RandomForest + feature_columns there."
        )
    data = joblib.load(MODEL_PATH)
    if not isinstance(data, dict) or "model" not in data or "scaler" not in data:
        raise ValueError(
            "7_rf_model.joblib must be a dict with keys: 'scaler', 'model', and 'feature_columns'."
        )
    _artifacts = data
    return _artifacts


def predict_row(payload: dict[str, Any]) -> dict[str, Any]:
    data = load_artifacts()
    scaler = data["scaler"]
    model = data["model"]
    cols = data.get("feature_columns")
    if not cols:
        raise ValueError("Artifacts missing 'feature_columns'.")

    row: dict[str, Any] = {}
    missing = [c for c in cols if c not in payload]
    if missing:
        return {"_error": True, "status": 400, "body": {"error": "Missing fields", "fields": missing}}

    for c in cols:
        row[c] = payload[c]

    vec = np.array([[float(row[c]) for c in cols]], dtype=float)
    X_sc = scaler.transform(vec)
    pred = int(model.predict(X_sc)[0])
    proba = model.predict_proba(X_sc)[0]
    classes = list(model.classes_)
    probs = {SEVERITY_LABELS[int(cl)]: float(p) for cl, p in zip(classes, proba)}
    return {
        "_error": False,
        "body": {
            "prediction": pred,
            "label": SEVERITY_LABELS.get(pred, str(pred)),
            "probabilities": probs,
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "AccidentPredict/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def _send_json(self, code: int, obj: Any) -> None:
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def _send_html(self, code: int, html: str) -> None:
        b = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path or "/"
        if path not in ("/", "/index.html"):
            self.send_error(404)
            return
        if not os.path.isfile(INDEX_PATH):
            self._send_html(
                500,
                "<!DOCTYPE html><html><body><p>Missing index.html next to predict_server.py</p></body></html>",
            )
            return
        with open(INDEX_PATH, encoding="utf-8") as f:
            raw = f.read()
        host = self.headers.get("Host", "127.0.0.1:8765")
        scheme = "https" if self.headers.get("X-Forwarded-Proto") == "https" else "http"
        predict_url = f"{scheme}://{host}/predict"
        html = raw.replace(PREDICT_TOKEN, predict_url)
        self._send_html(200, html)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/predict":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
            return
        try:
            result = predict_row(payload)
        except FileNotFoundError as e:
            self._send_json(500, {"error": str(e)})
            return
        except Exception as e:
            warnings.warn(f"predict_row: {e}", stacklevel=1)
            self._send_json(500, {"error": str(e)})
            return

        if result.get("_error"):
            self._send_json(int(result.get("status", 400)), result["body"])
        else:
            self._send_json(200, result["body"])


class ReuseThreadingHTTPServer(ThreadingHTTPServer):
    """Allow quick restart on the same port (avoids TIME_WAIT bind failures)."""

    allow_reuse_address = True


def main() -> None:
    p = argparse.ArgumentParser(description="Serve index.html and POST /predict (7_rf_model.joblib).")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = p.parse_args()

    try:
        load_artifacts()
    except Exception as e:
        print(f"Cannot load model: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        httpd = ReuseThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        if e.errno == errno.EADDRINUSE:
            print(
                f"Port {args.port} is already in use. Stop the other process "
                f"(e.g. another predict_server) or use a different port:\n"
                f"  python3 predict_server.py --port 8766\n"
                f"Find what is listening:  lsof -i :{args.port}",
                file=sys.stderr,
            )
        else:
            print(f"Could not bind to {args.host}:{args.port}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Serving http://{args.host}:{args.port}/  (model: {MODEL_PATH})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
