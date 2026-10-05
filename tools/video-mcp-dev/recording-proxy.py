"""Acceptance-only transparent HTTP observer; never part of the product server."""
import argparse
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
from urllib.parse import urlsplit

parser = argparse.ArgumentParser()
parser.add_argument("--upstream", required=True)
parser.add_argument("--log", default="/tmp/video-mcp-wire.jsonl")
args = parser.parse_args()
upstream = urlsplit(args.upstream)
lock = threading.Lock()


def video_urls(value):
    if isinstance(value, dict):
        if value.get("type") == "video_url":
            yield value["video_url"]
        for child in value.values():
            yield from video_urls(child)
    elif isinstance(value, list):
        for child in value:
            yield from video_urls(child)


class Proxy(BaseHTTPRequestHandler):
    def handle_request(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        record = {"time": time.time(), "path": self.path, "method": self.command}
        if body:
            data = json.loads(body)
            record.update({"rpc_method": data.get("method"), "rpc_id": data.get("id"),
                           "rpc_params": data.get("params")})
            if "messages" in data:
                record["model"] = data.get("model")
                record["video_urls"] = list(video_urls(data))
                record["messages"] = [{"role": m["role"], "tool_call_id": m.get("tool_call_id"),
                    "tool_calls": m.get("tool_calls")} for m in data["messages"]]
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in {"host", "connection", "content-length", "transfer-encoding"}}
        # The test proxy forwards to the same local NInfer server, whose Host guard is retained.
        headers["Host"] = "127.0.0.1:8080"
        connection = http.client.HTTPConnection(upstream.hostname, upstream.port, timeout=180)
        try:
            connection.request(self.command, self.path, body=body or None, headers=headers)
            response = connection.getresponse()
            record["status"] = response.status
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {"connection", "transfer-encoding"}:
                    self.send_header(key, value)
            self.end_headers()
            observed = bytearray()
            while block := response.read(4096):
                if self.path == "/mcp":
                    observed.extend(block)
                self.wfile.write(block)
                self.wfile.flush()
            if observed:
                record["rpc_response"] = json.loads(observed)
        finally:
            connection.close()
            with lock, open(args.log, "a") as output:
                output.write(json.dumps(record) + "\n")

    do_POST = do_GET = do_DELETE = do_OPTIONS = handle_request


ThreadingHTTPServer(("0.0.0.0", 8080), Proxy).serve_forever()
