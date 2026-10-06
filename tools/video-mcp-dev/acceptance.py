"""Real-model HTTP/MCP acceptance. Run inside the WSLC acceptance container."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8080")
    parser.add_argument("--key", default="video-mcp-test")
    parser.add_argument("--report", default="/acceptance/http-report.json")
    args = parser.parse_args()
    report = {"complete": False}
    session = None
    counter = 0

    def http(path, body=None, method=None, authorized=True, extra=None):
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["Authorization"] = "Bearer " + args.key
        headers.update(extra or {})
        request = urllib.request.Request(args.base + path, method=method,
            data=None if body is None else json.dumps(body).encode(), headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=180)
        except urllib.error.HTTPError as error:
            response = error
        payload = response.read().decode()
        return response.status, response.headers, json.loads(payload) if payload else None

    def rpc(method, params=None, ident=True):
        nonlocal counter
        counter += 1
        request = {"jsonrpc": "2.0", "method": method}
        if ident:
            request["id"] = counter
        if params is not None:
            request["params"] = params
        headers = {"Accept": "application/json, text/event-stream"}
        if session:
            headers.update({"Mcp-Session-Id": session, "MCP-Protocol-Version": "2025-11-25"})
        status, response_headers, payload = http("/mcp", request, extra=headers)
        assert status == (200 if ident else 202), (status, payload)
        return response_headers, payload

    def tool(name, arguments):
        _, response = rpc("tools/call", {"name": name, "arguments": arguments})
        result = response["result"]
        assert not result.get("isError"), result
        return result

    try:
        deadline = time.monotonic() + 120
        while http("/health")[0] != 200:
            assert time.monotonic() < deadline, "model readiness timeout"
            time.sleep(0.5)
        init = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "ninfer-video-acceptance", "version": "1"}}}
        assert http("/mcp", init, authorized=False)[0] == 401
        assert http("/mcp", method="GET", authorized=False)[0] == 401
        assert http("/mcp", method="DELETE", authorized=False)[0] == 401
        assert http("/v1/models", authorized=False)[0] == 401
        assert http("/v1/models")[0] == 200
        headers, initialized = rpc("initialize", init["params"])
        session = headers["Mcp-Session-Id"]
        rpc("notifications/initialized", ident=False)
        report["initialize"] = initialized["result"]
        report["tools"] = [t["name"] for t in rpc("tools/list")[1]["result"]["tools"]]
        report["metadata_before"] = tool("get_video_metadata", {"path": "/videos/red.mp4"})
        report["resolve_first"] = tool("resolve_video_time", {"path": "/videos/red.mp4", "time_seconds": 0.5})
        report["resolve_second"] = tool("resolve_video_time", {"path": "/videos/red.mp4", "time_seconds": 1})
        report["inspect"] = tool("inspect_video", {"path": "/videos/red.mp4", "instruction": "Identify the color",
            "start_frame": 0, "end_frame": 1})
        uri = report["inspect"]["content"][1]["uri"]
        status, _, video = http("/v1/chat/completions", {"model": "qwen3.8-27b", "temperature": 0,
            "max_tokens": 64, "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What color fills these frames? Answer with one color word."},
                {"type": "video_url", "video_url": {"url": uri}}]}]})
        assert status == 200, video
        report["native_video_answer"] = video["choices"][0]["message"]["content"]
        assert "red" in report["native_video_answer"].lower(), video
        report["single_frame_images"] = []
        for frame, color in ((0, "red"), (2, "blue"), (3, "blue")):
            single = tool("inspect_video", {"path": "/videos/mystery.mp4", "instruction": "See this image",
                "frame": frame, "bbox": {"x": 0, "y": 0, "width": 96, "height": 64}, "scale": 2})
            single_uri = single["content"][1]["uri"]
            assert f"?frame={frame}&" in single_uri
            status, _, image = http("/v1/chat/completions", {"model": "qwen3.8-27b", "temperature": 0,
                "max_tokens": 64, "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What color fills this image? Answer with one color word."},
                    {"type": "video_url", "video_url": {"url": single_uri}}]}]})
            assert status == 200, image
            answer = image["choices"][0]["message"]["content"]
            assert color in answer.lower(), image
            report["single_frame_images"].append({"frame": frame, "uri": single_uri, "answer": answer})
        for query in ("frame=0&start_frame=0", "frame=0&end_frame=0", "frame=0&skip_frame=0",
                      "frame=-1", "frame=4"):
            status, _, error = http("/v1/chat/completions", {"model": "qwen3.8-27b", "max_tokens": 8,
                "messages": [{"role": "user", "content": [{"type": "video_url",
                    "video_url": {"url": "ninfer-video:///videos/mystery.mp4?" + query}}]}]})
            assert status == 400, (query, status, error)
        report["metadata_after_reader"] = tool("get_video_metadata", {"path": "/videos/red.mp4"})
        base = "/v1/models/qwen3.8-27b"
        status, _, suspended = http(base + "/suspend", {"auto_resume": False})
        assert status == 200 and suspended["state"] == "suspended", suspended
        report["suspended_cpu_metadata"] = tool("get_video_metadata", {"path": "/videos/red.mkv"})
        report["suspended_cpu_resolve"] = tool("resolve_video_time", {"path": "/videos/red.mkv", "time_seconds": 0.5})
        report["suspended_cpu_inspect"] = tool("inspect_video", {"path": "/videos/red.mkv", "instruction": "color"})
        report["suspended_cpu_frame"] = tool("inspect_video", {"path": "/videos/red.mkv", "instruction": "See image", "frame": 0})
        assert http(base + "/residency")[2]["state"] == "suspended"
        status, _, resumed = http(base + "/resume", {})
        assert status == 200 and resumed["state"] == "ready", resumed
        report["auth_and_suspend"] = "passed"
        # Existing protocol endpoints still answer; the real visual request above is the GPU check.
        status, _, response = http("/v1/responses", {"model": "qwen3.8-27b", "input": "Say OK",
            "max_output_tokens": 8, "reasoning": {"effort": "none"}})
        assert status == 200, response
        assert http("/mcp", method="GET")[0] == 405
        headers = {"Mcp-Session-Id": session, "MCP-Protocol-Version": "2025-11-25"}
        assert http("/mcp", method="DELETE", extra=headers)[0] == 204
        report["complete"] = True
    finally:
        Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps({"complete": report["complete"], "answer": report["native_video_answer"]}))


if __name__ == "__main__":
    main()
