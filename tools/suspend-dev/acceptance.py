#!/usr/bin/env python3
"""Direct HTTP/GPU acceptance against an already running, explicitly selected model."""
import argparse
import json
import pathlib
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--report", required=True)
    parser.add_argument("--trace")
    args = parser.parse_args()
    report = {"model": args.model, "cycles": [], "complete": False}
    report_path = pathlib.Path(args.report)

    def save():
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def call(path, body=None, expected=200):
        request = urllib.request.Request(args.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                status, data = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, data = error.code, error.read()
        result = json.loads(data)
        if status != expected:
            raise RuntimeError(f"{path}: expected {expected}, received {status}: {result}")
        return result

    def gpu():
        line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                                        "--format=csv,noheader,nounits"], text=True).strip()
        used, free = map(int, line.split(","))
        return {"used_mib": used, "free_mib": free}

    def ram():
        process = {}
        for line in pathlib.Path("/proc/1/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSwap:")):
                name, value = line.split(":", 1)
                process[name + "_kib"] = int(value.split()[0])
        return process

    def graphs():
        if not args.trace:
            return None
        counts = {"capture": 0, "instantiate": 0, "launch": 0, "destroy": 0}
        instantiated, launched = set(), set()
        for line in pathlib.Path(args.trace).read_text().splitlines():
            event, handle, result = line.split()
            if result != "0":
                raise RuntimeError(f"CUDA Graph failure: {line}")
            counts[event] += 1
            if event == "instantiate":
                instantiated.add(handle)
            if event == "launch":
                launched.add(handle)
        return {"counts": counts, "instantiated": sorted(instantiated), "launched": sorted(launched)}

    base = "/v1/models/" + args.model
    try:
        ready_deadline = time.monotonic() + 180
        while True:
            try:
                report["ready"] = call(base + "/residency")
                break
            except RuntimeError as error:
                if "model_loading" not in str(error) or time.monotonic() >= ready_deadline:
                    raise
                time.sleep(1)
        report["gpu_ready"] = gpu()
        report["ram_ready"] = ram()
        # A stored root and store:false siblings exercise the same Engine-local continuation lineage.
        common = {"model": args.model, "temperature": 0, "max_output_tokens": 32,
                  "reasoning": {"effort": "none"}}
        root = call("/v1/responses", dict(common, input="Remember these words: red, green, blue.", store=True))
        child_body = dict(common, input="Repeat those three words, in the same order.",
                          previous_response_id=root["id"], store=False)
        control = call("/v1/responses", child_body)

        def signature(response):
            return [{"type": item["type"], "content": item.get("content"),
                     "summary": item.get("summary")} for item in response["output"]]

        report["control"] = {"output": signature(control), "usage": control["usage"]}
        report["graphs_before"] = graphs()
        save()
        for iteration in range(args.cycles):
            start = time.monotonic()
            suspended = call(base + "/suspend", {})
            gpu_suspended = gpu()
            ram_suspended = ram()
            assert suspended["state"] == "suspended" and suspended["persistent_snapshot_bytes"] > 0
            assert suspended["retained_device_bytes"] == 0
            assert suspended["released_device_bytes"] >= 20 * 1024**3
            assert gpu_suspended["free_mib"] >= 20 * 1024
            call("/v1/chat/completions", {"model": args.model, "messages": [], "stream": True}, 503)
            assert call(base + "/residency")["state"] == "suspended"
            resumed = call(base + "/resume", {})
            assert resumed["state"] == "ready" and resumed["persistent_snapshot_bytes"] == 0
            assert resumed["vmm_map_seconds"] > 0 and resumed["persistent_snapshot_h2d_seconds"] > 0
            result = call("/v1/responses", child_body)
            assert signature(result) == signature(control), "continuation output changed"
            assert result["usage"]["output_tokens"] == control["usage"]["output_tokens"]
            trace = graphs()
            if trace:
                before = report["graphs_before"]
                for event in ("capture", "instantiate", "destroy"):
                    assert trace["counts"][event] == before["counts"][event], f"Graph {event} count changed"
                assert set(trace["launched"]) <= set(before["instantiated"]), "new GraphExec launched"
            item = {"iteration": iteration + 1, "suspended": suspended, "resumed": resumed,
                    "gpu_suspended": gpu_suspended, "gpu_resumed": gpu(),
                    "ram_suspended": ram_suspended, "ram_resumed": ram(),
                    "usage": result["usage"], "graphs": trace, "seconds": time.monotonic() - start}
            report["cycles"].append(item)
            save()
            print(json.dumps({"iteration": iteration + 1, "released_gib": suspended["released_device_bytes"] / 1024**3,
                              "free_mib": gpu_suspended["free_mib"], "seconds": item["seconds"]}), flush=True)
        report["complete"] = True
        save()
    except BaseException as error:
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
