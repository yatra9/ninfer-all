"""Require wire-level native video evidence, not just a plausible model answer."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--wire", required=True)
parser.add_argument("--events", required=True)
parser.add_argument("--expected-color", default="blue")
parser.add_argument("--report", required=True)
args = parser.parse_args()
wire = [json.loads(line) for line in Path(args.wire).read_text().splitlines() if line.strip()]
events = [json.loads(line) for line in Path(args.events).read_text().splitlines() if line.strip()]
event_times = [event["timestamp"] / 1000 for event in events if "timestamp" in event]
wire = [record for record in wire if min(event_times) - 10 <= record["time"] <= max(event_times) + 10] if event_times else []
session_ids = {event["sessionID"] for event in events if "sessionID" in event}
tool_ids = {event["part"]["callID"] for event in events if event.get("type") == "tool_use"}
answers = [event["part"]["text"] for event in events if event.get("type") == "text"]
links = []
for index, record in enumerate(wire):
    params = record.get("rpc_params") or {}
    if params.get("name") != "inspect_video":
        continue
    for block in record.get("rpc_response", {}).get("result", {}).get("content", []):
        if block.get("type") == "resource_link":
            links.append((index, block["uri"]))
matched = []
for index, uri in links:
    for record in wire[index + 1:]:
        calls = {m.get("tool_call_id") for m in record.get("messages", [])}
        if record["path"] == "/v1/chat/completions" and calls & tool_ids and any(
                video.get("url") == uri for video in record.get("video_urls", [])):
            matched.append(uri)
            break
report = {"single_agent_session": len(session_ids) == 1, "session_ids": sorted(session_ids),
          "inspect_resource_links": len(links), "native_video_urls_matched": matched,
          "expected_color_in_answer": any(args.expected_color in text.lower() for text in answers)}
report["complete"] = report["single_agent_session"] and bool(matched) and report["expected_color_in_answer"]
Path(args.report).write_text(json.dumps(report, indent=2))
print(json.dumps(report))
raise SystemExit(0 if report["complete"] else 1)
