"""CPU-only real HTTP contracts against the production MCP route module."""
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


def main():
    with tempfile.TemporaryDirectory(prefix="ninfer-mcp-") as directory:
        root = pathlib.Path(directory)
        path = root / "日本語 %?#&.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                        "color=red:size=64x48:rate=4:duration=2", "-c:v", "libx264",
                        "-bf", "2", "-y", str(path)], check=True)
        mapped_dir = root / "mapped"
        mapped_dir.mkdir()
        (mapped_dir / path.name).write_bytes(path.read_bytes())
        (root / "escape.mp4").symlink_to("/etc/passwd")
        process = subprocess.Popen([sys.argv[1], directory,
            "C:\\Host Videos", directory, "/host/videos", directory,
            "c:/host videos/special", str(mapped_dir), "\\\\Server\\Share", directory], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        try:
            port = int(process.stdout.readline())
            session = None
            version = "2025-11-25"

            def http(body=None, method="POST", extra=None):
                headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
                if session:
                    headers.update({"Mcp-Session-Id": session, "MCP-Protocol-Version": version})
                headers.update(extra or {})
                request = urllib.request.Request(f"http://127.0.0.1:{port}/mcp",
                    data=None if body is None else body.encode(), headers=headers, method=method)
                try:
                    response = urllib.request.urlopen(request, timeout=10)
                except urllib.error.HTTPError as error:
                    response = error
                data = response.read().decode()
                return response.status, response.headers, json.loads(data) if data else None

            def rpc(method, params=None, ident=1):
                request = {"jsonrpc": "2.0", "method": method}
                if ident is not None:
                    request["id"] = ident
                if params is not None:
                    request["params"] = params
                return http(json.dumps(request))

            status, headers, body = rpc("initialize", {"protocolVersion": version,
                "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}})
            assert status == 200 and body["result"]["protocolVersion"] == version
            session = headers["Mcp-Session-Id"]
            assert rpc("tools/list")[2]["error"]["code"] == -32600
            assert rpc("notifications/initialized", ident=None)[0] == 202
            expected = json.loads(pathlib.Path(sys.argv[2]).read_text())["tools"]
            assert rpc("tools/list")[2]["result"]["tools"] == expected
            assert http(method="GET")[0] == 405
            assert http("{")[2]["error"]["code"] == -32700
            assert http("[]")[2]["error"]["code"] == -32600
            assert rpc("unknown")[2]["error"]["code"] == -32601
            assert rpc("tools/call", {"name": "unknown"})[2]["error"]["code"] == -32602
            assert http(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1}),
                        extra={"MCP-Protocol-Version": "invalid"})[0] == 400
            assert http(method="GET", extra={"Origin": "https://evil.example"})[0] == 403
            assert http(method="GET", extra={"Host": "evil.example"})[0] == 421
            assert http(method="GET", extra={"Origin": f"http://127.0.0.1:{port}"})[0] == 405
            assert http(method="GET", extra={"Origin": "null"})[0] == 403
            assert http(method="GET", extra={"Origin": ""})[0] == 403
            assert http(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1}),
                        extra={"Accept": "application/json"})[0] == 406
            assert http(json.dumps({"jsonrpc": "2.0", "method": "ping", "id": 1}),
                        extra={"Content-Type": "text/plain"})[0] == 415
            assert rpc("ping", {"_meta": False})[2]["error"]["code"] == -32600

            def call(name, arguments):
                status, _, body = rpc("tools/call", {"name": name, "arguments": arguments})
                assert status == 200
                return body["result"]

            metadata = call("get_video_metadata", {"path": str(path)})
            assert not metadata["isError"]
            assert metadata["structuredContent"]["width"] == 64
            assert metadata["structuredContent"]["frame_count"] is None
            resolved = call("resolve_video_time", {"path": str(path), "time_seconds": 0.125})
            assert not resolved["isError"]
            assert resolved["structuredContent"]["nearest_frame_number"] == 0
            decimal_path = root / "decimal-midpoints.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                            "color=red:size=64x48:rate=25:duration=1", "-c:v", "libx264",
                            "-y", str(decimal_path)], check=True)
            for midpoint, earlier in ((0.02, 0), (0.1, 2)):
                for target, expected_frame in ((midpoint, earlier),
                        (math.nextafter(midpoint, 0.0), earlier),
                        (math.nextafter(midpoint, math.inf), earlier + 1)):
                    response = call("resolve_video_time", {"path": str(decimal_path),
                        "time_seconds": target, "radius_frames": 0})
                    assert not response["isError"]
                    assert response["structuredContent"]["nearest_frame_number"] == expected_frame
            assert call("get_video_metadata", {"path": str(path)})["structuredContent"]["frame_count"] == 8
            for reference in ("c:/HOST VIDEOS/" + path.name,
                              "C://Host Videos/./" + path.name,
                              "/host/videos/" + path.name,
                              "\\\\server\\SHARE\\" + path.name):
                mapped_metadata = call("get_video_metadata", {"path": reference})
                assert mapped_metadata["structuredContent"]["path"] == str(path)
                assert not call("resolve_video_time", {"path": reference, "time_seconds": 0})["isError"]
                mapped_link = call("inspect_video", {"path": reference, "frame": 0, "instruction": "color"})
                assert not mapped_link["isError"]
                assert mapped_link["content"][1]["uri"].startswith("ninfer-video:///tmp/")
                assert urllib.parse.unquote(mapped_link["content"][1]["uri"].removeprefix(
                    "ninfer-video://").split("?", 1)[0]) == str(path)
                assert "frame=0" in mapped_link["content"][1]["uri"]
            specific = call("get_video_metadata", {"path": "C:\\Host Videos\\Special\\" + path.name})
            assert specific["structuredContent"]["path"] == str(mapped_dir / path.name)
            for reference in ("D:\\Host Videos\\demo.mp4", "C:\\Host Videos-other\\demo.mp4"):
                assert call("get_video_metadata", {"path": reference})["isError"]
            escaped = call("get_video_metadata", {"path": "C:\\Host Videos\\escape.mp4"})
            assert escaped["isError"] and "path_outside_root" in escaped["content"][0]["text"]
            assert call("get_video_metadata", {"path": "C:\\Host Videos\\..\\outside.mp4"})["isError"]
            link = call("inspect_video", {"path": str(path), "instruction": "Identify color",
                "start_frame": 1, "end_frame": 4, "skip_frame": 1,
                "bbox": {"x": 0, "y": 0, "width": 32, "height": 24}, "scale": 0.5})
            assert not link["isError"]
            assert link["content"][1]["type"] == "resource_link"
            assert "%3F%23%26.mp4?" in link["content"][1]["uri"]
            assert link["content"][1]["mimeType"] == "video/mp4"
            for frame in (0, 7):
                single = call("inspect_video", {"path": str(path), "instruction": "See image",
                    "frame": frame, "scale": 2, "deinterlace": "off",
                    "bbox": {"x": 0, "y": 0, "width": 32, "height": 24}})
                assert not single["isError"]
                assert f"?frame={frame}&" in single["content"][1]["uri"]
                assert single["content"][1]["mimeType"] == "video/mp4"
            for range_key in ("start_frame", "end_frame", "skip_frame"):
                assert call("inspect_video", {"path": str(path), "instruction": "x",
                    "frame": 0, range_key: 0})["isError"]
            for invalid_frame in (-1, 1.5, True, 8, 2**64 - 1):
                assert call("inspect_video", {"path": str(path), "instruction": "x",
                    "frame": invalid_frame})["isError"]
            assert not call("inspect_video", {"path": str(path), "instruction": "small", "scale": 0.001})["isError"]
            # Raw pixels fit 64 Mi pixels, but rounding to 32 pushes them over it.
            assert call("inspect_video", {"path": str(path), "instruction": "aligned", "scale": 147.78})["isError"]
            mkv = root / "native.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-c", "copy", "-y", str(mkv)], check=True)
            assert call("inspect_video", {"path": str(mkv), "instruction": "color"})["content"][1]["mimeType"] == "video/mp4"
            for name, args in [
                ("inspect_video", {"path": "ninfer-video:///x", "instruction": "x"}),
                ("inspect_video", {"path": str(path), "instruction": "x", "start_frame": 8}),
                ("inspect_video", {"path": str(path), "instruction": "x", "scale": 1e300}),
                ("inspect_video", {"path": str(path), "instruction": "x", "start_frame": 3, "end_frame": 2}),
                ("inspect_video", {"path": str(path), "instruction": "x", "bbox": {"x": 63, "y": 0, "width": 2, "height": 1}}),
                ("get_video_metadata", {"path": "relative.mp4"}),
                ("get_video_metadata", {"path": "/etc/passwd"}),
                ("get_video_metadata", {"path": str(path), "unknown": 0}),
                ("get_video_metadata", {"path": str(path) + "?scale=2"}),
                ("resolve_video_time", {"path": str(path), "time_seconds": 2}),
                ("resolve_video_time", {"path": str(path), "time_seconds": 0, "radius_frames": True}),
                ("resolve_video_time", {"path": str(path), "time_seconds": 0, "radius_frames": 17}),
                ("inspect_video", {"path": str(path), "instruction": "x", "skip_frame": 2**64 - 1}),
            ]:
                assert call(name, args)["isError"], (name, args)
            # Long CPU fixture makes cancellation observable while the index is active.
            long_video = root / "cancel.mkv"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "color=blue:size=16x16:rate=1000", "-frames:v", "100000", "-c:v", "ffv1",
                "-y", str(long_video)], check=True)
            pending = []
            worker = threading.Thread(target=lambda: pending.append(rpc("tools/call",
                {"name": "resolve_video_time", "arguments": {"path": str(long_video), "time_seconds": 1}}, 42)))
            worker.start()
            time.sleep(0.03)
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as health:
                assert health.status == 200 and health.read() == b"ok"
            assert rpc("notifications/cancelled", {"requestId": 42}, ident=None)[0] == 202
            worker.join(timeout=10)
            assert pending and pending[0][2]["result"]["isError"]
            assert "cancelled" in pending[0][2]["result"]["content"][0]["text"]
            # Failed/cancelled indexes are not published; retry succeeds.
            assert not call("resolve_video_time", {"path": str(long_video), "time_seconds": 1})["isError"]
            for second in (10, 20):
                resolved = call("resolve_video_time", {"path": str(long_video), "time_seconds": second})
                assert resolved["structuredContent"]["nearest_frame_number"] == second * 1000
            assert http(method="DELETE")[0] == 204
            assert rpc("ping")[0] == 404
            print("video MCP HTTP contracts passed")
        finally:
            process.communicate("stop\n", timeout=10)
            assert process.returncode == 0
        process = subprocess.Popen([sys.argv[1], ""], stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, text=True)
        try:
            port = int(process.stdout.readline())
            session = None
            status, headers, body = rpc("initialize", {"protocolVersion": "2025-06-18",
                "capabilities": {}, "clientInfo": {"name": "disabled-test", "version": "1"}})
            assert status == 200 and body["result"]["protocolVersion"] == "2025-06-18"
            session = headers["Mcp-Session-Id"]
            version = "2025-06-18"
            assert rpc("notifications/initialized", ident=None)[0] == 202
            assert len(rpc("tools/list")[2]["result"]["tools"]) == 3
            disabled = call("get_video_metadata", {"path": str(path)})
            assert disabled["isError"] and "local_media_disabled" in disabled["content"][0]["text"]
        finally:
            process.communicate("stop\n", timeout=10)
            assert process.returncode == 0


if __name__ == "__main__":
    main()
