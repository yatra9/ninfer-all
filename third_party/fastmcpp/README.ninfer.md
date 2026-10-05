# fastmcpp transport subset

Source: https://github.com/0xeb/fastmcpp
Pinned commit: `29144985f51f41247584efe0c9cd2064de01b6fa` (3.4.7.1).
License: Apache-2.0; upstream LICENSE and NOTICE are retained.

NInfer vendors the Streamable HTTP transport, protocol-version negotiation and
the headers they require. Tool dispatch and application schemas belong to
NInfer. No upstream client, CLI, plugin/provider system or second listener is
built. The target consumes NInfer's pinned cpp-httplib and nlohmann/json headers.

Local changes:

- Expose `register_routes(httplib::Server&)` so the existing NInfer HTTP server
  owns binding, authentication, readiness, threads and shutdown.
- Validate JSON-RPC envelopes, initialization parameters, HTTP media headers
  and the negotiated protocol-version header; distinguish parse/request errors.
- Keep per-session negotiated versions, make the session-limit check atomic
  with insertion, and clear version state on DELETE/shutdown.
- Accepted response messages use HTTP 202. NInfer exposes no server-initiated
  requests, so the application accepts request/notification envelopes only.

Compatibility profile: MCP 2025-11-25 and 2025-06-18; older revisions lack
the resource-link contract. GET returns 405 because this server has no
server-initiated SSE stream; POST returns JSON. Sessions are closed with DELETE.
The parent server enforces Origin and authentication on all MCP methods.

Initialized notification state is bounded by the session map and cleared on DELETE/shutdown. Session IDs use OS entropy directly on Linux.
