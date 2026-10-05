# Repository-pinned subset: HTTP transport, protocol negotiation and its dependencies.
# Use NInfer's exact HTTP/JSON headers; never fetch a second private copy.
add_library(ninfer_mcp_transport STATIC
  src/server/streamable_http_server.cpp
  src/server/host_origin_guard.cpp
)
target_include_directories(ninfer_mcp_transport SYSTEM PUBLIC include)
target_link_libraries(ninfer_mcp_transport PUBLIC ninfer::json ninfer::httplib Threads::Threads)
