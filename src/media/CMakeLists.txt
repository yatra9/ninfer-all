# The prebuilt Windows vcpkg FFmpeg tree ships without the png decoder, so a native PNG path
# (png_decode.cpp) decodes PNG directly through zlib. It is on by default on Windows; elsewhere
# FFmpeg decodes PNG, and NINFER_MEDIA_NATIVE_PNG=ON builds the native path to test it.
if(WIN32)
  set(_ninfer_native_png_default ON)
else()
  set(_ninfer_native_png_default OFF)
endif()
option(NINFER_MEDIA_NATIVE_PNG "Decode PNG images natively instead of through FFmpeg"
       ${_ninfer_native_png_default})
set(_ninfer_media_sources decode/decode.cpp)
if(NINFER_MEDIA_NATIVE_PNG)
  list(APPEND _ninfer_media_sources decode/png_decode.cpp)
endif()
add_library(ninfer_media_decode STATIC
  ${_ninfer_media_sources}
)
ninfer_internal_includes(ninfer_media_decode)
target_link_libraries(ninfer_media_decode PRIVATE ${NINFER_FFMPEG_TARGET})

# Local-video ingestion is currently supported by the Linux/WSLC product build. Keep its
# libavfilter dependency out of the native Windows build until that platform is in scope.
if(NOT WIN32)
  add_library(ninfer_local_video STATIC
    local_video/video_pipeline.cpp
    local_video/video_source_service.cpp
  )
  ninfer_internal_includes(ninfer_local_video)
  target_link_libraries(ninfer_local_video PRIVATE ${NINFER_FFMPEG_TARGET})
endif()
if(NINFER_MEDIA_NATIVE_PNG)
  if(DEFINED FFMPEG_VCPKG_TREE)
    target_link_libraries(ninfer_media_decode PRIVATE "${FFMPEG_VCPKG_TREE}/lib/z.lib")
  else()
    find_package(ZLIB REQUIRED)
    target_link_libraries(ninfer_media_decode PRIVATE ZLIB::ZLIB)
  endif()
  target_compile_definitions(ninfer_media_decode PUBLIC NINFER_MEDIA_NATIVE_PNG)
endif()
