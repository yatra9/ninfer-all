#include "media/local_video/video_pipeline.h"
#include <nlohmann/json.hpp>
#include <png.h>
#include <charconv>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <sys/resource.h>
extern "C" {
#include <libavutil/sha.h>
#include <libavutil/mem.h>
}
using nlohmann::json;
using namespace ninfer::media::local_video;
namespace {
volatile std::sig_atomic_t interrupted=0;
void signal_handler(int){interrupted=1;}
std::int64_t integer(const std::string& s) {
    std::int64_t n; auto [p,e]=std::from_chars(s.data(),s.data()+s.size(),n);
    if(e!=std::errc{} || p!=s.data()+s.size()) throw std::runtime_error("invalid integer: "+s);
    return n;
}
int small_integer(const std::string& s){auto n=integer(s);if(n<INT_MIN||n>INT_MAX)throw std::runtime_error("integer out of range: "+s);return int(n);}
std::string hash(const Frame& f) {
    auto* sha=av_sha_alloc(); if(!sha)throw std::bad_alloc();
    av_sha_init(sha,256); av_sha_update(sha,f.rgb.data(),f.rgb.size());
    std::uint8_t bytes[32];av_sha_final(sha,bytes);av_free(sha);
    std::ostringstream s;for(auto b:bytes)s<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);return s.str();
}
json frame_json(const Frame& f){return {{"source_frame_index",f.source_index},{"source_pts",f.source_pts},{"timestamp_seconds",f.timestamp_seconds},{"width",f.width},{"height",f.height},{"deinterlaced",f.deinterlaced},{"rgb_sha256",hash(f)}};}
json stats_json(const Stats& s){return {{"indexed_frames",s.indexed_frames},{"index_reached_eof",s.index_reached_eof},{"decoded_frames",s.decoded_frames},{"selected_frames",s.selected_frames},{"first_decoded_index",s.first_decoded_index},{"seek_used",s.seek_used},{"seek_note",s.seek_note},{"timing",s.timing},{"index_seconds",s.index_seconds},{"processing_seconds",s.processing_seconds}};}
json info_json(const Info& i){return {{"width",i.width},{"height",i.height},{"stream_index",i.stream_index},{"codec",i.codec},{"pixel_format",i.pixel_format},{"reported_frames",i.reported_frames},{"duration_seconds",i.duration_seconds},{"fps_num",i.fps_num},{"fps_den",i.fps_den},{"time_base_num",i.time_base_num},{"time_base_den",i.time_base_den},{"field_order",i.field_order},{"sample_aspect_num",i.sample_aspect_num},{"sample_aspect_den",i.sample_aspect_den},{"has_display_transform",i.has_display_transform}};}
void png_write(const Frame& f,const std::filesystem::path& p){
    png_image image{};image.version=PNG_IMAGE_VERSION;image.width=f.width;image.height=f.height;image.format=PNG_FORMAT_RGB;
    if(!png_image_write_to_file(&image,p.c_str(),0,f.rgb.data(),0,nullptr)){
        std::string error=image.message;png_image_free(&image);throw std::runtime_error("PNG write: "+error);
    }
    png_image_free(&image);
}
const char* help=R"(video-lab: local video pipeline validation (CPU, FFmpeg libraries)
  video-lab probe INPUT
  video-lab extract INPUT --output NEW_DIRECTORY [options]
  video-lab verify INPUT [options]   # sequential vs indexed-seek pixel SHA-256
Options:
  --start-frame N       default 0, original display-frame order
  --end-frame N         inclusive, default EOF
  --skip-frame N        skip N between selected frames, default 0
  --bbox X,Y,W,H        source pixel coordinates (no implicit rotation)
  --scale NUMBER        positive scale, default 1
  --align N             nearest multiple, ties up; default 1 (Qwen: 32)
  --deinterlace MODE    auto|on|off; bwdif send_frame, default auto
  --mode MODE           indexed|sequential; default indexed
  --max-pixels N        per source/output frame, default 67108864
  --max-frames N        max selected frames, default 10000
  --max-scan-frames N   default 2000000; indexing/decoding limit
  --chunk-frames N      selected frames returned per library read, default 16
  --no-images          extract JSON/hash only (still performs preprocessing)
  --autotone 0         reserved; enabling is an explicit error
Indexed mode builds a PTS/keyframe index first, then seeks. Index cost is reported
separately; it is not a claim of fast first-request access. verify compares exact
output pixels and source timing. PNGs and manifest.json are written by extract.
)";
}
int main(int argc,char** argv){
    std::signal(SIGINT,signal_handler);std::signal(SIGTERM,signal_handler);
    std::optional<std::filesystem::path> output_created;
    try {
        if(argc==2 && std::string(argv[1])=="--help"){std::cout<<help;return 0;}
        if(argc<3)throw std::runtime_error(help);
        std::string command=argv[1]; std::filesystem::path path=argv[2];Options o;
        bool images=true;std::string output;std::size_t chunk_frames=16;std::map<std::string,bool> seen;
        for(int a=3;a<argc;++a){
            std::string key=argv[a];if(seen[key])throw std::runtime_error("duplicate option: "+key);seen[key]=true;
            if(key=="--no-images"){images=false;continue;}
            if(a+1>=argc)throw std::runtime_error("missing value for "+key);std::string v=argv[++a];
            if(key=="--start-frame")o.start=integer(v);
            else if(key=="--end-frame")o.end=integer(v);
            else if(key=="--skip-frame")o.skip=integer(v);
            else if(key=="--scale"){std::size_t p;o.scale=std::stod(v,&p);if(p!=v.size())throw std::runtime_error("invalid scale");}
            else if(key=="--align")o.alignment=small_integer(v);
            else if(key=="--max-pixels")o.max_pixels=integer(v);
            else if(key=="--max-frames")o.max_selected_frames=integer(v);
            else if(key=="--max-scan-frames")o.max_scan_frames=integer(v);
            else if(key=="--chunk-frames"){auto n=integer(v);if(n<=0)throw std::runtime_error("chunk-frames must be > 0");chunk_frames=static_cast<std::size_t>(n);}
            else if(key=="--output")output=v;
            else if(key=="--autotone"){if(v!="0"&&v!="false")throw std::runtime_error("autotone is not supported; use 0 or false");}
            else if(key=="--mode"){
                if(v=="indexed")o.mode=ReadMode::IndexedSeek;else if(v=="sequential")o.mode=ReadMode::Sequential;else throw std::runtime_error("mode must be indexed or sequential");
            } else if(key=="--deinterlace"){
                if(v=="auto")o.deinterlace=Deinterlace::Auto;else if(v=="on")o.deinterlace=Deinterlace::On;else if(v=="off")o.deinterlace=Deinterlace::Off;else throw std::runtime_error("deinterlace must be auto, on or off");
            } else if(key=="--bbox"){
                std::istringstream s(v);std::string part;std::vector<int> values;
                while(std::getline(s,part,','))values.push_back(small_integer(part));
                if(values.size()!=4||v.back()==',')throw std::runtime_error("bbox must be X,Y,W,H");
                o.crop=Rect{values[0],values[1],values[2],values[3]};
            }else throw std::runtime_error("unknown option: "+key);
        }
        o.checkpoint=[]{if(interrupted)throw std::runtime_error("processing cancelled");};
        VideoSource video(path);json report={{"schema_version",1},{"source",info_json(video.info())}};
        if(command=="probe"){
            if(argc!=3)throw std::runtime_error("probe takes only INPUT");
            std::cout<<report.dump(2)<<'\n';return 0;
        }
        if(command!="extract"&&command!="verify")throw std::runtime_error("command must be probe, extract or verify");
        report["options"]={{"start_frame",o.start},{"end_frame",o.end?json(*o.end):json(nullptr)},{"skip_frame",o.skip},{"scale",o.scale},{"alignment",o.alignment},{"chunk_frames",chunk_frames},{"deinterlace",o.deinterlace==Deinterlace::Auto?"auto":o.deinterlace==Deinterlace::On?"on":"off"},{"mode",o.mode==ReadMode::IndexedSeek?"indexed":"sequential"}};
        report["options"]["bbox"]=o.crop?json::array({o.crop->x,o.crop->y,o.crop->width,o.crop->height}):json(nullptr);
        report["frames"]=json::array();
        if(command=="extract"){
            if(output.empty())throw std::runtime_error("extract requires --output NEW_DIRECTORY");
            if(std::filesystem::exists(output))throw std::runtime_error("output directory already exists; choose a new directory");
            std::filesystem::create_directories(output);output_created=output;
            auto reader=video.create_reader(o); Stats stats;
            while(true){auto chunk=reader.read_chunk(chunk_frames);for(auto& f:chunk.frames){
                auto record=frame_json(f);
                if(images){std::string name="frame_"+std::to_string(f.source_index)+".png";png_write(f,std::filesystem::path(output)/name);record["image"]=name;}
                report["frames"].push_back(std::move(record));
            }stats=chunk.cumulative_stats;if(chunk.eof)break;}
            report["stats"]=stats_json(stats);
        }else{
            if(!output.empty())throw std::runtime_error("verify writes JSON to stdout; --output is only for extract");
            o.mode=ReadMode::Sequential;
            auto reference_reader=video.create_reader(o);Stats reference;
            while(true){auto chunk=reference_reader.read_chunk(chunk_frames);for(auto& f:chunk.frames)report["frames"].push_back(frame_json(f));reference=chunk.cumulative_stats;if(chunk.eof)break;}
            std::size_t cursor=0;o.mode=ReadMode::IndexedSeek;
            auto seek_reader=video.create_reader(o);Stats seek;
            while(true){auto chunk=seek_reader.read_chunk(chunk_frames);for(auto& f:chunk.frames){
                auto current=frame_json(f);
                if(cursor>=report["frames"].size()||current!=report["frames"][cursor])throw std::runtime_error("seek/reference mismatch at source frame "+std::to_string(f.source_index));
                ++cursor;
            }seek=chunk.cumulative_stats;if(chunk.eof)break;}
            if(cursor!=report["frames"].size())throw std::runtime_error("seek/reference selected count mismatch");
            report["verified"]=true;report["reference_stats"]=stats_json(reference);report["stats"]=stats_json(seek);
        }
        auto source_stats=video.source_stats();
        report["source_stats"]={{"index_builds",source_stats.index_builds},{"index_scanned_frames",source_stats.index_scanned_frames},{"index_reuses",source_stats.index_reuses}};
        rusage usage{};getrusage(RUSAGE_SELF,&usage);report["peak_rss_kib"]=usage.ru_maxrss;
        report["complete"]=true;
        if(output_created){std::ofstream f(*output_created/"manifest.json");f<<report.dump(2)<<'\n';f.close();if(!f)throw std::runtime_error("manifest write failed");}
        std::cout<<report.dump(2)<<'\n';return 0;
    }catch(const std::exception& e){
        json error={{"complete",false},{"error",e.what()}};
        if(output_created){std::ofstream f(*output_created/"error.json");f<<error.dump(2)<<'\n';}
        std::cerr<<error.dump()<<'\n';return 1;
    }
}
