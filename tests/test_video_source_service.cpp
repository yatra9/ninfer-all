#include "media/local_video/video_source_service.h"
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>

using namespace ninfer::media::local_video;
static void expect(bool ok,const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc,char** argv) {
    try {
        expect(argc==2,"expected fixture manifest");
        std::ifstream file(argv[1]); nlohmann::json manifest; file>>manifest;
        VideoSourceService service(1);
        for (const auto& fixture:manifest) {
            const std::filesystem::path path=fixture.at("path").get<std::string>();
            auto source=service.acquire(path);
            auto concurrent=std::async(std::launch::async,[&]{return service.acquire(path);});
            expect(concurrent.get()==source,"concurrent source probe was duplicated");
            auto before=source->metadata();
            Options tiny; tiny.scale=0.001; tiny.alignment=32;
            const auto dimensions=output_geometry(before.info,tiny);
            expect(dimensions.width==32 && dimensions.height==32,"small scale must align to minimum geometry");
            expect(source->source_stats().index_builds==0,"geometry must not build an index");
            auto tiny_plan=source->plan(tiny);
            expect(tiny_plan.width==dimensions.width && tiny_plan.height==dimensions.height,"native geometry disagrees");
            expect(before.info.audio_stream_count==fixture.at("audio"),"wrong audio stream count");
            expect(!before.frame_count && !before.variable_frame_rate,"probe invented exact metadata");
            const auto times=fixture.at("times").get<std::vector<double>>();
            auto first=std::async(std::launch::async,[&]{return source->resolve_time(0,0);});
            auto last=source->resolve_time(times.back(),2);
            expect(first.get().nearest.source_index==0,"frame zero was not time zero");
            expect(last.nearest.source_index==std::int64_t(times.size()-1),"wrong final frame");
            for (std::size_t i=0;i<times.size();++i) {
                auto resolved=source->resolve_time(times[i],0);
                expect(resolved.nearest.source_index==std::int64_t(i),"wrong frame mapping");
                expect(std::abs(resolved.nearest.timestamp_seconds-times[i])<1e-9,"wrong timestamp origin");
            }
            auto after=source->metadata();
            expect(after.frame_count==std::int64_t(times.size()),"wrong exact count");
            expect(after.variable_frame_rate==fixture.at("vfr").get<bool>(),"wrong CFR/VFR classification");
            expect(after.interlace_mode==fixture.at("interlace").get<std::string>(),"wrong interlace state");
            bool out=false;
            try { source->resolve_time(times.back()+0.01); }
            catch (const Error& e) { out=e.kind()==ErrorKind::InvalidInput; }
            expect(out,"time after final PTS accepted");
            if (fixture.value("tie",false))
                expect(source->resolve_time(0.125,0).nearest.source_index==0,"tie did not select earlier frame");
            if (path.extension()==".mpeg") {
                for (const auto [midpoint, earlier] :
                     {std::pair{0.02, 0}, std::pair{0.1, 2}}) {
                    expect(source->resolve_time(midpoint,0).nearest.source_index==earlier,
                           "decimal midpoint did not select earlier frame");
                    expect(source->resolve_time(std::nextafter(midpoint,0.0),0).nearest.source_index==earlier,
                           "time immediately before midpoint selected later frame");
                    expect(source->resolve_time(std::nextafter(midpoint,std::numeric_limits<double>::infinity()),0)
                               .nearest.source_index==earlier+1,
                           "time immediately after midpoint selected earlier frame");
                }
            }
            Options options; options.start=1; options.end=2; options.deinterlace=Deinterlace::Off;
            auto reader=source->create_reader(options);
            auto chunk=reader.read_chunk(3);
            expect(chunk.eof && chunk.frames.size()==2,"reader changed selection");
            expect(std::abs(chunk.frames.front().timestamp_seconds-times[1])<1e-9,"reader reset selected-range origin");
            expect(source->source_stats().index_builds==1,"full index scan was repeated");
            expect(source->source_stats().metadata_probes==1,"metadata probe was repeated");
            // Retained LRU slot may be evicted while the reader/source stays active.
            if (manifest.size()>1) {
                const auto other=manifest.back().at("path").get<std::string>();
                (void)service.acquire(other);
                expect(service.acquire(path)==source,"evicted active source was rebuilt");
            }
            auto old=std::filesystem::last_write_time(path);
            std::filesystem::last_write_time(path,old+std::chrono::seconds(2));
            auto changed=service.acquire(path);
            expect(changed!=source,"file replacement did not invalidate source");
            bool stale=false;
            try { source->metadata(); } catch (const Error& e) { stale=e.kind()==ErrorKind::SourceChanged; }
            expect(stale,"active old source accepted a changed file");
            std::filesystem::last_write_time(path,old);
            VideoSource retry(path); Options cancelled;
            cancelled.checkpoint=[]{throw std::runtime_error("cancelled");};
            try { retry.resolve_time(0,0,cancelled); } catch (const std::runtime_error&) {}
            expect(retry.source_stats().index_builds==0,"cancelled index was published");
            retry.resolve_time(0);
            expect(retry.source_stats().index_builds==1,"failed index could not be retried");
        }
        std::cout<<"PASS: source reuse, metadata, CFR/VFR, time origin, reader, invalidation, retry\n";
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
