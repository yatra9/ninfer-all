#include "media/local_video/video_pipeline.h"
#include <atomic>
#include <iostream>
#include <stdexcept>
using namespace ninfer::media::local_video;
void require(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
template<class F> void throws(F f) {
    bool failed=false;
    try { f(); } catch(const std::exception&) { failed=true; }
    require(failed,"expected exception");
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"expected fixture");
        VideoSource source(argv[1]);
        for(int repeat=0;repeat<10;++repeat) {
            for(auto mode:{ReadMode::Sequential,ReadMode::IndexedSeek}) {
                for(std::size_t n:{1,2,3,16}) {
                    Options o; o.mode=mode;o.start=100;o.end=124;o.skip=1;
                    auto reader=source.create_reader(o);
                    std::int64_t count=0,decoded=0;
                    while(true) {
                        auto c=reader.read_chunk(n);
                        for(auto& f:c.frames) require(f.source_index==100+2*count++,"frame order");
                        require(c.cumulative_stats.selected_frames==count,"selected stats");
                        require(c.cumulative_stats.decoded_frames>=decoded && c.cumulative_stats.decoded_frames>0,"decoded stats");
                        decoded=c.cumulative_stats.decoded_frames;
                        require(c.cumulative_stats.first_decoded_index>=0,"first index");
                        require(c.cumulative_stats.seek_used==(mode==ReadMode::IndexedSeek),"seek stats");
                        if(c.eof) break;
                    }
                    require(count==13,"frame count");
                    auto eof=reader.read_chunk(n);
                    require(eof.eof && eof.frames.empty(),"stable EOF");
                    o.start=0;o.end.reset();o.max_selected_frames=1;
                    auto failed=source.create_reader(o);
                    throws([&]{failed.read_chunk(n);});
                    throws([&]{failed.read_chunk(n);});
                }
            }
        }
        std::atomic<bool> cancelled=false;
        Options o;o.mode=ReadMode::Sequential;
        o.checkpoint=[&] { if(cancelled.load()) throw std::runtime_error("cancelled"); };
        auto reader=source.create_reader(o);
        reader.read_chunk(1); cancelled=true;
        throws([&]{reader.read_chunk(32);});
        throws([&]{reader.read_chunk(1);});
        std::cout << "PASS: 80 chunk/EOF/stat runs, 80 persistent-error runs, decode cancellation\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
