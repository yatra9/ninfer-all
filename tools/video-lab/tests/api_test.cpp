#include "media/local_video/video_pipeline.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace ninfer::media::local_video;

static std::vector<Frame> read_all(VideoReader& reader,std::size_t chunk_size) {
    std::vector<Frame> result;
    while(true) {
        auto chunk=reader.read_chunk(chunk_size);
        for(auto& value:chunk.frames) result.push_back(std::move(value));
        if(chunk.eof) break;
    }
    auto after_eof=reader.read_chunk(chunk_size);
    if(!after_eof.eof || !after_eof.frames.empty()) throw std::runtime_error("EOF is not stable");
    return result;
}

int main(int argc,char** argv) {
    try {
        if(argc!=2) throw std::runtime_error("expected video path");
        VideoSource source(argv[1]);
        Options options; options.start=31; options.end=67; options.skip=2;
        auto first=source.create_reader(options);
        auto a=read_all(first,1);
        auto second=source.create_reader(options);
        auto b=read_all(second,7);
        if(a.size()!=b.size()) throw std::runtime_error("chunk sizes changed frame count");
        for(std::size_t i=0;i<a.size();++i) {
            if(a[i].source_index!=b[i].source_index || a[i].source_pts!=b[i].source_pts ||
               a[i].width!=b[i].width || a[i].height!=b[i].height || a[i].rgb!=b[i].rgb)
                throw std::runtime_error("chunk sizes changed frame data");
        }
        auto plan=source.plan(options);
        if(plan.selected_frames.size()!=a.size() || plan.width!=source.info().width ||
           plan.height!=source.info().height)
            throw std::runtime_error("metadata plan changed selection or geometry");
        for(std::size_t i=0;i<a.size();++i) {
            if(plan.selected_frames[i].source_index!=a[i].source_index ||
               plan.selected_frames[i].source_pts!=a[i].source_pts ||
               plan.selected_frames[i].timestamp_seconds!=a[i].timestamp_seconds)
                throw std::runtime_error("metadata plan differs from decoded timing");
        }
        auto stats=source.source_stats();
        if(stats.index_builds!=1 || stats.index_reuses!=2 || stats.index_scanned_frames<=0)
            throw std::runtime_error("index was not built once and reused once");
        bool rejected=false;
        try { second.read_chunk(0); } catch(const std::exception&) { rejected=true; }
        if(!rejected) throw std::runtime_error("zero chunk size was accepted");
        Options progress_options; progress_options.start=31; progress_options.end=40;
        auto progress_reader=source.create_reader(progress_options);
        auto progress=progress_reader.read_chunk(1);
        if(progress.cumulative_stats.decoded_frames<=0 ||
           progress.cumulative_stats.first_decoded_index<0)
            throw std::runtime_error("in-progress decode stats were not published");
        Options limit_options; limit_options.mode=ReadMode::Sequential;
        limit_options.max_selected_frames=1;
        auto limited=source.create_reader(limit_options);
        bool limit_failed=false;
        try { limited.read_chunk(1); }
        catch(const std::exception&) { limit_failed=true; }
        if(!limit_failed) throw std::runtime_error("reader failure was reported as normal EOF");
        bool remained_poisoned=false;
        try { limited.read_chunk(1); }
        catch(const std::exception&) { remained_poisoned=true; }
        if(!remained_poisoned) throw std::runtime_error("reader failure did not poison later reads");
        VideoSource retry_source(argv[1]);
        Options cancelled; int checkpoints=0;
        cancelled.checkpoint=[&] {
            if(++checkpoints==20) throw std::runtime_error("test cancellation");
        };
        bool index_cancelled=false;
        try { retry_source.create_reader(cancelled); }
        catch(const std::exception&) { index_cancelled=true; }
        if(!index_cancelled || retry_source.source_stats().index_builds!=0)
            throw std::runtime_error("cancelled index was published");
        cancelled.checkpoint={}; cancelled.end=9;
        auto retried=retry_source.create_reader(cancelled);
        read_all(retried,3);
        auto reused=retry_source.create_reader(cancelled);
        read_all(reused,2);
        auto retry_stats=retry_source.source_stats();
        if(retry_stats.index_builds!=1 || retry_stats.index_reuses!=1)
            throw std::runtime_error("index retry/reuse counters are incorrect");
        const auto mutation_path=std::filesystem::temp_directory_path()/"video-lab-mutation.mp4";
        std::filesystem::copy_file(argv[1],mutation_path,
                                   std::filesystem::copy_options::overwrite_existing);
        VideoSource mutation_source(mutation_path);
        Options mutation_options; mutation_options.end=9;
        (void)mutation_source.plan(mutation_options);
        auto mutation_reader=mutation_source.create_reader(mutation_options);
        (void)mutation_reader.read_chunk(2);
        const auto modified=std::filesystem::last_write_time(mutation_path);
        std::filesystem::last_write_time(mutation_path,modified+std::chrono::seconds(10));
        bool mutation_rejected=false;
        try { (void)mutation_reader.read_chunk(2); }
        catch(const Error& error) { mutation_rejected=error.kind()==ErrorKind::SourceChanged; }
        std::filesystem::remove(mutation_path);
        if(!mutation_rejected) throw std::runtime_error("file mutation between chunks was accepted");
        Options resource_options; resource_options.max_selected_frames=1;
        auto resource_reader=source.create_reader(resource_options);
        bool resource_classified=false;
        try { (void)resource_reader.read_chunk(2); }
        catch(const Error& error) { resource_classified=error.kind()==ErrorKind::ResourceLimit; }
        if(!resource_classified) throw std::runtime_error("resource limit was not typed");
        std::cout << "PASS: chunks, stats, errors, cancellation, mutation, EOF, index reuse\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
