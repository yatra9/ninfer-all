#include "media/local_video/video_pipeline.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <deque>
#include <condition_variable>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

namespace ninfer::media::local_video {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
class FfmpegError final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
void check(int rc, const char* stage) {
    if (rc >= 0) return;
    char text[AV_ERROR_MAX_STRING_SIZE]; av_strerror(rc, text, sizeof(text));
    throw FfmpegError(std::string(stage) + ": " + text);
}
void require(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
void require_resource(bool ok, const char* text) {
    if (!ok) throw Error(ErrorKind::ResourceLimit, text);
}
void require_unchanged(bool ok, const char* text) {
    if (!ok) throw Error(ErrorKind::SourceChanged, text);
}
bool file_identity_matches(const std::filesystem::path& path, std::uintmax_t size,
                           std::filesystem::file_time_type modified) {
    std::error_code error;
    const auto current_size = std::filesystem::file_size(path, error);
    if (error || current_size != size) return false;
    const auto current_modified = std::filesystem::last_write_time(path, error);
    return !error && current_modified == modified;
}
void checkpoint(const Options& o) { if (o.checkpoint) o.checkpoint(); }
struct FrameDelete { void operator()(AVFrame* p) const { av_frame_free(&p); } };
using AvFrame = std::unique_ptr<AVFrame, FrameDelete>;
AvFrame frame() { AvFrame f(av_frame_alloc()); if (!f) throw std::bad_alloc(); return f; }
struct Input {
    AVFormatContext* fmt = nullptr;
    AVCodecContext* dec = nullptr;
    AVPacket* packet = nullptr;
    int stream = -1;
    bool drained = false;
    const Options* options;
    std::exception_ptr interrupted;
    static int interrupt(void* p) {
        auto* self = static_cast<Input*>(p);
        try {
            if(self->options) checkpoint(*self->options);
            return 0;
        } catch (...) {
            self->interrupted = std::current_exception();
            return 1;
        }
    }
    void check_io(int rc, const char* stage) {
        if (rc < 0 && interrupted) {
            std::rethrow_exception(std::exchange(interrupted, {}));
        }
        check(rc, stage);
    }
    explicit Input(const std::filesystem::path& path, const Options* o = nullptr): options(o) {
        require(std::filesystem::is_regular_file(path), "input must be a local regular file");
        try {
            fmt = avformat_alloc_context(); if (!fmt) throw std::bad_alloc();
            fmt->interrupt_callback = {interrupt, this};
            AVDictionary* opts = nullptr;
            av_dict_set(&opts, "protocol_whitelist", "file", 0);
            int rc = avformat_open_input(&fmt, path.c_str(), nullptr, &opts);
            av_dict_free(&opts); check_io(rc, "open video");
            check_io(avformat_find_stream_info(fmt, nullptr), "probe video");
            stream = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
            check(stream, "find video stream");
            const AVCodec* codec = avcodec_find_decoder(fmt->streams[stream]->codecpar->codec_id);
            require(codec != nullptr, "video decoder not available");
            dec = avcodec_alloc_context3(codec); if (!dec) throw std::bad_alloc();
            check(avcodec_parameters_to_context(dec, fmt->streams[stream]->codecpar), "configure decoder");
            dec->thread_count = 2;
            dec->err_recognition = AV_EF_CAREFUL | AV_EF_EXPLODE;
            if (o) {
                require(dec->width > 0 && dec->height > 0,
                        "video decoder reports invalid source dimensions");
                require_resource(std::int64_t(dec->width) * dec->height <= o->max_pixels,
                                 "source dimensions exceed max_pixels");
                dec->max_pixels = o->max_pixels;
            }
            check(avcodec_open2(dec, codec, nullptr), "open decoder");
            packet = av_packet_alloc(); if (!packet) throw std::bad_alloc();
        } catch (...) { close(); throw; }
    }
    ~Input() { close(); }
    void close() { av_packet_free(&packet); avcodec_free_context(&dec); avformat_close_input(&fmt); }
    AVStream* st() const { return fmt->streams[stream]; }
    bool next(AVFrame* f) {
        av_frame_unref(f);
        while (true) {
            if (options) checkpoint(*options);
            int rc = avcodec_receive_frame(dec, f);
            if (rc == 0) return true;
            if (rc == AVERROR_EOF) return false;
            if (rc != AVERROR(EAGAIN)) check(rc, "decode frame");
            require(!drained, "decoder requested input after drain");
            do {
                av_packet_unref(packet);
                rc = av_read_frame(fmt, packet);
                if (rc == AVERROR_EOF) {
                    check(avcodec_send_packet(dec, nullptr), "drain decoder");
                    drained = true; break;
                }
                check_io(rc, "read packet");
            } while (packet->stream_index != stream);
            if (!drained) check(avcodec_send_packet(dec, packet), "send packet");
        }
    }
    bool seek(std::int64_t pts) {
        const int rc = avformat_seek_file(fmt, stream, INT64_MIN, pts, pts, 0);
        if (rc < 0) {
            if (interrupted) { std::rethrow_exception(std::exchange(interrupted, {})); }
            return false;
        }
        avcodec_flush_buffers(dec); av_packet_unref(packet); drained = false; return true;
    }
};
bool interlaced(const AVFrame* f) { return (f->flags & AV_FRAME_FLAG_INTERLACED) != 0; }
Info inspect_input(const Input& in) {
    Info x;
    auto* st = in.st(); auto* c = st->codecpar;
    x.width=c->width; x.height=c->height; x.stream_index=in.stream;
    x.time_base_num=st->time_base.num; x.time_base_den=st->time_base.den;
    auto fps=av_guess_frame_rate(in.fmt, st, nullptr); x.fps_num=fps.num; x.fps_den=fps.den;
    x.duration_seconds=st->duration == AV_NOPTS_VALUE ?
        (in.fmt->duration == AV_NOPTS_VALUE ? 0 : double(in.fmt->duration)/AV_TIME_BASE) : st->duration*av_q2d(st->time_base);
    x.reported_frames=st->nb_frames; x.codec=avcodec_get_name(c->codec_id);
    const char* pix=av_get_pix_fmt_name(static_cast<AVPixelFormat>(c->format)); x.pixel_format=pix?pix:"unknown";
    const char* orders[]={"unknown","progressive","tt","bb","tb","bt"};
    x.field_order=(c->field_order>=0 && c->field_order<=5)?orders[c->field_order]:"unknown";
    x.sample_aspect_num=st->sample_aspect_ratio.num; x.sample_aspect_den=st->sample_aspect_ratio.den;
    size_t size=0;
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    x.has_display_transform=
        av_stream_get_side_data(st, AV_PKT_DATA_DISPLAYMATRIX, &size)!=nullptr;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
    return x;
}
void validate(const Options& o) {
    require(o.start>=0, "start_frame must be >= 0");
    require(!o.end || (*o.end>=o.start && *o.end<INT64_MAX-2), "end_frame must be >= start_frame and < INT64_MAX-2");
    require(o.skip>=0 && o.skip<INT64_MAX, "skip_frame must be in [0, INT64_MAX-1]");
    require(std::isfinite(o.scale) && o.scale>0, "scale must be finite and > 0");
    require(o.alignment>0 && o.alignment<=4096, "alignment must be in [1,4096]");
    require_resource(o.max_pixels>0 && o.max_pixels<=INT_MAX/4 && o.max_scan_frames>0 && o.max_selected_frames>0, "resource limits must be positive and max_pixels <= INT_MAX/4");
    if (o.crop) require(o.crop->x>=0 && o.crop->y>=0 && o.crop->width>0 && o.crop->height>0, "invalid bbox");
}
int output_dimension(int source, const Options& o) {
    const double requested = std::round(source * o.scale);
    const double aligned =
        std::max(double(o.alignment), std::floor(requested / o.alignment + 0.5) * o.alignment);
    require(std::isfinite(aligned) && aligned <= INT_MAX / 4, "scale produces invalid dimension");
    return static_cast<int>(aligned);
}
struct Timing { std::int64_t pts; bool key; };
std::int64_t pts(const AVFrame* f) {
    require(f->best_effort_timestamp!=AV_NOPTS_VALUE, "missing frame timestamp; refusing FPS-derived approximation");
    return f->best_effort_timestamp;
}
void geometry(const AVFrame* f, const Options& o) {
    require_resource(f->width>0 && f->height>0 && std::int64_t(f->width)*f->height<=o.max_pixels, "source dimensions exceed max_pixels");
    require(!(f->flags & AV_FRAME_FLAG_CORRUPT), "corrupt decoded frame");
}
struct Filter {
    AVFilterGraph* graph=nullptr; AVFilterContext* src=nullptr; AVFilterContext* sink=nullptr;
    int width, height, format;
    Filter(const AVFrame* f, Deinterlace mode):width(f->width),height(f->height),format(f->format) {
        try {
            graph=avfilter_graph_alloc(); if(!graph) throw std::bad_alloc(); graph->nb_threads=1;
            std::string args="video_size="+std::to_string(width)+"x"+std::to_string(height)+
                ":pix_fmt="+std::to_string(format)+":time_base=1/1:pixel_aspect=1/1";
            check(avfilter_graph_create_filter(&src,avfilter_get_by_name("buffer"),"in",args.c_str(),nullptr,graph),"filter source");
            AVFilterContext* deint=nullptr;
            check(avfilter_graph_create_filter(&deint,avfilter_get_by_name("bwdif"),"deinterlace",
                mode==Deinterlace::On?"mode=send_frame:parity=auto:deint=all":"mode=send_frame:parity=auto:deint=interlaced",nullptr,graph),"bwdif");
            check(avfilter_graph_create_filter(&sink,avfilter_get_by_name("buffersink"),"out",nullptr,nullptr,graph),"filter sink");
            check(avfilter_link(src,0,deint,0),"filter link"); check(avfilter_link(deint,0,sink,0),"filter link");
            check(avfilter_graph_config(graph,nullptr),"configure bwdif");
        } catch(...) { avfilter_graph_free(&graph); throw; }
    }
    ~Filter(){ avfilter_graph_free(&graph); }
};
struct Scaler {
    SwsContext* rgb=nullptr; SwsContext* resize=nullptr;
    ~Scaler(){sws_freeContext(rgb); sws_freeContext(resize);}
    Frame convert(AVFrame* f, const Options& o) {
        Rect r=o.crop.value_or(Rect{0,0,f->width,f->height});
        require(std::int64_t(r.x)+r.width<=f->width && std::int64_t(r.y)+r.height<=f->height,"bbox exceeds decoded frame bounds");
        Frame out; out.width=output_dimension(r.width,o); out.height=output_dimension(r.height,o);
        require_resource(std::int64_t(out.width)*out.height<=o.max_pixels,"scaled dimensions exceed max_pixels");
        // Exact pixel-coordinate ROI for odd YUV420 offsets too. Low-copy YUV ROI is a later optimization.
        rgb=sws_getCachedContext(rgb,f->width,f->height,static_cast<AVPixelFormat>(f->format),f->width,f->height,AV_PIX_FMT_RGB24,SWS_BICUBIC,nullptr,nullptr,nullptr);
        require(rgb!=nullptr,"RGB converter allocation failed");
        int cs=f->colorspace==AVCOL_SPC_UNSPECIFIED?SWS_CS_DEFAULT:int(f->colorspace);
        const int* coeff=sws_getCoefficients(cs);
        check(sws_setColorspaceDetails(rgb,coeff,f->color_range==AVCOL_RANGE_JPEG,coeff,1,0,1<<16,1<<16),"set color conversion");
        std::vector<std::uint8_t> full(std::size_t(f->width)*f->height*3);
        std::uint8_t* dst[]={full.data(),nullptr,nullptr,nullptr}; int strides[]={f->width*3,0,0,0};
        require(sws_scale(rgb,f->data,f->linesize,0,f->height,dst,strides)==f->height,"RGB conversion incomplete");
        out.rgb.resize(std::size_t(out.width)*out.height*3);
        const std::uint8_t* roi=full.data()+(std::size_t(r.y)*f->width+r.x)*3;
        if(out.width==r.width && out.height==r.height) {
            for(int y=0;y<r.height;++y) std::copy_n(roi+std::size_t(y)*strides[0],r.width*3,out.rgb.data()+std::size_t(y)*r.width*3);
        } else {
            resize=sws_getCachedContext(resize,r.width,r.height,AV_PIX_FMT_RGB24,out.width,out.height,AV_PIX_FMT_RGB24,SWS_BICUBIC,nullptr,nullptr,nullptr);
            require(resize!=nullptr,"resize allocation failed");
            const std::uint8_t* data[]={roi,nullptr,nullptr,nullptr};
            dst[0]=out.rgb.data(); int output_strides[]={out.width*3,0,0,0};
            require(sws_scale(resize,data,strides,0,r.height,dst,output_strides)==out.height,"resize incomplete");
        }
        return out;
    }
};
} // namespace

Info inspect(const std::filesystem::path& path) { Input in(path); return inspect_input(in); }

static Stats process_impl(const std::filesystem::path& path,const Options& o,
                          const std::function<void(Frame&&)>& consume,
                          const std::vector<Timing>* reusable_index=nullptr,
                          const Stats* reusable_index_stats=nullptr,
                          const std::function<void(const Stats&)>& report_progress={}) {
    validate(o); require(bool(consume),"frame callback is required");
    const auto original_size=std::filesystem::file_size(path);
    const auto original_time=std::filesystem::last_write_time(path);
    auto unchanged=[&]{require_unchanged(file_identity_matches(path,original_size,original_time),"input changed during processing");};
    Stats stats; std::vector<Timing> owned_index;
    const std::vector<Timing>* index_ptr=reusable_index;
    if(reusable_index_stats) {
        stats.indexed_frames=reusable_index_stats->indexed_frames;
        stats.index_reached_eof=reusable_index_stats->index_reached_eof;
        stats.timing=reusable_index_stats->timing;
        stats.index_seconds=reusable_index_stats->index_seconds;
    }
    if(o.mode==ReadMode::IndexedSeek && !index_ptr) {
        auto started=Clock::now(); Input scan(path,&o); auto f=frame();
        std::optional<std::int64_t> delta; bool variable=false;
        while(true) {
            if(o.end && owned_index.size()>std::uint64_t(*o.end)+2) break;
            if(!scan.next(f.get())) {stats.index_reached_eof=true; break;}
            geometry(f.get(),o); require_resource(owned_index.size()<std::uint64_t(o.max_scan_frames),"index exceeds max_scan_frames");
            auto stamp=pts(f.get());
            if(!owned_index.empty()) {
                require(stamp>owned_index.back().pts,"non-increasing PTS cannot be indexed unambiguously");
                auto d=stamp-owned_index.back().pts;
                if(delta && d!=*delta) variable=true;
                delta=d;
            }
            owned_index.push_back({stamp,(f->flags & AV_FRAME_FLAG_KEY)!=0});
        }
        stats.indexed_frames=owned_index.size(); stats.index_seconds=seconds(started);
        stats.timing=variable?"variable_pts_observed":"uniform_pts_in_scanned_range";
        require(std::uint64_t(o.start)<owned_index.size(),"start_frame is beyond EOF");
        unchanged();
        index_ptr=&owned_index;
    }
    static const std::vector<Timing> no_index;
    const auto& index=index_ptr?*index_ptr:no_index;
    if(o.mode==ReadMode::IndexedSeek)
        require(std::uint64_t(o.start)<index.size(),"start_frame is beyond EOF");
    if(report_progress) report_progress(stats);
    auto started=Clock::now(); auto input=std::make_unique<Input>(path,&o); auto f=frame();
    std::int64_t cursor=0; bool first_ready=false;
    if(!index.empty() && o.start>1) {
        // Leave a preceding source frame for bwdif, then choose a preceding keyframe.
        std::size_t key=std::size_t(o.start-2);
        while(key>0 && !index[key].key) --key;
        bool seek_verified=false;
        if(key>0) {
            // AVFrame's KEY flag is not sufficient to prove random-access safety (notably for
            // H.264 open GOPs). Decode through the first requested frame before publishing any
            // output. A reference failure then has a side-effect-free sequential fallback.
            try {
                Input probe(path,&o);
                if(probe.seek(index[key].pts)) {
                    auto probe_frame=frame();
                    while(probe.next(probe_frame.get())) {
                        const auto stamp=pts(probe_frame.get());
                        const auto it=std::lower_bound(
                            index.begin(),index.end(),stamp,
                            [](const Timing& a,auto b){return a.pts<b;});
                        if(it==index.end() || it->pts!=stamp) break;
                        const auto probe_index=std::distance(index.begin(),it);
                        if(probe_index>=o.start) {
                            seek_verified=probe_index==o.start;
                            break;
                        }
                    }
                }
            } catch(const FfmpegError&) {
                seek_verified=false;
            }
        }
        if(seek_verified && input->seek(index[key].pts) && input->next(f.get())) {
            const auto stamp=pts(f.get());
            const auto it=std::lower_bound(
                index.begin(),index.end(),stamp,[](const Timing& a,auto b){return a.pts<b;});
            if(it!=index.end() && it->pts==stamp && std::distance(index.begin(),it)<=o.start-1) {
                cursor=std::distance(index.begin(),it); first_ready=true; stats.seek_used=true;
            }
        }
        if(!first_ready) {
            input=std::make_unique<Input>(path,&o);
            stats.seek_note=key==0
                ? "sequential fallback: no earlier indexed random-access point"
                : "sequential fallback: seek decode could not reproduce the indexed frame order";
        }
    }
    const auto tb=input->st()->time_base;
    if(report_progress) report_progress(stats);
    std::unique_ptr<Filter> filter; Scaler scaler;
    struct Pending {std::int64_t index, pts; bool deinterlaced;};
    std::deque<Pending> pending;
    auto emit=[&](AVFrame* decoded,const Pending& p) {
        if(p.index<o.start || (o.end && p.index>*o.end) || (p.index-o.start)%(o.skip+1)!=0) return;
        require_resource(stats.selected_frames<o.max_selected_frames,"selection exceeds max_selected_frames"); checkpoint(o);
        auto result=scaler.convert(decoded,o); result.source_index=p.index; result.source_pts=p.pts;
        result.timestamp_seconds=p.pts*av_q2d(tb); result.deinterlaced=p.deinterlaced;
        consume(std::move(result)); ++stats.selected_frames;
    };
    auto filtered=frame();
    auto drain=[&]{
        while(true) {
            int rc=av_buffersink_get_frame(filter->sink,filtered.get());
            if(rc==AVERROR(EAGAIN)||rc==AVERROR_EOF) break;
            check(rc,"read bwdif output");
            require(!pending.empty(),"bwdif produced an unexpected frame");
            auto number=av_rescale_q(filtered->pts,av_buffersink_get_time_base(filter->sink),AVRational{1,1});
            require(number==pending.front().index,"bwdif changed source frame correspondence");
            emit(filtered.get(),pending.front()); pending.pop_front(); av_frame_unref(filtered.get());
        }
    };
    int initial_width=0,initial_height=0,initial_format=-1;
    while(first_ready || input->next(f.get())) {
        first_ready=false; checkpoint(o); geometry(f.get(),o);
        require_resource(stats.decoded_frames<o.max_scan_frames,"decode exceeds max_scan_frames");
        if(stats.first_decoded_index<0) stats.first_decoded_index=cursor;
        ++stats.decoded_frames;
        if(report_progress) report_progress(stats);
        if(!initial_width){initial_width=f->width;initial_height=f->height;initial_format=f->format;}
        require(f->width==initial_width && f->height==initial_height && f->format==initial_format,"midstream geometry/pixel-format change is unsupported");
        auto stamp=pts(f.get());
        if(!index.empty()) require(std::uint64_t(cursor)<index.size() && index[std::size_t(cursor)].pts==stamp,"seek decode differs from indexed display order");
        const bool apply=o.deinterlace==Deinterlace::On ||
            (o.deinterlace==Deinterlace::Auto && interlaced(f.get()));
        Pending p{cursor,stamp,apply};
        if(o.deinterlace==Deinterlace::Off) emit(f.get(),p);
        else {
            // In auto mode a progressive frame is authoritative even when the stream-level
            // field_order is interlaced (mixed content is common in MPEG-TS). Avoid constructing
            // bwdif at all for progressive-only inputs; this also permits geometries bwdif cannot
            // accept. Once an interlaced frame activates the filter, later progressive frames pass
            // through its deint=interlaced path unchanged.
            if(!filter && !apply) {
                emit(f.get(),p);
            } else {
                if(!filter) filter=std::make_unique<Filter>(f.get(),o.deinterlace);
                pending.push_back(p); f->pts=cursor; // Internal filter clock only; original PTS stays in Pending.
                check(av_buffersrc_add_frame_flags(filter->src,f.get(),AV_BUFFERSRC_FLAG_KEEP_REF),"feed bwdif"); drain();
            }
        }
        if(o.end && cursor>=*o.end+(o.deinterlace==Deinterlace::Off?0:1)) break;
        ++cursor;
    }
    if(filter){check(av_buffersrc_add_frame_flags(filter->src,nullptr,0),"flush bwdif");drain();require(pending.empty(),"bwdif dropped source frames");}
    require(stats.selected_frames>0,"no frames selected (start_frame may be beyond EOF)");
    unchanged(); stats.processing_seconds=seconds(started); return stats;
}

struct VideoSource::Impl {
    std::filesystem::path path;
    Info metadata;
    std::uintmax_t size;
    std::filesystem::file_time_type modified;
    mutable std::mutex mutex;
    std::condition_variable index_changed;
    bool index_building = false;
    std::shared_ptr<const std::vector<Timing>> index;
    std::int64_t index_max_pixels = 0;
    Stats index_stats;
    SourceStats diagnostics;

    explicit Impl(std::filesystem::path p):path(std::move(p)) {
        Input input(path);
        metadata=inspect_input(input);
        size=std::filesystem::file_size(path);
        modified=std::filesystem::last_write_time(path);
    }
    void require_unchanged() const {
        ninfer::media::local_video::require_unchanged(
            file_identity_matches(path,size,modified),
            "input changed after VideoSource was opened");
    }
    void ensure_index(const Options& o) {
        for (;;) {
            checkpoint(o);
            require_unchanged();
            {
                std::unique_lock lock(mutex);
                if(index) {
                    require_resource(index->size() <= std::uint64_t(o.max_scan_frames),
                                     "index exceeds max_scan_frames");
                    require_resource(index_max_pixels <= o.max_pixels,
                                     "source dimensions exceed max_pixels");
                    ++diagnostics.index_reuses;
                    return;
                }
                if(index_building) {
                    index_changed.wait_for(lock, std::chrono::milliseconds(10));
                    continue;
                }
                index_building = true;
            }
            try {
                auto started=Clock::now();
                auto candidate=std::make_shared<std::vector<Timing>>();
                std::int64_t candidate_max_pixels=0;
                Input scan(path,&o); auto f=frame();
                std::optional<std::int64_t> delta; bool variable=false;
                while(scan.next(f.get())) {
                    geometry(f.get(),o);
                    candidate_max_pixels=std::max(candidate_max_pixels,
                                                  std::int64_t(f->width)*f->height);
                    require_resource(candidate->size()<std::uint64_t(o.max_scan_frames),"index exceeds max_scan_frames");
                    auto stamp=pts(f.get());
                    if(!candidate->empty()) {
                        require(stamp>candidate->back().pts,"non-increasing PTS cannot be indexed unambiguously");
                        auto d=stamp-candidate->back().pts;
                        if(delta && d!=*delta) variable=true;
                        delta=d;
                    }
                    candidate->push_back({stamp,(f->flags & AV_FRAME_FLAG_KEY)!=0});
                }
                require_unchanged();
                Stats completed;
                completed.indexed_frames=static_cast<std::int64_t>(candidate->size());
                completed.index_reached_eof=true;
                completed.timing=variable?"variable_pts_observed":"uniform_pts_in_scanned_range";
                completed.index_seconds=seconds(started);
                {
                    std::lock_guard lock(mutex);
                    index_stats=completed;
                    index_max_pixels=candidate_max_pixels;
                    index=std::move(candidate); // Publish only a complete, unchanged index.
                    ++diagnostics.index_builds;
                    diagnostics.index_scanned_frames+=completed.indexed_frames;
                    index_building=false;
                }
                index_changed.notify_all();
                return;
            } catch (...) {
                {
                    std::lock_guard lock(mutex);
                    index_building=false;
                }
                index_changed.notify_all();
                throw;
            }
        }
    }
};

struct VideoReader::Impl {
    std::shared_ptr<VideoSource::Impl> source;
    Options options;
    std::shared_ptr<const std::vector<Timing>> index;
    Stats index_stats;
    std::mutex mutex;
    std::condition_variable cv;
    std::optional<Frame> ready;
    std::exception_ptr failure;
    Stats final_stats;
    Stats live_stats;
    std::int64_t delivered=0;
    bool finished=false;
    std::atomic<bool> stopping=false;
    std::thread worker;

    Impl(std::shared_ptr<VideoSource::Impl> s,Options o):source(std::move(s)),options(std::move(o)) {
        if(options.mode==ReadMode::IndexedSeek) {
            std::lock_guard lock(source->mutex);
            index=source->index;
            index_stats=source->index_stats;
        }
    }
    void start() {
        worker=std::thread([this] {
            try {
                auto user_checkpoint=options.checkpoint;
                options.checkpoint=[this,user_checkpoint] {
                    if(stopping.load()) throw std::runtime_error("reader stopped");
                    if(user_checkpoint) user_checkpoint();
                };
                auto stats=process_impl(source->path,options,[this](Frame&& value) {
                    std::unique_lock lock(mutex);
                    cv.wait(lock,[this]{return !ready || stopping.load();});
                    if(stopping.load()) throw std::runtime_error("reader stopped");
                    ready.emplace(std::move(value));
                    cv.notify_all();
                },index.get(),index?&index_stats:nullptr,[this](const Stats& stats) {
                    std::lock_guard lock(mutex); live_stats=stats;
                });
                std::lock_guard lock(mutex); final_stats=std::move(stats); finished=true;
            } catch(...) {
                std::lock_guard lock(mutex);
                if(!stopping.load()) failure=std::current_exception();
                finished=true;
            }
            cv.notify_all();
        });
    }
    ~Impl() {
        stopping.store(true); cv.notify_all();
        if(worker.joinable()) worker.join();
    }
    Stats current_stats() const {
        Stats result=finished&&!failure?final_stats:live_stats;
        if(result.indexed_frames==0 && index) result=index_stats;
        result.selected_frames=delivered;
        return result;
    }
};

VideoSource::VideoSource(std::filesystem::path path):impl_(std::make_shared<Impl>(std::move(path))) {}
VideoSource::~VideoSource()=default;
VideoSource::VideoSource(VideoSource&&) noexcept=default;
VideoSource& VideoSource::operator=(VideoSource&&) noexcept=default;
const Info& VideoSource::info() const noexcept { return impl_->metadata; }
SourceStats VideoSource::source_stats() const {
    std::lock_guard lock(impl_->mutex); return impl_->diagnostics;
}
VideoPlan VideoSource::plan(const Options& options) {
    validate(options);
    impl_->require_unchanged();
    impl_->ensure_index(options);
    std::lock_guard lock(impl_->mutex);
    require(!impl_->index->empty(), "video contains no decoded frame");
    require(static_cast<std::uint64_t>(options.start) < impl_->index->size(),
            "start_frame is beyond EOF");
    const Rect crop = options.crop.value_or(Rect{0, 0, impl_->metadata.width,
                                                  impl_->metadata.height});
    require(std::int64_t(crop.x) + crop.width <= impl_->metadata.width &&
                std::int64_t(crop.y) + crop.height <= impl_->metadata.height,
            "bbox exceeds decoded frame bounds");
    VideoPlan result;
    result.width = output_dimension(crop.width, options);
    result.height = output_dimension(crop.height, options);
    require_resource(std::int64_t(result.width) * result.height <= options.max_pixels,
                     "scaled dimensions exceed max_pixels");
    result.index_stats = impl_->index_stats;
    const std::int64_t available_end = static_cast<std::int64_t>(impl_->index->size() - 1);
    const std::int64_t selected_end =
        options.end ? std::min(*options.end, available_end) : available_end;
    const std::int64_t stride = options.skip + 1;
    for (std::int64_t index = options.start; index <= selected_end;) {
        checkpoint(options);
        require_resource(result.selected_frames.size() <
                             static_cast<std::size_t>(options.max_selected_frames),
                         "selection exceeds max_selected_frames");
        const auto stamp = impl_->index->at(static_cast<std::size_t>(index)).pts;
        result.selected_frames.push_back(
            {index, stamp, stamp * (double(impl_->metadata.time_base_num) /
                                    impl_->metadata.time_base_den)});
        if (selected_end - index < stride) { break; }
        index += stride;
    }
    require(!result.selected_frames.empty(), "no frames selected");
    impl_->require_unchanged();
    return result;
}
VideoReader VideoSource::create_reader(Options options) {
    validate(options); impl_->require_unchanged();
    if(options.mode==ReadMode::IndexedSeek) impl_->ensure_index(options);
    auto reader=std::make_unique<VideoReader::Impl>(impl_,std::move(options));
    reader->start();
    return VideoReader(std::move(reader));
}

VideoReader::VideoReader(std::unique_ptr<Impl> impl):impl_(std::move(impl)) {}
VideoReader::~VideoReader()=default;
VideoReader::VideoReader(VideoReader&&) noexcept=default;
VideoReader& VideoReader::operator=(VideoReader&&) noexcept=default;
Chunk VideoReader::read_chunk(std::size_t max_frames) {
    require(max_frames>0,"max_frames must be > 0");
    require(impl_!=nullptr,"reader has been moved from");
    impl_->source->require_unchanged();
    Chunk chunk; chunk.frames.reserve(max_frames);
    std::unique_lock lock(impl_->mutex);
    while(chunk.frames.size()<max_frames) {
        impl_->cv.wait(lock,[this]{return impl_->ready || impl_->finished;});
        if(impl_->failure) std::rethrow_exception(impl_->failure);
        if(impl_->ready) {
            chunk.frames.push_back(std::move(*impl_->ready));
            impl_->ready.reset(); ++impl_->delivered;
            impl_->cv.notify_all();
            continue;
        }
        break;
    }
    // Once a full chunk has been consumed, let the worker establish whether there is a
    // following frame, clean EOF, or an error. This keeps one-frame lookahead bounded and
    // prevents an already-observed failure from being mistaken for EOF on this chunk.
    if(chunk.frames.size()==max_frames && !impl_->ready && !impl_->finished) {
        impl_->cv.wait(lock,[this]{return impl_->ready || impl_->finished;});
    }
    if(impl_->failure) std::rethrow_exception(impl_->failure);
    chunk.eof=impl_->finished && !impl_->ready;
    chunk.cumulative_stats=impl_->current_stats();
    lock.unlock();
    impl_->source->require_unchanged();
    return chunk;
}

Stats process(const std::filesystem::path& path,const Options& o,const std::function<void(Frame&&)>& consume) {
    return process_impl(path,o,consume);
}
} // namespace ninfer::media::local_video
