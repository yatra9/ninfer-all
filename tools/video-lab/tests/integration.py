"""Observable CPU pipeline tests. ffmpeg/ffprobe are fixture/oracle tools only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

app = sys.argv[1]
api_test = sys.argv[2]

def run(*args):
    return subprocess.run([str(a) for a in args], check=True, capture_output=True).stdout

def cli(*args):
    return json.loads(run(app, *args))

def ffmpeg(*args):
    return run('ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args)

with tempfile.TemporaryDirectory(prefix='video-lab-test-') as d:
    root = Path(d)
    cfr = root / 'cfr.mp4'
    ffmpeg('-f', 'lavfi', '-i', 'testsrc2=size=160x96:rate=30:duration=6',
           '-c:v', 'libx264', '-g', '30', '-bf', '3', '-pix_fmt', 'yuv420p', cfr)
    info = cli('probe', cfr)['source']
    assert (info['width'], info['height'], info['codec']) == (160, 96, 'h264')
    run(api_test, cfr)
    for start, end, skip, expected in [(0, 9, 0, list(range(10))), (0, 9, 1, [0,2,4,6,8]),
                                     (1,10,2,[1,4,7,10]), (100,100,99,[100])]:
        r = cli('verify', cfr, '--start-frame', start, '--end-frame', end, '--skip-frame', skip)
        assert [f['source_frame_index'] for f in r['frames']] == expected
        assert all(abs(f['timestamp_seconds'] - f['source_frame_index']/30) < 1e-8 for f in r['frames'])
    baseline = cli('verify', cfr, '--start-frame', 17, '--end-frame', 66, '--skip-frame', 2, '--chunk-frames', 64)
    for chunk_size in [1, 2, 3, 16]:
        chunked = cli('verify', cfr, '--start-frame', 17, '--end-frame', 66,
                      '--skip-frame', 2, '--chunk-frames', chunk_size)
        assert chunked['frames'] == baseline['frames']
    r = cli('verify', cfr, '--start-frame', 100, '--end-frame', 140, '--skip-frame', 2,
            '--bbox', '3,5,121,71', '--scale', 1.5, '--align', 32)
    assert r['stats']['seek_used'] and r['stats']['first_decoded_index'] > 0
    assert r['stats']['decoded_frames'] < r['reference_stats']['decoded_frames']
    assert {(f['width'], f['height']) for f in r['frames']} == {(192,96)}
    # Independent full RGB conversion + exact Python crop (including odd chroma offsets).
    raw = ffmpeg('-i', cfr, '-vf', "select=eq(n\\,7)", '-frames:v', '1', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-')
    cropped = b''.join(raw[(y*160+3)*3:(y*160+3+121)*3] for y in range(5,76))
    r = cli('extract', cfr, '--start-frame',7,'--end-frame',7,'--bbox','3,5,121,71',
            '--deinterlace','off','--output',root/'crop')
    assert r['frames'][0]['rgb_sha256'] == hashlib.sha256(cropped).hexdigest()
    assert (root/'crop'/'frame_7.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    # Scale geometry, EOF handling, and default auto on progressive material.
    for scale in [0.5,0.75,1,1.25,1.5,2]:
        r = cli('verify', cfr,'--start-frame',100,'--end-frame',103,'--scale',scale)
        assert r['frames'][0]['width'] == round(160*scale)
    auto = cli('verify', cfr,'--start-frame',170,'--end-frame',999)
    off = cli('verify', cfr,'--start-frame',170,'--end-frame',999,'--deinterlace','off')
    assert auto['frames'] == off['frames'] and len(auto['frames']) == 10
    # VFR timestamps must equal ffprobe display-frame PTS, not average-FPS estimates.
    vfr = root/'vfr.mp4'
    ffmpeg('-f','lavfi','-i','testsrc2=size=160x96:rate=30:duration=4',
           '-vf', "select='if(lt(n,60),not(mod(n,2)),not(mod(n,3)))'", '-fps_mode','vfr',
           '-c:v','libx264','-g','10','-bf','3',vfr)
    reference=json.loads(run('ffprobe','-v','error','-select_streams','v:0','-show_frames',
                             '-show_entries','frame=best_effort_timestamp','-of','json',vfr))['frames']
    r=cli('verify',vfr,'--start-frame',25,'--end-frame',45,'--skip-frame',2)
    assert r['stats']['timing']=='variable_pts_observed'
    for f in r['frames']:
        assert f['source_pts']==int(reference[f['source_frame_index']]['best_effort_timestamp'])
    # Interlaced source: same send_frame filter run independently by ffmpeg is the oracle.
    inter=root/'interlaced.mp4'
    ffmpeg('-f','lavfi','-i','testsrc2=size=160x96:rate=60:duration=4',
           '-vf','tinterlace=mode=interleave_top','-c:v','libx264','-flags','+ilme+ildct',
           '-x264-params','tff=1','-g','30',inter)
    for mode in ['auto','on','off']:
        r=cli('verify',inter,'--start-frame',70,'--end-frame',79,'--skip-frame',2,'--deinterlace',mode)
        assert all(f['deinterlaced']==(mode!='off') for f in r['frames'])
    inter_large=cli('verify',inter,'--start-frame',31,'--end-frame',70,
                    '--deinterlace','on','--chunk-frames',64)
    inter_single=cli('verify',inter,'--start-frame',31,'--end-frame',70,
                     '--deinterlace','on','--chunk-frames',1)
    assert inter_single['frames']==inter_large['frames']
    r=cli('extract',inter,'--start-frame',70,'--end-frame',70,'--deinterlace','on','--output',root/'deint')
    raw=ffmpeg('-i',inter,'-vf',"bwdif=mode=send_frame:parity=auto:deint=all,select=eq(n\\,70)",
               '-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','-')
    assert r['frames'][0]['rgb_sha256']==hashlib.sha256(raw).hexdigest()
    # Progressive frames in a stream marked interlaced remain progressive in auto mode.
    inter_segment=root/'inter-segment.ts'
    progressive_segment=root/'progressive-segment.ts'
    ffmpeg('-f','lavfi','-i','testsrc2=size=160x96:rate=25:duration=2',
           '-c:v','mpeg2video','-flags','+ilme+ildct','-top','1','-g','12','-f','mpegts',inter_segment)
    ffmpeg('-f','lavfi','-i','testsrc2=size=160x96:rate=25:duration=2',
           '-c:v','mpeg2video','-g','12','-f','mpegts',progressive_segment)
    mixed=root/'mixed.ts'
    mixed.write_bytes(inter_segment.read_bytes()+progressive_segment.read_bytes())
    mixed_auto=cli('extract',mixed,'--mode','sequential','--deinterlace','auto',
                   '--start-frame',60,'--end-frame',60,'--output',root/'mixed-auto')
    mixed_off=cli('extract',mixed,'--mode','sequential','--deinterlace','off',
                  '--start-frame',60,'--end-frame',60,'--output',root/'mixed-off')
    assert mixed_auto['frames']==mixed_off['frames']
    # A source whose decoded geometry changes cannot produce one immutable Qwen grid. The full
    # initial index must reject it before a lazy reader can reach the change during inference.
    changed_segment=root/'changed-geometry-segment.ts'
    ffmpeg('-f','lavfi','-i','testsrc2=size=128x96:rate=25:duration=2',
           '-c:v','mpeg2video','-g','12','-f','mpegts',changed_segment)
    changed_geometry=root/'changed-geometry.ts'
    changed_geometry.write_bytes(progressive_segment.read_bytes()+changed_segment.read_bytes())
    changed=subprocess.run([app,'verify',str(changed_geometry),'--deinterlace','off'],
                           capture_output=True)
    assert changed.returncode != 0
    assert 'midstream geometry/pixel-format change is unsupported' in changed.stderr.decode()
    # Auto never instantiates bwdif for a progressive stream, including tiny valid videos that
    # bwdif itself rejects due to its minimum-height requirement.
    tiny=root/'tiny.mp4'
    ffmpeg('-f','lavfi','-i','color=size=32x2:rate=1:duration=3','-c:v','libx264',tiny)
    tiny_auto=cli('verify',tiny,'--start-frame',0,'--end-frame',1,'--deinterlace','auto')
    tiny_off=cli('verify',tiny,'--start-frame',0,'--end-frame',1,'--deinterlace','off')
    assert tiny_auto['frames']==tiny_off['frames']
    # A decoded KEY frame in an H.264 open GOP may still need references from the previous GOP.
    # Failed random-access preflight must fall back before publishing any selected frame.
    open_gop=root/'open-gop.mp4'
    ffmpeg('-f','lavfi','-i','testsrc2=size=160x96:rate=30:duration=6',
           '-c:v','libx264','-x264-params','open-gop=1:keyint=30:min-keyint=30:scenecut=0',
           '-bf','3',open_gop)
    open_result=cli('verify',open_gop,'--start-frame',32,'--end-frame',63,'--skip-frame',2)
    assert open_result['verified'] and not open_result['stats']['seek_used']
    assert open_result['stats']['seek_note'].startswith('sequential fallback:')
    for options in [('--scale','0'),('--scale','nan'),('--bbox','150,0,20,30'),('--skip-frame','-1'),
                    ('--start-frame','9999'),('--autotone','1'),('--wat','1'),('--end-frame','-1'),
                    ('--max-frames','1'),('--scale','100000'),('--skip-frame','9223372036854775807')]:
        p=subprocess.run([app,'verify',str(cfr),*options],capture_output=True)
        assert p.returncode != 0, options
        assert json.loads(p.stderr)['complete'] is False
    print('PASS: CFR/B-frames, exact/open-GOP seek, selection, timestamps, odd ROI, scale/alignment, PNG, VFR, mixed/tiny bwdif, geometry change, EOF, errors')
