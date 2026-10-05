function [out, report] = vsTapnext(cfg, varargin)
%VSTAPNEXT Point tracking and the Tomasi-Kanade observation matrix.
%   OUT = VSTAPNEXT(CFG, 'video', PATH) tracks a grid of points through a
%   video. OUT = VSTAPNEXT(CFG, 'images', PATHS) does the same for frames.
%
%   State is per session and CUMULATIVE: each reply covers every frame the
%   session has seen so far, not only the ones in this call. 'grid_size'
%   only takes effect on a session's first frame - change it and you need a
%   new session (or vsReset) for it to matter.
%
%   Name/value
%     'images'      string array of frame paths     (exclusive with video)
%     'video'       one video file                  (exclusive with images)
%     'command'     "track" (default) | "reset" | "list"
%     'session'     session id              (default: cfg.session_id)
%     'grid_size'   grid_size^2 query points        (box default 32)
%     'frame_step'  video: sample every Nth frame   (box default 1)
%     'max_frames'  video: 0 = no cap               (box default 0)
%
%   OUT fields
%     tracks              FxNx2 - (y, x) in ORIGINAL pixels. Note the order:
%                         this box reports row first, unlike the keypoint
%                         boxes. Plot with tracks(:,p,2) against tracks(:,p,1).
%     visibles            FxN logical-ish visibility
%     observation_matrix  2FxN, x and y interleaved per frame, NaN where a
%                         point is absent - the matrix a factorisation wants
%     config_json         the box's reply config
%
%   Example
%     out = vsTapnext(cfg, 'video', "clip.mp4", 'grid_size', 24);
%     P = double(out.observation_matrix);
%     imagesc(P); xlabel("point"); ylabel("2F");
%
%   See also VSRUN, VSRESET, VSTRACKSTREAM.

p = inputParser;
p.FunctionName = 'vsTapnext';
addParameter(p, 'images', string.empty);
addParameter(p, 'video', string.empty);
addParameter(p, 'command', "track");
addParameter(p, 'session', "");
addParameter(p, 'grid_size', []);
addParameter(p, 'frame_step', []);
addParameter(p, 'max_frames', []);
addParameter(p, 'name', "tapnext");
parse(p, varargin{:});
a = p.Results;

images = string(a.images);
video  = string(a.video);
if ~isempty(images) && ~isempty(video)
    error("visionist:tapnext", "images and video are mutually exclusive");
end

session = string(a.session);
if strlength(session) == 0
    session = cfg.session_id;
end

% session_id at the top of the section (same contract as yolo).
section = struct('command', string(a.command), 'session_id', session);

params = struct();
params = vsSet(params, 'grid_size',  a.grid_size);
params = vsSet(params, 'frame_step', a.frame_step);
params = vsSet(params, 'max_frames', a.max_frames);
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(video);  io.files.video  = video;  end
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "tapnext", section, io);
end
