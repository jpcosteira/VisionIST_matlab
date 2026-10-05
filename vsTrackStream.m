function res = vsTrackStream(cfg, source, varargin)
%VSTRACKSTREAM Feed frames through the lightglue stream and collect matches.
%   RES = VSTRACKSTREAM(CFG, VIDEOPATH) samples frames from a video and
%   streams them through one lightglue session.
%   RES = VSTRACKSTREAM(CFG, IMAGEPATHS) streams the given frames in order.
%
%   The box keeps a sliding window of stored frames per session: each call
%   sends ONE frame and the box reports which of its keypoints match which
%   keypoints of the last few frames. This function collects those match
%   edges; VSOBSERVATION turns them into tracks and an observation matrix.
%   The split is deliberate - the network pass is slow and the association
%   is cheap, so you can try both association modes without re-streaming.
%
%   Name/value
%     'n_frames'           how many frames to sample from a video (default 12)
%     'window'             box-side sliding window, 1..16 (default 3)
%     'session'            session id            (default: cfg.session_id)
%     'feature_extractor'  "SUPERPOINT" (default) | "DISK"
%     'max_keypoints'      box default 1024
%     'quality'            JPEG quality for extracted frames (default 85)
%     'reset_first'        clear the session before streaming (default true)
%
%   RES fields
%     edges       Mx4 [refFrame refKpt newFrame newKpt], frames and keypoint
%                 indices BOTH 1-BASED (converted from the box's 0-based)
%     frameKpts   1xF cell, frameKpts{f} is Nx2 (x, y) for frame f
%     framePaths  Fx1 string, the frames as sent
%     numFrames   F
%     window      the window the box actually used
%
%   Example
%     s = vsTrackStream(cfg, "clip.mp4", 'n_frames', 12, 'window', 3);
%     b = vsObservation(s, 'mode', "backbone", 'min_alive', 3);
%     g = vsObservation(s, 'mode', "greedy",   'min_alive', 3);
%
%   See also VSOBSERVATION, VSLIGHTGLUE, VSTAPNEXT.

p = inputParser;
p.FunctionName = 'vsTrackStream';
addParameter(p, 'n_frames', 12);
addParameter(p, 'window', 3);
addParameter(p, 'session', "");
addParameter(p, 'feature_extractor', "SUPERPOINT");
addParameter(p, 'max_keypoints', []);
addParameter(p, 'quality', 85);
addParameter(p, 'reset_first', true);
parse(p, varargin{:});
a = p.Results;

session = string(a.session);
if strlength(session) == 0
    session = cfg.session_id;
end

framePaths = resolveFrames(cfg, string(source), a.n_frames, a.quality);
F = numel(framePaths);
if F < 2
    error("visionist:trackStream", "need at least 2 frames, got %d", F);
end

if a.reset_first
    vsReset(cfg, "lightglue", session);
end

edges = zeros(0, 4);
frameKpts = cell(1, F);
usedWindow = a.window;

for k = 1:F
    out = vsLightglue(cfg, 'images', framePaths(k), 'command', "stream", ...
                      'window', a.window, 'session', session, ...
                      'feature_extractor', a.feature_extractor, ...
                      'max_keypoints', a.max_keypoints, ...
                      'name', sprintf("lgstream_%03d", k));

    reply = jsondecode(out.config_json);
    section = reply.lightglue;
    J = 0;
    if isfield(section, 'window')
        J = double(section.window);
    end
    usedWindow = max(usedWindow, J);

    % keypoints is (J+1) x N x 2: rows 1..J are the stored references
    % (newest first), the LAST row is the frame we just sent.
    kp = double(out.keypoints);
    frameKpts{k} = squeeze(kp(end, :, :));
    if isfield(out, 'kp_counts')
        n = double(out.kp_counts(end));
        frameKpts{k} = frameKpts{k}(1:n, :);
    end

    % matches_j pairs reference frame (k-j) with the new frame k.
    % Column 1 indexes the reference, column 2 the new frame, both 0-based.
    for j = 1:J
        field = sprintf('matches_%d', j);
        if ~isfield(out, field) || isempty(out.(field))
            continue
        end
        m = double(out.(field));
        if size(m, 2) ~= 2
            continue
        end
        refFrame = k - j;
        if refFrame < 1
            continue
        end
        edges = [edges; ...
                 repmat(refFrame, size(m,1), 1), m(:,1) + 1, ...
                 repmat(k, size(m,1), 1),        m(:,2) + 1];   %#ok<AGROW>
    end
end

res = struct('edges', edges, 'frameKpts', {frameKpts}, ...
             'framePaths', framePaths, 'numFrames', F, ...
             'window', usedWindow, 'session', session);
end

% ------------------------------------------------------------------------
function paths = resolveFrames(cfg, source, nFrames, quality)
%RESOLVEFRAMES Image paths as given, or frames extracted from a video.
if numel(source) > 1 || (isfile(source(1)) && isImageFile(source(1)))
    paths = source(:);
    return
end

video = source(1);
if ~isfile(video)
    error("visionist:trackStream", "no such file: %s", video);
end

v = VideoReader(video);
total = 0;
frames = {};
while hasFrame(v)                    % robust across releases; NumFrames can
    frames{end+1} = readFrame(v);    %#ok<AGROW>  be unavailable for some codecs
    total = total + 1;
end
if total < 2
    error("visionist:trackStream", "%s holds %d frame(s)", video, total);
end

nFrames = max(2, min(nFrames, total));
idx = round(linspace(1, total, nFrames));

outDir = fullfile(cfg.workdir, "stream_frames");
if ~isfolder(outDir); mkdir(outDir); end
paths = strings(nFrames, 1);
for i = 1:nFrames
    paths(i) = string(fullfile(outDir, sprintf("frame_%03d.jpg", i)));
    imwrite(frames{idx(i)}, paths(i), 'Quality', quality);
end
end

function tf = isImageFile(p)
[~, ~, e] = fileparts(p);
tf = any(lower(string(e)) == [".jpg" ".jpeg" ".png" ".bmp" ".tif" ".tiff"]);
end
