function [out, report] = vsD4rt(cfg, varargin)
%VSD4RT 4D reconstruction and tracking (OpenD4RT box).
%   OUT = VSD4RT(CFG, 'video', PATH) follows a grid of points through a clip
%   in 3D. OUT = VSD4RT(CFG, 'images', PATHS) does the same for ordered frames.
%   OUT = VSD4RT(CFG, 'video', PATH, 'points', UV) follows the points you give.
%   OUT = VSD4RT(CFG, ..., 'command', "reconstruct") returns a point cloud.
%   OUT = VSD4RT(CFG, ..., 'command', "cameras") returns K and poses.
%
%   The model answers one question - where is the point at pixel (u, v) of
%   frame t_src, at frame t_tgt, in the camera frame of t_cam - and all three
%   commands are built from it, which is why they share one box.
%
%   Name/value
%     'images'       string array of frame paths     (exclusive with video)
%     'video'        one video file                  (exclusive with images)
%     'points'       Qx2 [u v] normalised to [0, 1], or a flat [u1 v1 u2 v2 ...]
%                    row. track only; omit for a grid.
%     'command'      "track" (default) | "reconstruct" | "cameras" | "reset"
%     'device'       "cuda" | "cuda:N" | "cpu"
%     'frame_step'   sample every Nth frame          (box default 1)
%     'max_frames'   cap on decoded frames           (box default 48)
%     'chunk_size'   queries per decoder call        (box default 4096)
%   track
%     'grid_size'    side of the uv grid when no points are given (default 32)
%     't_src'        frame the query pixels are read from         (default 0)
%   reconstruct
%     'point_grid_size'  grid side                   (box default 64)
%     'max_points'       cap after gridding          (box default 4096)
%   cameras
%     'camera_grid_size' grid for the fits           (box default 16)
%     'intrinsics'       logical                     (box default true)
%     'extrinsics'       logical                     (box default true)
%
%   OUT fields, "track"
%     query_uv           Qx2 the pixels actually queried
%     tracks_xyz_ref0    QxTx3 each point per frame, in frame 1's camera frame
%                        - THIS is the one to use for motion
%     tracks_xyz_local   QxTx3 the same points, each re-expressed in its own
%                        frame's camera, so a static point appears to move
%                        whenever the camera does
%     tracks_uv          QxTx2 2D reprojection, normalised
%     tracks_visibility  QxT logical, already thresholded
%     tracks_confidence  QxT raw scalar, good for ranking, not a probability
%
%   OUT fields, "reconstruct"
%     points_xyz         TxPx3 point cloud per frame, frame-1 coordinates;
%                        NaN where the model could not resolve a point
%     points_visibility  TxP      points_confidence TxP      query_uv Px2
%
%   OUT fields, "cameras"
%     intrinsics         Tx3x3 per-frame K
%     extrinsics         Tx4x4 T_ref0_cam per frame
%     valid_intrinsics   Tx1 logical - a frame whose fit failed is left as
%     valid_extrinsics   Tx1 logical   identity and marked false, so CHECK these
%
%   Two things that will mislead you otherwise:
%
%   SCALE. Predictions are up to scale, not metric. Every upstream evaluation
%   aligns them to ground truth with a similarity transform before measuring,
%   so ratios and shapes are meaningful but absolute distances are not. Do not
%   read xyz as metres.
%
%   CLIP LENGTH. The model sees at most clip_frames at once (48 for the
%   default checkpoint). Upstream CLAMPS out-of-range timesteps instead of
%   rejecting them, so a longer video yields confident wrong answers rather
%   than an error. 'max_frames' defaults to the clip length for that reason -
%   raise it only if you know what the anchoring does.
%
%   Example
%     cfg = vsConfig();
%     out = vsD4rt(cfg, 'video', "clip.mp4", 'grid_size', 24);
%     xyz = out.tracks_xyz_ref0;                  % Q x T x 3
%     moved = vecnorm(squeeze(xyz(:,end,:) - xyz(:,1,:)), 2, 2);
%     [~, i] = max(moved);
%     plot3(squeeze(xyz(i,:,1)), squeeze(xyz(i,:,2)), squeeze(xyz(i,:,3)), '-o');
%     title(sprintf("most-displaced track (%.3f, up to scale)", moved(i)));
%
%   See also VSTAPNEXT, VSMOGE, VSVGGT, VSRUN.

p = inputParser;
p.FunctionName = 'vsD4rt';
addParameter(p, 'images', string.empty);
addParameter(p, 'video', string.empty);
addParameter(p, 'points', []);
addParameter(p, 'command', "track");
addParameter(p, 'grid_size', []);
addParameter(p, 't_src', []);
addParameter(p, 'point_grid_size', []);
addParameter(p, 'max_points', []);
addParameter(p, 'camera_grid_size', []);
addParameter(p, 'intrinsics', []);
addParameter(p, 'extrinsics', []);
addParameter(p, 'chunk_size', []);
addParameter(p, 'frame_step', []);
addParameter(p, 'max_frames', []);
addParameter(p, 'device', "");
addParameter(p, 'name', "d4rt");
parse(p, varargin{:});
a = p.Results;

images  = string(a.images);
video   = string(a.video);
command = string(a.command);

if ~isempty(images) && ~isempty(video)
    error("visionist:d4rt", "images and video are mutually exclusive");
end
if command ~= "reset" && isempty(images) && isempty(video)
    error("visionist:d4rt", "give 'video' or 'images'");
end
if ~isempty(a.points) && command ~= "track"
    error("visionist:d4rt", "'points' only applies to the track command");
end

section = struct('command', command);

params = struct();
params = vsSet(params, 'device',           string(a.device));
params = vsSet(params, 'frame_step',       a.frame_step);
params = vsSet(params, 'max_frames',       a.max_frames);
params = vsSet(params, 'chunk_size',       a.chunk_size);
params = vsSet(params, 'grid_size',        a.grid_size);
params = vsSet(params, 't_src',            a.t_src);
params = vsSet(params, 'point_grid_size',  a.point_grid_size);
params = vsSet(params, 'max_points',       a.max_points);
params = vsSet(params, 'camera_grid_size', a.camera_grid_size);
params = vsSet(params, 'intrinsics',       a.intrinsics);
params = vsSet(params, 'extrinsics',       a.extrinsics);
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(video);  io.files.video  = video;  end
if ~isempty(images); io.files.images = images; end

if ~isempty(a.points)
    uv = double(a.points);
    if size(uv, 2) == 2 && ~isvector(uv)
        % Qx2 -> [u1 v1 u2 v2 ...]: transpose first, since (:) reads columns.
        flat = reshape(uv.', 1, []);
    elseif isvector(uv)
        flat = reshape(uv, 1, []);
    else
        error("visionist:d4rt", ...
              "'points' must be Qx2 or a flat [u v u v ...] row, got %s", ...
              mat2str(size(uv)));
    end
    if mod(numel(flat), 2) ~= 0
        error("visionist:d4rt", "'points' has %d values; it must be (u, v) pairs", ...
              numel(flat));
    end
    if numel(flat) > 0 && (min(flat) < 0 || max(flat) > 1)
        error("visionist:d4rt", ...
              "'points' must be normalised to [0, 1] (got %.3f..%.3f)", ...
              min(flat), max(flat));
    end
    % data.points is the ff FloatList the box reads first.
    io.numbers.points = flat;
end

[out, report] = vsRun(cfg, "d4rt", section, io);
end
