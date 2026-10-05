%% VisionIST fleet - MATLAB walkthrough
% The MATLAB counterpart of notebooks/boxes_walkthrough.ipynb: yolo detection
% on a video, tapnext point tracking, lightglue matching and stream tracking,
% unimatch optical flow and stereo disparity.
%
% All box knowledge lives on this side. Each box has a vs* function that
% builds its config section, names its data fields and documents what comes
% back; visionist_run.py is a generic bridge that sends ONE Envelope and saves
% the reply as a .mat, and knows no box, no command and no field name. Adding
% a box means adding a vs* function here - the Python never changes.
%
% Run it section by section (Ctrl+Enter). To make a Live Script: Save As >
% MATLAB Live Code File (.mlx), or
%
%   matlab.internal.liveeditor.openAndSave('visionist_walkthrough.m', ...
%                                          'visionist_walkthrough.mlx');
%
% Prerequisites: the fleet is up (cd fleet && docker compose up -d); Python
% with visionist-client, numpy, scipy (and torch for the boxes that declare
% the torch codec: clip, tapnext, textEmbedding, vggt); Image Processing
% Toolbox for the plotting.

%% Settings
% vsConfig holds the interpreter, the bridge path, the fleet addresses and the
% session id. Everything else is derived from it.

cfg = vsConfig();                      % add 'hostPrefix', "ifetch..." for a
                                       % remote fleet, or edit cfg.hosts below
cfg.session_id = "matlab-walkthrough";
% cfg.hosts.moge = "ifetch.isr.tecnico.ulisboa.pt:9067";

repo = string(fullfile(pwd, "VisionIST-PIV"));   % <- adjust if it lives elsewhere

% The original notebook uses ../cozinha.mp4, which is not in the repo (*.mp4
% is gitignored); apple.mp4 ships with the tapnext box.
%video = fullfile(repo, "/Users/jpc/Downloads/ssofia.mp4");
video = "/Users/jpc/Downloads/ssofia.mp4";
if ~isfile(video)
    video = fullfile(repo, "images", "tapnext_tracker", "test", "apple.mp4");
end
pair   = [fullfile(repo, "images", "lightglue_box", "test", "00.jpg"), ...
          fullfile(repo, "images", "lightglue_box", "test", "01.jpg")];
flowIm = [fullfile(repo, "images", "unimatch", "test", "flow_0.jpg"), ...
          fullfile(repo, "images", "unimatch", "test", "flow_1.jpg")];
stereo = [fullfile(repo, "images", "unimatch", "test", "stereo_0.png"), ...
          fullfile(repo, "images", "unimatch", "test", "stereo_1.png")];

assert(isfile(video), "no video at %s", video);
fprintf("python : %s\nbridge : %s\nout    : %s\nsession: %s\nvideo  : %s\n", ...
        cfg.python, cfg.runner, cfg.workdir, cfg.session_id, video);

%% 1. Is the fleet up?
% One reflection probe per configured box. A box that is down fails here in a
% second rather than after a full timeout mid-run.

disp(vsProbe(cfg));

%% 2. YOLO - detection and tracking on a video
% Tracking is always on, so each detection carries a track id that is stable
% within cfg.session_id.

yolo = vsYolo(cfg, 'video', video, 'frame_step', 30, 'save_annotated', true);

det = yolo.detections;                 % cell, one struct per sampled frame
fprintf("%d frame record(s)\n", numel(det));

r = det{min(2, numel(det))};
fprintf("frame %d (%d x %d): %d detection(s)\n", ...
        r.frame_index, r.width, r.height, size(r.boxes, 1));
disp(table(string(r.labels), r.scores(:), r.track_ids(:), r.boxes, ...
           'VariableNames', {'label', 'score', 'track_id', 'box_xyxy'}));

%%
% One annotated frame, drawn by the box.

files = splitlines(strtrim(string(yolo.annotated_files)));
files = files(strlength(files) > 0);
if isempty(files)
    disp("no annotated frames");
else
    k = min(2, numel(files));
    figure; imshow(imread(files(k)));
    title(sprintf("yolo - annotated frame %d of %d", k, numel(files)));
end

%% 3. TAPNext - point tracking and the observation matrix
% observation_matrix is the Tomasi-Kanade P matrix: 2F rows (x and y per
% frame) by one column per tracked point.
cfg.hosts.tapnext="sipgpu1.isrnet:9063";
cfg.session_id="lixo";
tap = vsTapnext(cfg, 'video', video, 'grid_size', 64);

P = double(tap.observation_matrix);
fprintf("tracks %s | visibles %s | P %s\n", mat2str(size(tap.tracks)), ...
        mat2str(size(tap.visibles)), mat2str(size(P)));
fprintf("missing entries: %.1f%%\n", 100 * mean(isnan(P(:))));

figure; imagesc(P); colorbar;
xlabel("tracked point"); ylabel("2F (x, y interleaved per frame)");
title("tapnext - observation matrix");

%%
% The trajectories over the first frame. Note the coordinate order: this box
% reports tracks as (y, x), unlike the keypoint boxes, so the column index 2
% is the horizontal axis.

T = double(tap.tracks);                % F x N x 2, (y, x)
v = VideoReader(video);
frame1 = readFrame(v);

figure; imshow(frame1); hold on;
step = max(1, round(size(T, 2) / 200));
for q = 1:step:size(T, 2)
    plot(squeeze(T(:, q, 2)), squeeze(T(:, q, 1)), '-', 'LineWidth', 0.5);
end
hold off;
title(sprintf("tapnext - %d trajectories", numel(1:step:size(T, 2))));

%% 4. LightGlue - matching an image pair
% The box returns keypoints per image plus the match index pairs. Those
% indices are the box's own, so they are 0-based; vsLightglue says so in its
% help and the conversion is the single +1 below.

lg = vsLightglue(cfg, 'images', pair, 'feature_extractor', "SUPERPOINT", ...
                 'max_keypoints', 1024);

m = double(lg.matches) + 1;            % to MATLAB indexing
p0 = squeeze(lg.keypoints(1, m(:, 1), :));
p1 = squeeze(lg.keypoints(2, m(:, 2), :));
fprintf("%d matches | mean confidence %.3f\n", size(m, 1), mean(lg.confidence));

figure('Position', [100 100 1100 450]);
pts = {p0, p1};
for k = 1:2
    subplot(1, 2, k);
    imshow(imread(pair(k))); hold on;
    scatter(pts{k}(:, 1), pts{k}(:, 2), 4, 'r', 'filled');
    [~, nm, ex] = fileparts(pair(k));
    title(sprintf("%s%s - %d matched points", nm, ex, size(pts{k}, 1)));
    hold off;
end

%%
% The correspondences as lines across a side-by-side montage.

I0 = imread(pair(1)); I1 = imread(pair(2));
h = max(size(I0, 1), size(I1, 1));
canvas = zeros(h, size(I0, 2) + size(I1, 2), 3, 'like', I0);
canvas(1:size(I0,1), 1:size(I0,2), :) = I0;
canvas(1:size(I1,1), size(I0,2) + (1:size(I1,2)), :) = I1;

figure; imshow(canvas); hold on;
n = min(150, size(p0, 1));
sel = unique(round(linspace(1, size(p0, 1), n)));
plot([p0(sel,1)'; p1(sel,1)' + size(I0,2)], [p0(sel,2)'; p1(sel,2)'], ...
     'LineWidth', 0.3);
hold off;
title(sprintf("lightglue - %d of %d correspondences", numel(sel), size(p0,1)));

%% 5. LightGlue stream - an observation matrix from the video
% vsTrackStream extracts the frames, streams them through one session and
% collects the match edges. vsObservation then builds tracks and the matrix -
% pure MATLAB, no box involved, so trying the other association mode costs
% nothing: the slow network pass is already done.

s = vsTrackStream(cfg, video, 'n_frames', 12, 'window', 3);
fprintf("%d frames | %d match edges\n", s.numFrames, size(s.edges, 1));

bb = vsObservation(s, 'mode', "backbone", 'min_alive', 3);
gg = vsObservation(s, 'mode', "greedy",   'min_alive', 3);

fprintf("backbone: %d tracks, %.1f%% missing, %d gap-bridged\n", ...
        bb.stats.nTracksKept, 100*mean(isnan(bb.obs_matrix(:))), ...
        bb.stats.nGappedTracks);
fprintf("greedy  : %d tracks, %.1f%% missing\n", ...
        gg.stats.nTracksKept, 100*mean(isnan(gg.obs_matrix(:))));

figure('Position', [100 100 1100 420]);
subplot(1,2,1); imagesc(bb.obs_matrix); colorbar;
title(sprintf("backbone - %d tracks", size(bb.obs_matrix, 2)));
xlabel("track"); ylabel("2F");
subplot(1,2,2); imagesc(gg.obs_matrix); colorbar;
title(sprintf("greedy - %d tracks", size(gg.obs_matrix, 2)));
xlabel("track"); ylabel("2F");

%%
% Why the two differ: backbone can re-link a point that vanished for one
% frame through a 2- or 3-frame match, keeping it as one track with a NaN
% gap. Greedy only follows adjacent-frame matches, so the same point dies and
% its return starts a new, shorter track - more columns, each covering less.

%% 6. UniMatch - dense optical flow
% flow is H x W x 2: where each pixel of image 1 went in image 2, in pixels.

fl = vsUnimatch(cfg, 'images', flowIm);

flow = double(fl.flow);
u = flow(:,:,1); v2 = flow(:,:,2);
mag = hypot(u, v2);
fprintf("flow %s | mean shift %.2f px | max %.2f px\n", ...
        mat2str(size(flow)), mean(mag(:)), max(mag(:)));

[H, W, ~] = size(flow);
step = max(1, round(max(H, W) / 40));
[xs, ys] = meshgrid(1:step:W, 1:step:H);

figure('Position', [100 100 1200 450]);
subplot(1,2,1);
imshow(imread(flowIm(1))); hold on;
quiver(xs, ys, u(1:step:end, 1:step:end), v2(1:step:end, 1:step:end), 0, 'y');
title("optical flow (arrows: where each pixel went)"); hold off;
subplot(1,2,2);
imagesc(mag); axis image off; colorbar; title("flow magnitude (px)");

%% 7. UniMatch - stereo disparity
% A rectified pair in, per-pixel disparity out (pixels; bigger = closer). Only
% the flow checkpoint ships in the box, so 'model' fetches the stereo one once.

model = "https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/" + ...
        "pretrained/gmstereo-scale1-sceneflow-124a438f.pth";
sd = vsUnimatch(cfg, 'images', stereo, 'command', "stereo", ...
                'model', model, 'inference_size', [800 1120]);

d = double(sd.disparity);
fprintf("disparity %s | mean %.1f | std %.1f | max %.0f px\n", ...
        mat2str(size(d)), mean(d(:)), std(d(:)), max(d(:)));

figure('Position', [100 100 1400 420]);
subplot(1,3,1); imshow(imread(stereo(1))); title("left");
subplot(1,3,2); imshow(imread(stereo(2))); title("right");
subplot(1,3,3); imagesc(d); axis image off; colormap(gca, hot); colorbar;
title("disparity (px, big = close)");

%% 8. Reset the sessions (optional)
% Clears this walkthrough's state on the stateful boxes. vsReset knows that
% the session id goes at the top of the section for yolo and tapnext but
% inside parameters for lightglue.

for box = ["yolo", "tapnext", "lightglue"]
    try
        vsReset(cfg, box, cfg.session_id);
        fprintf("  %-10s reset\n", box);
    catch err
        fprintf("  %-10s %s\n", box, err.message);
    end
end

%% Notes
% Where the knowledge lives. visionist_run.py builds an Envelope from a JSON
% config file plus --file/--text/--number data flags, and writes whatever
% comes back into a .mat. It has no list of boxes, no command names and no
% field names. Each vs* function here holds one box's contract: its section
% name, its commands, its parameters, which data fields it reads, and what
% the reply means. "help vsYolo" and friends document each one.
%
% Adding a box. Copy the shortest vs* function (vsSbert), change the section
% name, the parameters and the doc. Add its address to cfg.hosts in vsConfig.
% Nothing on the Python side changes.
%
% Indexing. Arrays arrive exactly as the box produced them, so any index INTO
% another array is 0-based and needs a +1 (lightglue's matches). Coordinates
% are pixel positions and need no shift - but mind that tapnext reports (y, x)
% while the keypoint boxes report (x, y).
%
% Missing entries. Observation matrices use NaN where a point was not seen.
% imagesc draws NaN as the lowest colour, so count with isnan rather than
% trusting the picture.
%
% Sessions. cfg.session_id keys yolo's track ids, tapnext's accumulated
% tracks and the lightglue stream window. Re-running section 2 or 3 with the
% same id CONTINUES that session - tapnext's reply covers every frame the
% session has seen, not just this call's. Section 8, or a new id, starts clean.

%% VGGT
cfg.hosts.vvg="sipgpu1.isrnet:9066"

res = b.run(
data={"images": ["frame_00.jpg", "frame_01.jpg", "frame_02.jpg"]},
config={"vggt": {"command": "reconstruct",
    "parameters": {"conf_threshold": 30}}},
)
print(res.config)                    # {"vggt": {"status": "done", ...}}
    print(res.world_points.shape)        # torch tensors, decoded via the declared codec
    with open("scene.glb", "wb") as f:
    f.write(res.glb_file)            # raw bytes (identity codec)