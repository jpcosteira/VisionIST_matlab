function [out, report] = vsLightglue(cfg, varargin)
%VSLIGHTGLUE Feature extraction and matching (LightGlue box).
%   OUT = VSLIGHTGLUE(CFG, 'images', [A B]) extracts SuperPoint (or DISK)
%   features in both images and matches them.
%   OUT = VSLIGHTGLUE(CFG, 'images', A) extracts features from one image.
%   OUT = VSLIGHTGLUE(CFG, 'images', F, 'command', "stream") feeds one frame
%   into a sliding-window session - see VSTRACKSTREAM, which drives that.
%
%   Name/value
%     'images'             image path(s): 1 or 2 for "match", exactly 1 for
%                          "stream"
%     'command'            "match" (default) | "stream" | "reset" | "list"
%     'feature_extractor'  "SUPERPOINT" (default) | "DISK"
%     'max_keypoints'      box default 1024, minimum 8
%     'filter_threshold'   match confidence cut, <= 1
%     'window'             stream: stored references to match against (1..16)
%     'session'            stream/reset session id  (default: cfg.session_id)
%     'device'             "cpu" | "cuda" | "cuda:0"
%
%   OUT fields, "match"
%     keypoints    n_images x N x 2 (x, y), zero-padded to the widest image
%     descriptors  n_images x N x D
%     scores       n_images x N
%     matches      Mx2 index pairs - ZERO-BASED, as the box returned them;
%                  add 1 before indexing keypoints in MATLAB
%     confidence   Mx1   (matches and confidence appear only with 2 images)
%
%   OUT fields, "stream"
%     keypoints    (J+1) x N x 2: rows 1..J are the stored references,
%                  newest first; row J+1 is the frame you just sent
%     kp_counts    (J+1)x1 true keypoint count per row
%     matches_1..J Mx2 - column 1 indexes reference j, column 2 the new
%                  frame; both ZERO-BASED
%     confidence_1..J
%
%   Example
%     out = vsLightglue(cfg, 'images', [a b], 'max_keypoints', 1024);
%     m   = double(out.matches) + 1;            % to MATLAB indexing
%     p0  = squeeze(out.keypoints(1, m(:,1), :));
%     p1  = squeeze(out.keypoints(2, m(:,2), :));
%
%   See also VSTRACKSTREAM, VSOPENCV, VSRESET.

p = inputParser;
p.FunctionName = 'vsLightglue';
addParameter(p, 'images', string.empty);
addParameter(p, 'command', "match");
addParameter(p, 'feature_extractor', "");
addParameter(p, 'max_keypoints', []);
addParameter(p, 'filter_threshold', []);
addParameter(p, 'window', []);
addParameter(p, 'session', "");
addParameter(p, 'device', "");
addParameter(p, 'name', "lightglue");
parse(p, varargin{:});
a = p.Results;

images  = string(a.images);
command = string(a.command);

if command == "match" && numel(images) > 2
    error("visionist:lightglue", ...
          "match is pairwise: 1 or 2 images, got %d", numel(images));
end
if command == "stream" && numel(images) ~= 1
    error("visionist:lightglue", ...
          "stream takes exactly 1 image per call, got %d", numel(images));
end

section = struct('command', command);

% This box reads session_id from PARAMETERS, not from the section top level.
params = struct();
params = vsSet(params, 'feature_extractor', upper(string(a.feature_extractor)));
params = vsSet(params, 'max_keypoints',     a.max_keypoints);
params = vsSet(params, 'filter_threshold',  a.filter_threshold);
params = vsSet(params, 'window',            a.window);
params = vsSet(params, 'device',            string(a.device));

if command == "stream" || command == "reset"
    session = string(a.session);
    if strlength(session) == 0
        session = cfg.session_id;
    end
    params.session_id = session;
end
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "lightglue", section, io);
end
