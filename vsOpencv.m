function [out, report] = vsOpencv(cfg, varargin)
%VSOPENCV Classic feature extraction and matching (OpenCV box).
%   OUT = VSOPENCV(CFG, 'images', [A B]) extracts features in both images,
%   matches them and estimates a RANSAC fundamental matrix.
%   OUT = VSOPENCV(CFG, 'images', A) extracts from one image only - the
%   matching fields are then ABSENT from OUT, not empty, so test with
%   isfield before using them.
%
%   Name/value
%     'images'             1 or 2 image paths
%     'command'            "match" (default) | "reset"
%     'feature_extractor'  "SIFT" (default) | "ORB" | "SUPERPOINT" | "DISK"
%                          SUPERPOINT and DISK go through LightGlue and need
%                          exactly 2 images
%     'ratio_thresh'       Lowe ratio, FLANN path only (box default 0.75)
%     'max_keypoints'      box default 500
%     'device'             "cpu" | "cuda" (LightGlue path only)
%
%   OUT fields
%     keypoints           n_images x N x 2, zero-padded
%     descriptors         n_images x N x D (128 SIFT, 32 ORB; the LightGlue
%                         path returns a zeros placeholder)
%     matches_inliers_a   Kx2 RANSAC inlier points in image 1   (2 images)
%     matches_inliers_b   Kx2 the same points in image 2        (2 images)
%     fundamental_matrix  3x3, or 0x0 when RANSAC had too few matches
%
%   vsFeatures is the other SIFT box: it answers "give me this image's
%   descriptors in the SIFT-Extractor layout", where this one answers
%   "match these two images".
%
%   Example
%     out = vsOpencv(cfg, 'images', [a b], 'feature_extractor', "SIFT");
%     F = out.fundamental_matrix;
%
%   See also VSFEATURES, VSLIGHTGLUE.

p = inputParser;
p.FunctionName = 'vsOpencv';
addParameter(p, 'images', string.empty);
addParameter(p, 'command', "match");
addParameter(p, 'feature_extractor', "");
addParameter(p, 'ratio_thresh', []);
addParameter(p, 'max_keypoints', []);
addParameter(p, 'device', "");
addParameter(p, 'name', "opencv");
parse(p, varargin{:});
a = p.Results;

images = string(a.images);
fx = upper(string(a.feature_extractor));
if any(strlength(fx) > 0) && any(contains(fx, ["SUPERPOINT", "DISK", "LIGHTGLUE"])) ...
        && numel(images) ~= 2
    error("visionist:opencv", ...
          "%s goes through LightGlue and needs exactly 2 images, got %d", ...
          fx, numel(images));
end

section = struct('command', string(a.command));

params = struct();
params = vsSet(params, 'feature_extractor', fx);
params = vsSet(params, 'ratio_thresh',      a.ratio_thresh);
params = vsSet(params, 'max_keypoints',     a.max_keypoints);
params = vsSet(params, 'device',            string(a.device));
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "opencv", section, io);
end
