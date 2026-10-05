function [out, report] = vsUnimatch(cfg, varargin)
%VSUNIMATCH Dense optical flow, stereo disparity, or multi-view depth.
%   OUT = VSUNIMATCH(CFG, 'images', [A B])                  optical flow
%   OUT = VSUNIMATCH(CFG, 'images', [L R], 'command', "stereo")  disparity
%   OUT = VSUNIMATCH(CFG, 'images', [A B], 'command', "depth", ...
%                    'intrinsics', K, 'pose', T)            metric depth
%
%   Only the flow checkpoint ships inside the box. "stereo" and "depth" need
%   'model' pointed at a checkpoint - a URL is fetched once on first call.
%   Inputs above about 1.6 MP are downscaled inside the box (the reply says
%   auto_resized) and the result is scaled back.
%
%   Name/value
%     'images'            2 paths for flow/stereo; 1 or 2 for depth
%     'command'           "flow" (default) | "stereo" | "depth" | "reset"
%     'model'             checkpoint name, local path, or http(s) URL
%     'inference_size'    [h w] to force the internal resolution
%     'padding_factor'    box default 16
%     'num_scales'        box default 1
%     'attn_type'         box default "swin"
%     'attn_splits_list'  per-scale, e.g. [2] - length must equal num_scales
%     'corr_radius_list'  per-scale, e.g. [-1]
%     'prop_radius_list'  per-scale, e.g. [-1]
%     'reg_refine'        logical; needs num_scales <= 2
%     'num_reg_refine'    box default 1
%     'device'            "cpu" | "cuda" | "cuda:0"
%   flow only
%     'pred_bidir_flow'   logical, also compute the backward flow
%     'fwd_bwd_check'     logical, occlusion masks (needs pred_bidir_flow)
%   depth only
%     'intrinsics'        3x3 or 4x4 matrix, or [fx fy cx cy]  (required)
%     'pose'              4x4 relative pose, reference -> target
%     'min_depth'         box default 0.5     'max_depth' box default 10.0
%     'num_depth_candidates'  box default 64
%
%   OUT fields (all numeric, at input resolution)
%     flow        HxWx2 pixel shift of image 1 -> image 2 (:,:,1)=u (:,:,2)=v
%     bwd_flow    HxWx2, only with pred_bidir_flow
%     occ_fwd, occ_bwd  HxW, only with fwd_bwd_check
%     disparity   HxW pixels, left view; bigger = closer
%     depth       HxW metres
%
%   Example
%     out = vsUnimatch(cfg, 'images', [a b]);
%     mag = hypot(out.flow(:,:,1), out.flow(:,:,2));
%
%   See also VSRUN, VSMOGE.

p = inputParser;
p.FunctionName = 'vsUnimatch';
addParameter(p, 'images', string.empty);
addParameter(p, 'command', "flow");
addParameter(p, 'model', "");
addParameter(p, 'inference_size', []);
addParameter(p, 'padding_factor', []);
addParameter(p, 'num_scales', []);
addParameter(p, 'attn_type', "");
addParameter(p, 'attn_splits_list', []);
addParameter(p, 'corr_radius_list', []);
addParameter(p, 'prop_radius_list', []);
addParameter(p, 'reg_refine', []);
addParameter(p, 'num_reg_refine', []);
addParameter(p, 'device', "");
addParameter(p, 'pred_bidir_flow', []);
addParameter(p, 'fwd_bwd_check', []);
addParameter(p, 'intrinsics', []);
addParameter(p, 'pose', []);
addParameter(p, 'min_depth', []);
addParameter(p, 'max_depth', []);
addParameter(p, 'num_depth_candidates', []);
addParameter(p, 'pred_bidir_depth', []);
addParameter(p, 'name', "unimatch");
parse(p, varargin{:});
a = p.Results;

images  = string(a.images);
command = lower(string(a.command));

if any(command == ["flow", "stereo"]) && numel(images) ~= 2
    error("visionist:unimatch", ...
          "%s needs exactly 2 images, got %d", command, numel(images));
end
if ~isempty(a.fwd_bwd_check) && a.fwd_bwd_check && ...
        (isempty(a.pred_bidir_flow) || ~a.pred_bidir_flow)
    error("visionist:unimatch", ...
          "fwd_bwd_check needs pred_bidir_flow true");
end
if command == "depth" && isempty(a.intrinsics)
    error("visionist:unimatch", "depth needs 'intrinsics'");
end

section = struct('command', command);

params = struct();
params = vsSet(params, 'model',          string(a.model));
params = vsSet(params, 'padding_factor', a.padding_factor);
params = vsSet(params, 'num_scales',     a.num_scales);
params = vsSet(params, 'attn_type',      string(a.attn_type));
params = vsSet(params, 'reg_refine',     a.reg_refine);
params = vsSet(params, 'num_reg_refine', a.num_reg_refine);
params = vsSet(params, 'device',         string(a.device));
params = vsSet(params, 'pred_bidir_flow', a.pred_bidir_flow);
params = vsSet(params, 'fwd_bwd_check',  a.fwd_bwd_check);
params = vsSet(params, 'min_depth',      a.min_depth);
params = vsSet(params, 'max_depth',      a.max_depth);
params = vsSet(params, 'num_depth_candidates', a.num_depth_candidates);
params = vsSet(params, 'pred_bidir_depth', a.pred_bidir_depth);

% These must be JSON ARRAYS even when they hold a single value, so they go
% through num2cell: jsonencode([2]) is "2", jsonencode({2}) is "[2]".
for f = ["inference_size", "attn_splits_list", "corr_radius_list", ...
         "prop_radius_list"]
    v = a.(f);
    if ~isempty(v)
        params.(f) = num2cell(double(v(:)'));
    end
end
if ~isempty(a.intrinsics)
    params.intrinsics = matrixCell(a.intrinsics);
end
if ~isempty(a.pose)
    params.pose = matrixCell(a.pose);
end
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "unimatch", section, io);
end

% ------------------------------------------------------------------------
function c = matrixCell(M)
%MATRIXCELL A matrix as nested JSON arrays (a vector stays one flat array).
M = double(M);
if isvector(M)
    c = num2cell(M(:)');
    return
end
c = cell(1, size(M, 1));
for r = 1:size(M, 1)
    c{r} = num2cell(M(r, :));
end
end
