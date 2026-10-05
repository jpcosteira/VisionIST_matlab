function [out, report] = vsMoge(cfg, varargin)
%VSMOGE Monocular 3D geometry: depth, point map, normals (MoGe-3 box).
%   OUT = VSMOGE(CFG, 'images', PATHS) runs MoGe-3 on each image.
%
%   CUDA only - the box rejects 'device', "cpu" outright, because the sparse
%   refinement uses CUDA-only kernels.
%
%   Name/value
%     'images'            string array of image paths
%     'command'           "infer" (default) | "reset"
%     'fov_x'             known horizontal FOV in degrees; pins the metric
%                         scale down. Without it MoGe estimates the FOV and
%                         an FOV error becomes a scale error.
%     'refine_steps'      0 fastest, box default 3
%     'resolution_level'  0..9, box default 9
%     'fp16'              logical, box default true (about 2x faster)
%     'device'            "cuda" | "cuda:0"  ("cpu" is rejected)
%
%   OUT fields
%     results   1xN cell, one struct per image:
%                 .depth       HxW metres, OpenCV camera coords (z forward)
%                 .points      HxWx3 point map
%                 .normal      HxWx3 unit normals
%                 .mask        HxW validity (1 = usable pixel)
%                 .intrinsics  3x3 normalised
%
%   Invalid pixels (sky, masked regions) come back as Inf or NaN in .depth
%   rather than a stand-in number - filter with .mask before computing
%   anything metric.
%
%   Example
%     out = vsMoge(cfg, 'images', ["a.jpg" "b.jpg"], 'fov_x', 60);
%     d = out.results{1}.depth; m = logical(out.results{1}.mask);
%     fprintf("median depth %.2f m\n", median(d(m)));
%
%   See also VSRUN, VSUNIMATCH, VSVGGT.

p = inputParser;
p.FunctionName = 'vsMoge';
addParameter(p, 'images', string.empty);
addParameter(p, 'command', "infer");
addParameter(p, 'fov_x', []);
addParameter(p, 'refine_steps', []);
addParameter(p, 'resolution_level', []);
addParameter(p, 'fp16', []);
addParameter(p, 'device', "");
addParameter(p, 'name', "moge");
parse(p, varargin{:});
a = p.Results;

if strlength(string(a.device)) > 0 && lower(string(a.device)) == "cpu"
    error("visionist:moge", ...
          "MoGe-3 is CUDA only; the box rejects device 'cpu'");
end

section = struct('command', string(a.command));

params = struct();
params = vsSet(params, 'fov_x',            a.fov_x);
params = vsSet(params, 'refine_steps',     a.refine_steps);
params = vsSet(params, 'resolution_level', a.resolution_level);
params = vsSet(params, 'fp16',             a.fp16);
params = vsSet(params, 'device',           string(a.device));
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
images = string(a.images);
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "moge", section, io);
end
