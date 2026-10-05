function [out, report] = vsVggt(cfg, varargin)
%VSVGGT Multi-view 3D reconstruction (VGGT box).
%   OUT = VSVGGT(CFG, 'images', PATHS) reconstructs a scene from an image
%   sequence: per-view depth, world points, camera poses, and a GLB model.
%
%   Every input image must have the SAME dimensions - the box errors and
%   lists the shapes it got otherwise. This is the heaviest box in the
%   fleet (about 5 GB of weights) and the first call blocks on the model
%   load.
%
%   Name/value
%     'images'          string array of image paths, same size each
%     'command'         omit to reconstruct, or "reset"
%     'conf_threshold'  percentile for the GLB point filter (box default 30)
%     'device'          "cpu" | "cuda" | "cuda:0"
%
%   OUT fields (S = number of views)
%     world_points       SxHxWx3        world_points_conf  SxHxW
%     depth              SxHxWx1        depth_conf         SxHxW
%     extrinsic          Sx3x4          intrinsic          Sx3x3
%     images             Sx3xHxW, the preprocessed tensor the model saw
%     glb_file           path to the GLB written beside the .mat
%
%   The numeric fields arrive over the torch codec (torch needed on the
%   Python side); glb_file is raw binary, so the bridge writes it to disk
%   and gives you the path.
%
%   Example
%     out = vsVggt(cfg, 'images', ["v1.jpg" "v2.jpg" "v3.jpg"]);
%     K = squeeze(out.intrinsic(1,:,:));
%     web(out.glb_file)   % or open it in any glTF viewer
%
%   See also VSMOGE, VSUNIMATCH.

p = inputParser;
p.FunctionName = 'vsVggt';
addParameter(p, 'images', string.empty);
addParameter(p, 'command', "");
addParameter(p, 'conf_threshold', []);
addParameter(p, 'device', "");
addParameter(p, 'name', "vggt");
parse(p, varargin{:});
a = p.Results;

images = string(a.images);
command = string(a.command);

section = struct();
if strlength(command) > 0
    section.command = command;
end

params = struct();
params = vsSet(params, 'conf_threshold', a.conf_threshold);
params = vsSet(params, 'device',         string(a.device));
if ~isempty(fieldnames(params))
    section.parameters = params;
end
if isempty(fieldnames(section))
    section.command = "reconstruct";   % anything but "reset" reconstructs
end

io = struct('name', string(a.name));
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "vggt", section, io);
end
