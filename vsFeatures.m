function [out, report] = vsFeatures(cfg, varargin)
%VSFEATURES SIFT keypoints in the SIFT-Extractor layout (features box).
%   OUT = VSFEATURES(CFG, 'images', PATHS) extracts SIFT features, returning
%   the same (2 + 128) x N matrix the original command-line tool wrote to a
%   MATLAB .mat under the key kp.
%
%   Name/value
%     'images'              string array of image paths
%     'names'               original file names, one per image; they name the
%                           .mat artifacts the box reports back
%     'command'             "extract" (default) | "reset"
%     'nfeatures'           cap on keypoints, 0 = keep all (box default 0).
%                           Approximate: OpenCV keeps keypoints tied with the
%                           last retained one, so 500 can return 501.
%     'n_octave_layers'     box default 3        'sigma' box default 1.6
%     'contrast_threshold'  box default 0.04     'edge_threshold' default 10
%     'draw'                logical, return the red-dot JPEGs (default true)
%     'radius'              annotation dot radius (box default 2)
%     'mat'                 logical, also return one .mat per image
%
%   OUT fields
%     keypoints        num_images x 130 x N, zero-padded to the widest image.
%                      Row 1 = x, row 2 = y, rows 3..130 = the descriptor.
%     counts           num_images x 1, the true N per image - ALWAYS slice
%                      with this, the rest of N is padding
%     annotated_files  newline-joined JPEG paths (when draw)
%     mat_files        newline-joined .mat paths (when mat), key kp
%
%   Example
%     out = vsFeatures(cfg, 'images', "frame.jpg", 'nfeatures', 500);
%     n   = double(out.counts(1));
%     xy  = squeeze(out.keypoints(1, 1:2, 1:n))';     % n x 2
%     des = squeeze(out.keypoints(1, 3:end, 1:n))';   % n x 128
%
%   See also VSOPENCV, VSLIGHTGLUE.

p = inputParser;
p.FunctionName = 'vsFeatures';
addParameter(p, 'images', string.empty);
addParameter(p, 'names', string.empty);
addParameter(p, 'command', "extract");
addParameter(p, 'nfeatures', []);
addParameter(p, 'n_octave_layers', []);
addParameter(p, 'contrast_threshold', []);
addParameter(p, 'edge_threshold', []);
addParameter(p, 'sigma', []);
addParameter(p, 'draw', []);
addParameter(p, 'radius', []);
addParameter(p, 'mat', []);
addParameter(p, 'name', "features");
parse(p, varargin{:});
a = p.Results;

images = string(a.images);
names  = string(a.names);
if ~isempty(names) && numel(names) ~= numel(images)
    error("visionist:features", ...
          "names has %d entries for %d images", numel(names), numel(images));
end

section = struct('command', string(a.command));

params = struct();
params = vsSet(params, 'nfeatures',          a.nfeatures);
params = vsSet(params, 'n_octave_layers',    a.n_octave_layers);
params = vsSet(params, 'contrast_threshold', a.contrast_threshold);
params = vsSet(params, 'edge_threshold',     a.edge_threshold);
params = vsSet(params, 'sigma',              a.sigma);
params = vsSet(params, 'draw',               a.draw);
params = vsSet(params, 'radius',             a.radius);
params = vsSet(params, 'mat',                a.mat);
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(images); io.files.images = images; end
if ~isempty(names);  io.texts.names  = names;  end

[out, report] = vsRun(cfg, "features", section, io);
end
