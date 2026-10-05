function [out, report] = vsLangSam(cfg, varargin)
%VSLANGSAM Text-guided segmentation (LangSAM / lang_segm box).
%   OUT = VSLANGSAM(CFG, 'images', PATHS, 'prompt', ["a dog" "a chair"])
%   segments every prompt in every image.
%
%   The prompts go at the TOP of the config section as text_prompt, not
%   inside parameters, and the same prompt list is applied to each image.
%   The box's section name is "lang_sam" even though the directory is
%   images/lang_segm.
%
%   Name/value
%     'images'          string array of image paths
%     'prompt'          string array of phrases (required)
%     'command'         omit to segment, or "reset"
%     'box_threshold'   detection cut, box default 0.3
%     'text_threshold'  phrase-grounding cut, box default 0.25
%     'device'          "cpu" | "cuda" | "cuda:0"
%
%   OUT fields
%     results   1xN cell, one struct per image, with the LangSAM prediction:
%               masks, boxes, scores and labels for that image
%
%   Example
%     out = vsLangSam(cfg, 'images', "street.jpg", 'prompt', "a bicycle");
%     r = out.results{1};
%
%   See also VSCLIP, VSYOLO.

p = inputParser;
p.FunctionName = 'vsLangSam';
addParameter(p, 'images', string.empty);
addParameter(p, 'prompt', string.empty);
addParameter(p, 'command', "");
addParameter(p, 'box_threshold', []);
addParameter(p, 'text_threshold', []);
addParameter(p, 'device', "");
addParameter(p, 'name', "lang_sam");
parse(p, varargin{:});
a = p.Results;

images = string(a.images);
prompt = string(a.prompt);
command = string(a.command);
if command ~= "reset" && isempty(prompt)
    error("visionist:langSam", "'prompt' is required");
end

section = struct();
if strlength(command) > 0
    section.command = command;
end
if ~isempty(prompt)
    % A JSON array of strings, even for one phrase - hence cellstr.
    section.text_prompt = cellstr(prompt(:)');
end

params = struct();
params = vsSet(params, 'box_threshold',  a.box_threshold);
params = vsSet(params, 'text_threshold', a.text_threshold);
params = vsSet(params, 'device',         string(a.device));
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "lang_sam", section, io);
end
