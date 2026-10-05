function [out, report] = vsClip(cfg, varargin)
%VSCLIP CLIP image and text embeddings, with their cross-modal similarity.
%   OUT = VSCLIP(CFG, 'images', PATHS, 'texts', PROMPTS) embeds both and
%   returns the image-text similarity logits.
%
%   Both inputs are required: the box answers empty_request without images
%   and errors without texts. The model is fixed at ViT-B/32 - the 'model'
%   parameter exists for API symmetry and is ignored.
%
%   Name/value
%     'images'   string array of image paths   (required)
%     'texts'    string array of prompts       (required)
%     'command'  omit for encoding, or "reset"
%
%   OUT fields
%     image_emb   num_images x 512
%     text_emb    num_texts x 512
%     similarity  num_images x num_texts cross-modal logits
%
%   These arrive over the torch codec, so the Python side needs torch
%   installed to decode them; without it the bridge reports the field as a
%   raw binary payload instead.
%
%   Example
%     out = vsClip(cfg, 'images', "dog.jpg", 'texts', ["a dog" "a cat"]);
%     [~, best] = max(out.similarity, [], 2);
%
%   See also VSSBERT, VSLANGSAM.

p = inputParser;
p.FunctionName = 'vsClip';
addParameter(p, 'images', string.empty);
addParameter(p, 'texts', string.empty);
addParameter(p, 'command', "");
addParameter(p, 'name', "clip");
parse(p, varargin{:});
a = p.Results;

images = string(a.images);
texts  = string(a.texts);
command = string(a.command);
if command ~= "reset"
    if isempty(images)
        error("visionist:clip", "'images' is required");
    end
    if isempty(texts)
        error("visionist:clip", "'texts' is required");
    end
end

section = struct();
if strlength(command) > 0
    section.command = command;
end
if isempty(fieldnames(section))
    % The box only inspects "command" for "reset"; anything else encodes.
    section.command = "encode";
end

io = struct('name', string(a.name));
if ~isempty(images); io.files.images = images; end
if ~isempty(texts);  io.texts.texts  = texts;  end

[out, report] = vsRun(cfg, "clip", section, io);
end
