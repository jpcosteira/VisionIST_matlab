function [out, report] = vsYolo(cfg, varargin)
%VSYOLO Object detection and tracking (YOLO box).
%   OUT = VSYOLO(CFG, 'video', PATH) detects and tracks through a video.
%   OUT = VSYOLO(CFG, 'images', PATHS) does the same for ordered frames.
%
%   Tracking is always on: every detection carries a track id that is stable
%   within the session, so re-calling with the same session continues the
%   same tracker rather than starting a new one.
%
%   Name/value (unset ones are not sent, so the box's default applies):
%     'images'          string array of image paths     (exclusive with video)
%     'video'           one video file                  (exclusive with images)
%     'command'         "detect" (default) | "reset" | "list"
%     'session'         session id              (default: cfg.session_id)
%     'conf'            confidence threshold           (box default 0.25)
%     'iou'             NMS IoU threshold              (box default 0.70)
%     'imgsz'           inference size                 (box default 640)
%     'max_det'         cap on detections per frame    (box default 300)
%     'classes'         numeric class ids to keep, e.g. [16 49]
%     'save_annotated'  logical, return annotated JPEGs (box default true)
%     'frame_step'      video: sample every Nth frame  (box default 1)
%     'max_frames'      video: cap on decoded frames   (box default 1024)
%     'weights'         checkpoint name, e.g. "yolov8s.pt"
%     'device'          "cpu" | "cuda" | "cuda:0"
%
%   OUT fields
%     detections       cell array, one struct per frame:
%                        .frame_index .width .height
%                        .boxes      Kx4 [x1 y1 x2 y2] in input pixels
%                        .class_ids  Kx1   .labels Kx1 cell of char
%                        .scores     Kx1   .track_ids Kx1
%     annotated_files  newline-joined JPEG paths (when save_annotated)
%     annotated_count  how many
%     config_json      the box's reply config
%
%   Example
%     cfg = vsConfig();
%     out = vsYolo(cfg, 'video', "clip.mp4", 'frame_step', 30, 'conf', 0.4);
%     d   = out.detections{2};
%     imshow(imread(strtrim(splitlines(out.annotated_files)(2))));
%
%   See also VSRUN, VSRESET, VSCONFIG.

p = inputParser;
p.FunctionName = 'vsYolo';
addParameter(p, 'images', string.empty);
addParameter(p, 'video', string.empty);
addParameter(p, 'command', "detect");
addParameter(p, 'session', "");
addParameter(p, 'conf', []);
addParameter(p, 'iou', []);
addParameter(p, 'imgsz', []);
addParameter(p, 'max_det', []);
addParameter(p, 'classes', []);
addParameter(p, 'save_annotated', []);
addParameter(p, 'frame_step', []);
addParameter(p, 'max_frames', []);
addParameter(p, 'weights', "");
addParameter(p, 'device', "");
addParameter(p, 'name', "yolo");
parse(p, varargin{:});
a = p.Results;

images = string(a.images);
video  = string(a.video);
if ~isempty(images) && ~isempty(video)
    error("visionist:yolo", "images and video are mutually exclusive");
end

session = string(a.session);
if strlength(session) == 0
    session = cfg.session_id;
end

% --- the config section (session_id sits at the TOP of the section here) --
section = struct('command', string(a.command), 'session_id', session);

params = struct();
params = vsSet(params, 'conf',           a.conf);
params = vsSet(params, 'iou',            a.iou);
params = vsSet(params, 'imgsz',          a.imgsz);
params = vsSet(params, 'max_det',        a.max_det);
params = vsSet(params, 'save_annotated', a.save_annotated);
params = vsSet(params, 'frame_step',     a.frame_step);
params = vsSet(params, 'max_frames',     a.max_frames);
params = vsSet(params, 'weights',        string(a.weights));
params = vsSet(params, 'device',         string(a.device));
if ~isempty(a.classes)
    % Must be a JSON array even for one class, hence the cell.
    params.classes = num2cell(double(a.classes(:)'));
end
if ~isempty(fieldnames(params))
    section.parameters = params;
end

io = struct('name', string(a.name));
if ~isempty(video);  io.files.video  = video;  end
if ~isempty(images); io.files.images = images; end

[out, report] = vsRun(cfg, "yolo", section, io);
end
