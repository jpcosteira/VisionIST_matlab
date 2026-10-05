function cfg = vsConfig(varargin)
%VSCONFIG Settings shared by every vs* function.
%   CFG = VSCONFIG() returns the defaults. Override any of them with
%   name/value pairs, or edit the returned struct afterwards:
%
%       cfg = vsConfig('workdir', "/tmp/vs", 'session_id', "alice");
%       cfg.hosts.moge = "ifetch.isr.tecnico.ulisboa.pt:9067";
%
%   Fields
%     python      interpreter to launch ("py -3" on Windows)
%     runner      path to visionist_run.py (the box-agnostic bridge)
%     workdir     where .mat results, request JSON and binary assets land
%     session_id  default session for the stateful boxes
%     hosts       struct of box key -> "host:port"
%     ports       which port convention to build hosts from (see below)
%     timeout     per-request seconds
%     verbose     print each command before running it
%
%   The host keys are the boxes' own config_json section names, which is
%   also what every vs* function passes to vsRun - so "sbert", not
%   "textEmbedding", and "lang_sam", not "lang_segm".
%
%   See also VSRUN, VSPROBE, VSRESET.

p = inputParser;
p.FunctionName = 'vsConfig';
addParameter(p, 'python', "python3");
addParameter(p, 'runner', "");
addParameter(p, 'workdir', "");
addParameter(p, 'session_id', "matlab");
addParameter(p, 'timeout', 1800);
addParameter(p, 'verbose', true);
addParameter(p, 'hostPrefix', "localhost");
addParameter(p, 'ports', "legacy");     % "legacy" | "generated" - see below
addParameter(p, 'basePort', 9061);      % only used by "generated"
parse(p, varargin{:});
a = p.Results;

cfg = struct();
cfg.python     = string(a.python);
cfg.timeout    = a.timeout;
cfg.verbose    = logical(a.verbose);
cfg.session_id = string(a.session_id);

if strlength(a.runner) > 0
    cfg.runner = string(a.runner);
else
    % Default: next to this file, which is where the two ship together.
    cfg.runner = string(fullfile(fileparts(mfilename('fullpath')), ...
                                 'visionist_run.py'));
end

if strlength(a.workdir) > 0
    cfg.workdir = string(a.workdir);
else
    cfg.workdir = string(fullfile(pwd, "visionist_out"));
end
if ~isfolder(cfg.workdir)
    mkdir(cfg.workdir);
end

% Fleet port map. There are two conventions in circulation and they do NOT
% agree, so 'ports' picks one:
%
%   "legacy"     (default) the hand-written fleet/docker-compose.yml of the
%                original VisionIST repo, in the order its services were
%                declared. d4rt and open_clip were added afterwards, each at
%                the next free port.
%   "generated"  a fleet produced by VisionIST_Library's tools/make_fleet.py,
%                which assigns ports in BOX-NAME order from base. Adding a box
%                therefore shifts every port after it alphabetically.
%
% Whichever you use, the fleet you are actually running is the authority:
% its docker-compose.yml has the host ports, and its data/fleet.json the
% service-name addresses. Override any single one afterwards:
%
%     cfg = vsConfig('ports', "generated");
%     cfg.hosts.d4rt = "ifetch.isr.tecnico.ulisboa.pt:9062";

h = string(a.hostPrefix);
names = ["clip" "d4rt" "features" "lang_sam" "lightglue" "moge" ...
         "open_clip" "opencv" "sbert" "tapnext" "unimatch" "vggt" "yolo"];

cfg.hosts = struct();
if string(a.ports) == "generated"
    % Name order from base, exactly as make_fleet.py assigns them.
    for k = 1:numel(names)
        cfg.hosts.(names(k)) = h + ":" + string(a.basePort + k - 1);
    end
else
    legacy = struct( ...
        'clip',       9061, ...
        'sbert',      9062, ...   % the textEmbedding box
        'tapnext',    9063, ...
        'lang_sam',   9064, ...   % the lang_segm box
        'opencv',     9065, ...
        'vggt',       9066, ...
        'moge',       9067, ...
        'yolo',       9068, ...
        'lightglue',  9069, ...
        'unimatch',   9070, ...
        'features',   9071, ...
        'd4rt',       9072, ...   % added after the others, so it goes last
        'open_clip',  9073);      % ... and so does this one
    for k = 1:numel(names)
        cfg.hosts.(names(k)) = h + ":" + string(legacy.(names(k)));
    end
end

if ~isfile(cfg.runner)
    warning("vsConfig:noRunner", ...
            "visionist_run.py not found at %s - set cfg.runner", cfg.runner);
end
end
