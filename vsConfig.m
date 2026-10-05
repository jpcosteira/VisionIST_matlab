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

% Fleet port map (fleet/docker-compose.yml maps each box's 8061 to these).
h = string(a.hostPrefix);
cfg.hosts = struct( ...
    'clip',       h + ":9061", ...
    'sbert',      h + ":9062", ...   % the textEmbedding box
    'tapnext',    h + ":9063", ...
    'lang_sam',   h + ":9064", ...   % the lang_segm box
    'opencv',     h + ":9065", ...
    'vggt',       h + ":9066", ...
    'moge',       h + ":9067", ...
    'yolo',       h + ":9068", ...
    'lightglue',  h + ":9069", ...
    'unimatch',   h + ":9070", ...
    'features',   h + ":9071", ...
    'd4rt',       h + ":9072");

if ~isfile(cfg.runner)
    warning("vsConfig:noRunner", ...
            "visionist_run.py not found at %s - set cfg.runner", cfg.runner);
end
end
