function [out, report] = vsRun(cfg, boxKey, section, io)
%VSRUN Send one Envelope to a box and load the reply.
%   [OUT, REPORT] = VSRUN(CFG, BOXKEY, SECTION, IO) wraps SECTION as
%   {BOXKEY: SECTION}, hands it to visionist_run.py together with the data
%   fields in IO, and loads the .mat the bridge writes.
%
%   This is the ONLY function that launches Python. Everything the box
%   needs - its section name, its command, its parameters, its data field
%   names - arrives here already built by the caller, so neither this
%   function nor the Python side knows anything about any box.
%
%   BOXKEY   char/string: the config_json section name ("yolo", "lang_sam", ...)
%   SECTION  struct: the contents of that section (command, parameters, ...)
%   IO       struct, all fields optional:
%              .name     base name for the .mat and the assets folder
%              .files    struct: data field -> string array of file paths
%              .texts    struct: data field -> string array of literals
%              .numbers  struct: data field -> numeric array
%              .host     override cfg.hosts.(boxKey)
%
%   OUT is the loaded .mat as a struct. Beyond the box's own fields it always
%   carries config_json (the box's whole reply config) and field_map_json
%   (original field name -> the variable it became, for names MATLAB cannot
%   use verbatim). REPORT is the bridge's own JSON report, decoded.
%
%   Raises visionist:boxError when the box does not answer "done", with the
%   box's own reason.
%
%   See also VSCONFIG, VSPROBE, VSRESET.

arguments
    cfg struct
    boxKey (1,1) string
    section struct
    io struct = struct()
end

if isfield(io, 'host') && strlength(io.host) > 0
    host = string(io.host);
else
    host = vsHost(cfg, boxKey);
end
if isfield(io, 'name') && strlength(string(io.name)) > 0
    name = string(io.name);
else
    name = boxKey;
end

% --- the request --------------------------------------------------------
config = struct();
config.(boxKey) = section;

cfgPath = fullfile(cfg.workdir, name + "_request.json");
matPath = fullfile(cfg.workdir, name + ".mat");
fid = fopen(cfgPath, 'w');
if fid < 0
    error("visionist:cannotWrite", "cannot write %s", cfgPath);
end
% jsonencode on a struct gives exactly the config_json the box expects.
fprintf(fid, '%s', jsonencode(config));
fclose(fid);

args = sprintf('run --host %s --config "%s" --out "%s" --timeout %g', ...
               host, cfgPath, matPath, cfg.timeout);
args = args + dataArgs(io, 'files',   '--file');
args = args + dataArgs(io, 'texts',   '--text');
args = args + dataArgs(io, 'numbers', '--number');

cmd = sprintf('%s "%s" %s', cfg.python, cfg.runner, args);
if cfg.verbose
    fprintf("$ %s\n", cmd);
end

% --- run ----------------------------------------------------------------
t = tic;
[rc, output] = system(cmd);
report = parseReport(output);

if rc ~= 0
    reason = strtrim(output);
    if isstruct(report) && isfield(report, 'error') && ~isempty(report.error)
        reason = string(report.error);
    end
    error("visionist:boxError", "%s box failed:\n%s", boxKey, reason);
end
if ~isfile(matPath)
    error("visionist:noOutput", ...
          "the bridge reported success but wrote no %s", matPath);
end

out = load(matPath);
if cfg.verbose
    fprintf("  %s (%.1fs)\n", strjoin(string(fieldnames(out))', ", "), toc(t));
end
end

% ------------------------------------------------------------------------
function s = dataArgs(io, group, flag)
%DATAARGS Turn one io group into repeated --file / --text / --number flags.
%   Repeating a field is what makes it a list on the wire, so a string array
%   of two image paths becomes two --file flags with the same field name.
s = "";
if ~isfield(io, group) || isempty(io.(group))
    return
end
fields = fieldnames(io.(group));
for k = 1:numel(fields)
    values = io.(group).(fields{k});
    if isnumeric(values)
        for v = values(:)'
            s = s + sprintf(' %s %s=%.17g', flag, fields{k}, v);
        end
    else
        for v = string(values(:))'
            s = s + sprintf(' %s "%s=%s"', flag, fields{k}, v);
        end
    end
end
end

function report = parseReport(output)
%PARSEREPORT The bridge prints one JSON line; stderr may add plain text.
report = struct();
lines = splitlines(string(output));
for k = numel(lines):-1:1
    line = strtrim(lines(k));
    if startsWith(line, "{")
        try
            report = jsondecode(line);
            return
        catch
            % not the report line after all - keep looking
        end
    end
end
end
