function info = vsProbe(cfg, boxKey)
%VSPROBE Is a box reachable?
%   INFO = VSPROBE(CFG, BOXKEY) runs the bridge's reflection probe against
%   the box and returns its report: reachable, service, methods, reflection.
%   With no BOXKEY every configured box is probed and INFO is a table.
%
%   A probe costs about a second; a call to a box that is down costs a full
%   timeout, so this is worth doing before a long run.
%
%   See also VSCONFIG, VSRUN.

if nargin < 2
    keys = string(fieldnames(cfg.hosts))';
    name = strings(0,1); addr = strings(0,1); up = false(0,1);
    for k = keys
        one = vsProbe(cfg, k);
        name(end+1,1) = k;                              %#ok<AGROW>
        addr(end+1,1) = vsHost(cfg, k);                 %#ok<AGROW>
        up(end+1,1)   = isfield(one, 'reachable') && one.reachable; %#ok<AGROW>
    end
    info = table(name, addr, up, 'VariableNames', {'box', 'host', 'reachable'});
    return
end

host = vsHost(cfg, boxKey);
cmd = sprintf('%s "%s" probe --host %s', cfg.python, cfg.runner, host);
[~, output] = system(cmd);

info = struct('reachable', false, 'host', host);
lines = splitlines(string(output));
for k = numel(lines):-1:1
    line = strtrim(lines(k));
    if startsWith(line, "{")
        try
            info = jsondecode(line);
            return
        catch
        end
    end
end
end
