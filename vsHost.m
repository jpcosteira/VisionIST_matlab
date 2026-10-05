function host = vsHost(cfg, boxKey)
%VSHOST The address configured for one box.
%   HOST = VSHOST(CFG, BOXKEY) looks BOXKEY up in CFG.hosts and errors with
%   the available keys if it is not there - a typo in a box name should say
%   so here rather than fail as a connection timeout later.
%
%   See also VSCONFIG.

boxKey = string(boxKey);
if ~isfield(cfg, 'hosts') || ~isfield(cfg.hosts, boxKey)
    known = "";
    if isfield(cfg, 'hosts')
        known = strjoin(string(fieldnames(cfg.hosts))', ", ");
    end
    error("visionist:unknownBox", ...
          "no host configured for box %s (known: %s)", boxKey, known);
end
host = string(cfg.hosts.(boxKey));
end
