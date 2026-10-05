function report = vsReset(cfg, boxKey, sessionId)
%VSRESET Clear a box's state (one session, or all of it).
%   REPORT = VSRESET(CFG, BOXKEY) resets the whole box.
%   REPORT = VSRESET(CFG, BOXKEY, SESSIONID) resets just that session.
%
%   The session id goes in a different place depending on the box - at the
%   top of the config section for yolo and tapnext, inside "parameters" for
%   lightglue - which is exactly the kind of box knowledge that now lives on
%   this side. Boxes with no state accept "reset" as a no-op, so calling
%   this on any of them is harmless.
%
%   See also VSRUN, VSCONFIG.

arguments
    cfg struct
    boxKey (1,1) string
    sessionId (1,1) string = ""
end

% Where each stateful box reads session_id from.
SESSION_IN_PARAMETERS = ["lightglue"];
SESSION_AT_SECTION    = ["yolo", "tapnext"];

section = struct('command', "reset");
if strlength(sessionId) > 0
    if any(boxKey == SESSION_IN_PARAMETERS)
        section.parameters = struct('session_id', sessionId);
    elseif any(boxKey == SESSION_AT_SECTION)
        section.session_id = sessionId;
    else
        % Stateless box: the id is meaningless, the reset still acks.
        warning("visionist:statelessReset", ...
                "%s keeps no per-session state; resetting the box", boxKey);
    end
end

io = struct('name', boxKey + "_reset");
[~, report] = vsRun(cfg, boxKey, section, io);
end
