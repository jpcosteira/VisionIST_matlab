function s = vsSet(s, name, value)
%VSSET Add a field to a parameters struct, but only if it was given.
%   S = VSSET(S, NAME, VALUE) sets S.(NAME) = VALUE unless VALUE is empty.
%
%   Every vs* box function defaults its optional arguments to [] and funnels
%   them through here, so a parameter the caller did not set never reaches
%   the box at all and the box's own default applies. That matters: sending
%   an explicit null or 0 is not the same as sending nothing.
%
%   Logical values pass through as logicals so jsonencode writes true/false.
%   To send a one-element JSON array (some unimatch parameters want one),
%   pass a 1x1 cell: vsSet(s, 'attn_splits_list', {2}) -> "attn_splits_list":[2]
%
%   See also VSRUN, JSONENCODE.

if isempty(value) && ~islogical(value)
    return
end
if isstring(value) && isscalar(value) && strlength(value) == 0
    return
end
s.(name) = value;
end
