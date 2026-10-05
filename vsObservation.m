function res = vsObservation(stream, varargin)
%VSOBSERVATION Match edges -> tracks -> Tomasi-Kanade observation matrix.
%   RES = VSOBSERVATION(STREAM) turns the output of VSTRACKSTREAM into an
%   observation matrix. Pure computation: no box, no network, no Python.
%
%   Why it is not simply "one connected component = one track": union-find
%   is transitive, so two nearby but distinct keypoints that both match a
%   shared neighbour get glued into one component. A component holding two
%   nodes in the SAME frame is not one world point and cannot be an honest
%   matrix column, which holds one (x, y) per frame. So each component is
%   PEELED: repeatedly take the longest chain that visits each frame at most
%   once, emit it as a track, remove it, repeat. One malformed blob becomes
%   several valid tracks and nothing is discarded.
%
%   Name/value
%     'mode'       "backbone" (default) - peel over matches of every frame
%                  gap, so a point that blinks out for one frame is re-linked
%                  by a 2- or 3-frame match and stays ONE track with a NaN gap.
%                  "greedy" - only adjacent-frame (gap 1) matches, so such a
%                  point dies and its return becomes a new track.
%                  Both read the same edges; only which are traversable differs.
%     'min_alive'  keep tracks seen in at least this many frames (default 2)
%
%   RES fields
%     obs_matrix   2F x T: rows 2f-1 and 2f are x and y in frame f, NaN where
%                  that track has no point in that frame
%     tracks       1xT cell, each Kx2 [frame kpt]
%     stats        struct: nComponents, nMergedComponents, nTracksKept,
%                  nTracksDropped, nGappedTracks, perFrameCounts,
%                  longestTrackFrames
%
%   Example
%     s = vsTrackStream(cfg, "clip.mp4");
%     b = vsObservation(s, 'mode', "backbone", 'min_alive', 3);
%     imagesc(b.obs_matrix); xlabel("track"); ylabel("2F");
%     fprintf("%d tracks, %.1f%% missing\n", size(b.obs_matrix, 2), ...
%             100 * mean(isnan(b.obs_matrix(:))));
%
%   See also VSTRACKSTREAM.

p = inputParser;
p.FunctionName = 'vsObservation';
addParameter(p, 'mode', "backbone");
addParameter(p, 'min_alive', 2);
parse(p, varargin{:});
mode = string(p.Results.mode);
minAlive = p.Results.min_alive;

if ~any(mode == ["backbone", "greedy"])
    error("visionist:observation", ...
          "mode must be backbone or greedy, got %s", mode);
end
if minAlive < 1
    error("visionist:observation", "min_alive must be >= 1");
end

edges = stream.edges;
frameKpts = stream.frameKpts;
F = stream.numFrames;

if isempty(edges)
    res = emptyResult(F, mode, minAlive);
    return
end

% --- nodes -------------------------------------------------------------
% A node is one observation: (frame, keypoint). Encode it as a single
% number so sorting gives frame-major order, which is also topological
% order: every edge runs from an earlier frame to a later one.
K = max([max(edges(:,2)), max(edges(:,4)), 1]) + 1;
encode = @(f, i) (f - 1) * K + i;

idA = encode(edges(:,1), edges(:,2));
idB = encode(edges(:,3), edges(:,4));

[nodeIds, ~, back] = unique([idA; idB]);
n = numel(nodeIds);
ia = back(1:numel(idA));
ib = back(numel(idA)+1:end);

nodeFrame = floor((nodeIds - 1) / K) + 1;
nodeKpt   = nodeIds - (nodeFrame - 1) * K;

% --- union-find (path halving) -----------------------------------------
parent = (1:n)';
for e = 1:numel(ia)
    ra = findRoot(ia(e));
    rb = findRoot(ib(e));
    if ra ~= rb
        parent(ra) = rb;
    end
end
roots = zeros(n, 1);
for k = 1:n
    roots(k) = findRoot(k);
end
[~, ~, compOf] = unique(roots);
nComp = max(compOf);

% --- predecessors ------------------------------------------------------
% greedy keeps only gap-1 edges; backbone keeps them all.
keep = true(numel(ia), 1);
if mode == "greedy"
    keep = (edges(:,3) - edges(:,1)) <= 1;
end
pred = cell(n, 1);
ka = ia(keep); kb = ib(keep);
for e = 1:numel(ka)
    pred{kb(e)}(end+1) = ka(e);
end

% --- peel each component ----------------------------------------------
tracks = {};
nDropped = 0;
for c = 1:nComp
    members = find(compOf == c);
    remaining = false(n, 1);
    remaining(members) = true;
    while any(remaining)
        path = longestPath(remaining);
        remaining(path) = false;
        if numel(path) >= minAlive
            tracks{end+1} = path;                               %#ok<AGROW>
        else
            nDropped = nDropped + 1;
        end
    end
end

% Stable order: earliest frame, then keypoint index.
if ~isempty(tracks)
    firstId = cellfun(@(t) min(nodeIds(t)), tracks);
    [~, order] = sort(firstId);
    tracks = tracks(order);
end

% --- the matrix --------------------------------------------------------
T = numel(tracks);
P = nan(2 * F, T);
trackNodes = cell(1, T);
for t = 1:T
    path = tracks{t};
    nodes = zeros(numel(path), 2);
    for q = 1:numel(path)
        f = nodeFrame(path(q));
        i = nodeKpt(path(q));
        nodes(q, :) = [f, i];
        xy = frameKpts{f};
        if i <= size(xy, 1)
            P(2*f - 1, t) = xy(i, 1);
            P(2*f,     t) = xy(i, 2);
        end
    end
    trackNodes{t} = nodes;
end

% --- stats -------------------------------------------------------------
nMerged = 0;
for c = 1:nComp
    frames = nodeFrame(compOf == c);
    if numel(frames) > numel(unique(frames))
        nMerged = nMerged + 1;
    end
end
nGapped = 0;
perFrame = zeros(1, F);
longest = 0;
for t = 1:T
    f = trackNodes{t}(:, 1);
    if (max(f) - min(f) + 1) > numel(f)
        nGapped = nGapped + 1;
    end
    perFrame(f) = perFrame(f) + 1;
    longest = max(longest, numel(unique(f)));
end

res = struct();
res.obs_matrix = P;
res.tracks = trackNodes;
res.mode = mode;
res.min_alive = minAlive;
res.numFrames = F;
res.stats = struct('nComponents', nComp, 'nMergedComponents', nMerged, ...
                   'nTracksKept', T, 'nTracksDropped', nDropped, ...
                   'nGappedTracks', nGapped, 'perFrameCounts', perFrame, ...
                   'longestTrackFrames', longest);

% ---------------------------------------------------------------- nested
    function r = findRoot(x)
        while parent(x) ~= x
            parent(x) = parent(parent(x));   % path halving
            x = parent(x);
        end
        r = x;
    end

    function path = longestPath(mask)
        %LONGESTPATH Longest frame-increasing chain inside MASK.
        %   Nodes are already in topological order when sorted by id, and
        %   edges only ever go forward in time, so one left-to-right sweep
        %   is enough and the chain visits each frame at most once.
        idxs = find(mask);
        best = ones(n, 1);
        par = zeros(n, 1);
        for q = 1:numel(idxs)
            v = idxs(q);
            for w = pred{v}
                if mask(w) && best(w) + 1 > best(v)
                    best(v) = best(w) + 1;
                    par(v) = w;
                end
            end
        end
        % Longest chain; ties go to the one ending latest, as in the client.
        scores = best(idxs) * (F + 1) + nodeFrame(idxs);
        [~, w] = max(scores);
        last = idxs(w);
        path = last;
        while par(last) ~= 0
            last = par(last);
            path(end+1) = last;                                 %#ok<AGROW>
        end
        path = flip(path(:));
    end
end

% ------------------------------------------------------------------------
function res = emptyResult(F, mode, minAlive)
res = struct('obs_matrix', nan(2*F, 0), 'tracks', {{}}, 'mode', mode, ...
             'min_alive', minAlive, 'numFrames', F, ...
             'stats', struct('nComponents', 0, 'nMergedComponents', 0, ...
                             'nTracksKept', 0, 'nTracksDropped', 0, ...
                             'nGappedTracks', 0, 'perFrameCounts', zeros(1,F), ...
                             'longestTrackFrames', 0));
end
