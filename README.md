# VisionIST from MATLAB

A MATLAB client for the box fleet. All box knowledge lives here, in MATLAB;
the Python side is one generic bridge that knows no box.

```
visionist_run.py          the bridge: one Envelope in, one .mat out
vsConfig.m                hosts, interpreter, workdir, session id
vsRun.m                   the only function that launches Python
vsProbe.m  vsReset.m      reachability; clear a box or one session
vsHost.m   vsSet.m        small helpers

vsClip.m      vsFeatures.m   vsLangSam.m    vsLightglue.m
vsMoge.m      vsOpencv.m     vsSbert.m      vsTapnext.m
vsUnimatch.m  vsVggt.m       vsYolo.m       one function per box

vsTrackStream.m           stream frames through lightglue, collect matches
vsObservation.m           match edges -> tracks -> observation matrix

visionist_walkthrough.m   the five demos, section by section
```

## The split

`visionist_run.py` takes a host, a JSON file holding the whole `config_json`,
and generic data flags:

```
--file FIELD=PATH     file bytes      (repeat the field to build a list)
--text FIELD=STRING   a literal string
--number FIELD=1.5    a number
```

It sends the Envelope, decodes the reply with whatever codec the box declared,
and writes every field into a `.mat`. It contains no box name, no command name
and no field name — adding a box never touches it.

Each `vs*` function holds one box's contract: its config section, its commands,
its parameters, the data fields it reads, and what the reply means. `help
vsYolo` (or any other) prints it.

```matlab
cfg = vsConfig();
cfg.session_id = "alice";

out = vsYolo(cfg, 'video', "clip.mp4", 'frame_step', 30, 'conf', 0.4);
d   = out.detections{2};         % struct: boxes, labels, scores, track_ids

out = vsMoge(cfg, 'images', ["a.jpg" "b.jpg"], 'fov_x', 60);
depth = out.results{1}.depth;    % H x W metres
```

## How replies land in the .mat

| the box returned | you get |
|---|---|
| numeric array | a numeric variable of the same name |
| torch tensor | numeric (needs `torch` on the Python side) |
| list of dicts | a cell array of structs: `results{1}.depth` |
| JSON object/list | numeric and char fields inside a struct, or `<field>_json` |
| one binary payload | written to disk, path in `<field>_file` |
| list of binaries | written to disk, paths in `<field>_files`, count in `<field>_count` |

Every reply also carries `config_json` (the box's own status section) and
`field_map_json` (original field name → the variable it became, for names
MATLAB cannot use verbatim).

## Adding a box

Copy `vsSbert.m` — the shortest one — change the section name, the parameters
and the documentation, and add the address to `cfg.hosts` in `vsConfig.m`.
Nothing on the Python side changes.

## Two things that bite

**Indexing.** Arrays arrive exactly as the box produced them, so anything that
is an *index into another array* is 0-based and needs a `+1`. That is
`lightglue`'s `matches`. Coordinates are pixel positions and need no shift —
but `tapnext` reports `(y, x)` while the keypoint boxes report `(x, y)`.

**Sessions.** `cfg.session_id` keys `yolo`'s track ids, `tapnext`'s accumulated
tracks and the `lightglue` stream window. Re-calling with the same id
*continues* that session: `tapnext` returns every frame the session has ever
seen, not just the ones in your call. `vsReset(cfg, box, id)` starts clean —
and it knows the id goes at the top of the section for `yolo` and `tapnext`
but inside `parameters` for `lightglue`.

## Requirements

Python with `visionist-client`, `numpy`, `scipy` — plus `torch` for the boxes
that declare the torch codec (`clip`, `tapnext`, `textEmbedding`, `vggt`).
MATLAB with Image Processing Toolbox for the walkthrough's plotting.
