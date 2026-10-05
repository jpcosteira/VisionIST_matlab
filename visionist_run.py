#!/usr/bin/env python3
"""Box-agnostic bridge: send one Envelope to any box, save the reply as .mat.

This script knows **no box, no command, no field name**. It mirrors what
``visionist_client.Visionist.run`` is: a generic transport. The caller (a
MATLAB function, a shell script, anything) supplies the whole ``config_json``
and names the data fields; this side builds the Envelope, decodes the reply
with whatever codec the box declared, and writes every field it got back into
a MATLAB .mat file.

    # the config is JSON written by the caller - its shape is the box's business
    python3 visionist_run.py run --host localhost:9067 \
        --config moge.json --file images=a.jpg --file images=b.jpg \
        --out moge.mat

    python3 visionist_run.py probe --host localhost:9067

Data arguments (repeat a field to build a list):

    --file  FIELD=PATH     file contents as bytes   (repeated -> BytesList)
    --text  FIELD=STRING   a literal string         (repeated -> StringList)
    --number FIELD=1.5     a number                 (repeated -> FloatList)

How reply fields land in the .mat, by what the codec produced:

    numeric array         -> a numeric variable of the same name
    bool array            -> a logical variable
    torch tensor          -> converted to numeric (needs torch installed)
    dict of arrays        -> a struct
    list of dicts         -> a 1xN cell array of structs  (results{1}.depth)
    JSON object or list   -> char, as <field>_json        (use jsondecode)
    one bytes payload     -> written to a file, path in char <field>_file
    list of bytes         -> written to files, newline-joined paths in
                             char <field>_files, count in <field>_count
    string / number       -> char / double

Always written: ``config_json`` (the box's whole reply config, as char) and
``field_map_json`` (original field name -> the .mat variable it became, for
names MATLAB cannot use verbatim).

Exit code is 0 on a ``done`` status, 1 otherwise, with a plain reason on
stderr. One JSON line of report goes to stdout either way.

Requires: visionist-client, numpy, scipy (torch only for boxes that declare
the ``torch`` codec: clip, tapnext, textEmbedding, vggt).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np

try:
    import scipy.io as sio
    from visionist_client import Visionist
except ImportError as e:                                   # pragma: no cover
    sys.exit(f"missing dependency ({e.name}): "
             f"pip install visionist-client numpy scipy")

MATLAB_NAME_MAX = 63          # MATLAB's namelengthmax


def fail(message: str):
    print(message, file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# Building the request
# --------------------------------------------------------------------------
def parse_pairs(items, kind):
    """``["field=value", ...]`` -> ``{field: [value, ...]}`` preserving order."""
    out = {}
    for item in items or []:
        if "=" not in item:
            fail(f"--{kind} expects FIELD=VALUE, got {item!r}")
        field, value = item.split("=", 1)
        if not field:
            fail(f"--{kind} has an empty field name in {item!r}")
        out.setdefault(field, []).append(value)
    return out


def build_data(args) -> dict:
    """Assemble the Envelope ``data`` dict from the generic flags.

    A field given once is a scalar; given more than once it becomes a list,
    which is what makes ``--file images=a.jpg --file images=b.jpg`` work
    without this script knowing what "images" means.
    """
    data = {}

    for field, paths in parse_pairs(args.file, "file").items():
        blobs = []
        for p in paths:
            path = pathlib.Path(p)
            if not path.is_file():
                fail(f"no such file: {path}")
            blobs.append(path.read_bytes())
        data[field] = blobs if len(blobs) > 1 else blobs[0]

    for field, values in parse_pairs(args.text, "text").items():
        data[field] = values if len(values) > 1 else values[0]

    for field, values in parse_pairs(args.number, "number").items():
        nums = []
        for v in values:
            try:
                nums.append(float(v))
            except ValueError:
                fail(f"--number {field}={v!r} is not a number")
        data[field] = nums if len(nums) > 1 else nums[0]

    kinds = [set(parse_pairs(getattr(args, k), k)) for k in ("file", "text", "number")]
    for i in range(len(kinds)):
        for j in range(i + 1, len(kinds)):
            clash = kinds[i] & kinds[j]
            if clash:
                fail(f"field(s) {sorted(clash)} given as more than one kind")
    return data


def load_config(path: str) -> dict:
    p = pathlib.Path(path)
    if not p.is_file():
        fail(f"no such config file: {p}")
    try:
        config = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        fail(f"{p} is not valid JSON: {e}")
    if not isinstance(config, dict):
        fail(f"{p} must hold a JSON object, got {type(config).__name__}")
    return config


# --------------------------------------------------------------------------
# Turning the reply into .mat variables
# --------------------------------------------------------------------------
def matlab_name(field: str, taken: set) -> str:
    """A valid, unique MATLAB variable name for ``field``."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", field)
    if not name or not name[0].isalpha():
        name = "f_" + name
    name = name[:MATLAB_NAME_MAX]
    base, n = name, 2
    while name in taken:
        suffix = f"_{n}"
        name = base[:MATLAB_NAME_MAX - len(suffix)] + suffix
        n += 1
    taken.add(name)
    return name


def to_numeric(value):
    """numpy array / torch tensor / number -> something savemat can store."""
    if isinstance(value, np.ndarray):
        return value
    # torch tensors, without importing torch ourselves
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if isinstance(value, (int, float, bool)):
        return np.array([[value]])
    return None


def list_to_array(items):
    """A list of numbers, or a rectangular list of lists -> a numeric array.

    Returns ``None`` when the list is not uniformly numeric (so the caller can
    fall back to JSON). This is what turns a detection record's ``boxes`` into
    a K x 4 matrix instead of a string MATLAB would have to parse.
    """
    if not items:
        return np.zeros((0, 0))
    try:
        arr = np.asarray(items)
    except ValueError:
        return None                      # ragged
    if arr.dtype.kind in "biufc" and arr.ndim <= 2:
        return arr
    return None


def dict_of_numeric(d, depth=0):
    """A dict -> a dict savemat writes as a struct, or ``None`` if it cannot.

    Values become: numeric arrays, char, cell arrays of char, nested structs,
    or a ``<key>_json`` char as the last resort.
    """
    if depth > 4:
        return None
    out, taken = {}, set()
    for key, value in d.items():
        name = matlab_name(str(key), taken)

        arr = to_numeric(value)
        if arr is not None:
            out[name] = arr
            continue
        if isinstance(value, str):
            out[name] = value
            continue
        if value is None:
            out[name] = np.zeros((0, 0))
            continue
        if isinstance(value, dict):
            nested = dict_of_numeric(value, depth + 1)
            out[name] = nested if nested is not None \
                else json.dumps(value, default=str)
            if nested is None:
                out[name + "_json"] = out.pop(name)
            continue
        if isinstance(value, (list, tuple)):
            items = list(value)
            arr = list_to_array(items)
            if arr is not None:
                out[name] = arr
                continue
            if all(isinstance(x, str) for x in items):
                cell = np.empty((len(items),), dtype=object)
                for i, s in enumerate(items):
                    cell[i] = s
                out[name] = cell
                continue
            try:
                out[name + "_json"] = json.dumps(items, default=str)
            except (TypeError, ValueError):
                return None
            continue
        try:
            out[name + "_json"] = json.dumps(value, default=str)
        except (TypeError, ValueError):
            return None
    return out


def store_field(field, value, payload, field_map, assets_dir, notes):
    """Put one decoded reply field into ``payload`` under a MATLAB-safe name."""
    taken = set(payload)
    name = matlab_name(field, taken)

    arr = to_numeric(value)
    if arr is not None:
        payload[name] = arr
        field_map[field] = name
        return

    if isinstance(value, str):
        payload[name] = value
        field_map[field] = name
        return

    if isinstance(value, (bytes, bytearray, memoryview)):
        assets_dir.mkdir(parents=True, exist_ok=True)
        path = assets_dir / f"{name}.bin"
        path.write_bytes(bytes(value))
        payload[name + "_file"] = str(path.resolve())
        field_map[field] = name + "_file"
        notes.append(f"{field}: 1 binary payload -> {path.name}")
        return

    if isinstance(value, dict):
        as_struct = dict_of_numeric(value)
        if as_struct is not None:
            payload[name] = as_struct
            field_map[field] = name
            return
        payload[name + "_json"] = json.dumps(value, default=str)
        field_map[field] = name + "_json"
        return

    if isinstance(value, (list, tuple)):
        items = list(value)

        # A list of binary payloads (annotated frames, .mat files, ...).
        if items and all(isinstance(x, (bytes, bytearray, memoryview)) for x in items):
            assets_dir.mkdir(parents=True, exist_ok=True)
            written = []
            for i, blob in enumerate(items):
                path = assets_dir / f"{name}_{i:04d}{guess_suffix(blob)}"
                path.write_bytes(bytes(blob))
                written.append(str(path.resolve()))
            payload[name + "_files"] = "\n".join(written)
            payload[name + "_count"] = np.array([[len(written)]])
            field_map[field] = name + "_files"
            notes.append(f"{field}: {len(written)} payloads -> {assets_dir.name}/")
            return

        # A list of dicts (the zstd_pickle boxes) -> cell array of structs.
        if items and all(isinstance(x, dict) for x in items):
            structs = [dict_of_numeric(x) for x in items]
            if all(s is not None for s in structs):
                cell = np.empty((len(structs),), dtype=object)
                for i, s in enumerate(structs):
                    cell[i] = s
                payload[name] = cell
                field_map[field] = name
                return

        # A list of strings -> cell array of char.
        if items and all(isinstance(x, str) for x in items):
            cell = np.empty((len(items),), dtype=object)
            for i, s in enumerate(items):
                cell[i] = s
            payload[name] = cell
            field_map[field] = name
            return

        # Numbers, or anything else JSON can express.
        arr = np.asarray(items)
        if arr.dtype.kind in "biufc":
            payload[name] = arr
            field_map[field] = name
            return
        payload[name + "_json"] = json.dumps(items, default=str)
        field_map[field] = name + "_json"
        return

    # Last resort: describe it rather than drop it silently.
    payload[name + "_json"] = json.dumps(value, default=str)
    field_map[field] = name + "_json"
    notes.append(f"{field}: stored as JSON ({type(value).__name__})")


def guess_suffix(blob: bytes) -> str:
    """File extension from magic bytes - only so the assets are openable."""
    head = bytes(blob[:12])
    if head.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if head.startswith(b"\x89PNG"):
        return ".png"
    if head.startswith(b"PK\x03\x04"):
        return ".zip"
    if head[4:8] == b"ftyp":
        return ".mp4"
    if head.startswith(b"MATLAB"):
        return ".mat"
    if head.startswith(b"glTF"):
        return ".glb"
    return ".bin"


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------
def cmd_probe(args):
    box = Visionist(args.host, timeout=30)
    info = box.info(timeout=args.connect_timeout)
    box.close()
    print(json.dumps({"ok": bool(info.get("reachable")), "host": args.host, **info}))
    return 0 if info.get("reachable") else 1


def cmd_run(args):
    config = load_config(args.config)
    data = build_data(args)

    out = pathlib.Path(args.out)
    assets_dir = pathlib.Path(args.assets_dir) if args.assets_dir \
        else out.with_suffix("")

    box = Visionist(args.host, timeout=args.timeout)
    if not box.info(timeout=args.connect_timeout).get("reachable"):
        box.close()
        fail(f"cannot reach a box at {args.host} within {args.connect_timeout:g}s")

    try:
        res = box.run(data=data, config=config, method=args.method)
    finally:
        box.close()

    reply = res.config if isinstance(res.config, dict) else {}

    # The status lives in the box's own namespaced section. We do not know the
    # section name, so take the only object that carries a "status" - which is
    # the contract, not box knowledge.
    status, section_key = None, None
    for key, value in reply.items():
        if isinstance(value, dict) and "status" in value:
            status, section_key = value, key
            break
    if status is None and "status" in reply:
        status, section_key = reply, None

    payload = {"config_json": json.dumps(reply, default=str)}
    field_map, notes = {}, []
    for field, value in res.fields.items():
        store_field(field, value, payload, field_map, assets_dir, notes)
    payload["field_map_json"] = json.dumps(field_map)

    out.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(out), payload, do_compression=True, oned_as="column")

    report = {
        "ok": (status or {}).get("status") == "done",
        "out": str(out.resolve()),
        "section": section_key,
        "status": (status or {}).get("status"),
        "fields": field_map,
        "encoding": res.encoding,
    }
    if (status or {}).get("status") != "done":
        report["error"] = (status or {}).get("error")
    if notes:
        report["notes"] = notes
    print(json.dumps(report, default=str))

    if not report["ok"]:
        print(f"box answered {report['status']!r}"
              + (f": {report['error']}" if report.get("error") else ""),
              file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Send one Envelope to any box and save the reply as .mat.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    q = sub.add_parser("probe", help="check a box is reachable")
    q.add_argument("--host", required=True)
    q.add_argument("--connect-timeout", type=float, default=10.0)
    q.set_defaults(func=cmd_probe)

    q = sub.add_parser("run", help="send an Envelope, save the reply")
    q.add_argument("--host", required=True, help="box address, host:port")
    q.add_argument("--config", required=True,
                   help="JSON file holding the whole config_json object")
    q.add_argument("--out", required=True, help="output .mat file")
    q.add_argument("--file", action="append", metavar="FIELD=PATH",
                   help="file contents as a data field (repeat for a list)")
    q.add_argument("--text", action="append", metavar="FIELD=STRING",
                   help="literal string as a data field (repeat for a list)")
    q.add_argument("--number", action="append", metavar="FIELD=VALUE",
                   help="number as a data field (repeat for a list)")
    q.add_argument("--assets-dir", default=None,
                   help="where binary payloads are written "
                        "(default: alongside --out, named after it)")
    q.add_argument("--method", default="Process",
                   help="the RPC to call (the contract has only Process)")
    q.add_argument("--timeout", type=float, default=1800.0)
    q.add_argument("--connect-timeout", type=float, default=10.0)
    q.set_defaults(func=cmd_run)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SystemExit:
        raise
    except Exception as e:                                  # noqa: BLE001
        fail(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    raise SystemExit(main())
