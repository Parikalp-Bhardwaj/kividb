"""Generate src/commands/key_spec_table.rs from Redis's src/commands/*.json.

usage: python -I gen_keyspecs.py <redis src/commands dir> <redis version> <out.rs>
"""
import json
import sys
from pathlib import Path

src, version, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
UPPER = {"RW", "RO", "OW", "RM"}
ORDER = ["RO", "RW", "OW", "RM", "ACCESS", "UPDATE", "INSERT", "DELETE", "NOT_KEY",
         "INCOMPLETE", "VARIABLE_FLAGS"]

entries = {}
for path in sorted(src.glob("*.json")):
    doc = json.loads(path.read_text(encoding="utf-8"))
    for name, cmd in doc.items():
        specs = cmd.get("key_specs")
        if not specs:
            continue
        full = name.upper()
        if "container" in cmd:
            full = f"{cmd['container'].upper()}|{full}"
        no_keys = "NO_MANDATORY_KEYS" in cmd.get("command_flags", [])
        lines = []
        for spec in specs:
            # In the order Redis replies with them (`addReplyFlagsForKeyArgs`).
            # Not INCOMPLETE: for such a spec Redis asks the command's own
            # getkeys function, whose keys never carry it.
            flags = " ".join(
                f if f in UPPER else f.lower()
                for f in sorted(spec.get("flags", []), key=ORDER.index)
                if f != "INCOMPLETE"
            )
            bs = spec["begin_search"]
            if "index" in bs:
                begin = f"Begin::Index({bs['index']['pos']})"
            elif "keyword" in bs:
                kw = bs["keyword"]
                begin = f"Begin::Keyword({json.dumps(kw['keyword'])}, {kw['startfrom']})"
            else:
                begin = "Begin::Unknown"
            fk = spec["find_keys"]
            if "range" in fk:
                r = fk["range"]
                find = f"Find::Range({r['lastkey']}, {r['step']}, {r['limit']})"
            elif "keynum" in fk:
                k = fk["keynum"]
                find = f"Find::KeyNum({k['keynumidx']}, {k['firstkey']}, {k['step']})"
            else:
                find = "Find::Unknown"
            lines.append(f"        Spec {{ flags: {json.dumps(flags)}, begin: {begin}, find: {find} }},")
        entries[full] = (no_keys, lines)

body = [
    f"// Generated from Redis {version} `src/commands/*.json` (the `key_specs` of each",
    "// command) by `gen_keyspecs.py`. Not edited by hand.",
    "",
    "use super::key_specs::{Begin, Find, Spec};",
    "",
    "/// `(command, whether it may have no keys, its key specs)`, by command.",
    "#[rustfmt::skip]",
    "pub(super) const TABLE: &[(&str, bool, &[Spec])] = &[",
]
for name in sorted(entries):
    no_keys, lines = entries[name]
    body.append(f"    ({json.dumps(name)}, {'true' if no_keys else 'false'}, &[")
    body.extend(lines)
    body.append("    ]),")
body.append("];")
out.write_text("\n".join(body) + "\n", encoding="utf-8")
print(len(entries), "commands")
