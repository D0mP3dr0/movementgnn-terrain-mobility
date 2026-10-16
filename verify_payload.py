"""Check that the published records carry the same numbers as the records as written, and
that every number quoted in README.md is the number in its record.

Before publication, the free-text notes of the JSON and CSV records were translated into
English, the absolute data-root prefix of the paths they quote was replaced by
${MOVEMENTGNN_DATA_ROOT} and the home folder of a local Python installation by ${USER_HOME},
and the keys of four audit records (t2, t4, t4b, t12) were translated into English; the
keys that the code writes or reads (for example the profile keys a_pe, motorizada,
mecanizada, blindada) were kept. KEYS_AS_WRITTEN below maps each translated key back.

For each file listed in PAYLOAD_SHA256 this script computes the sha256 of its "payload":
  - JSON: the canonical JSON with each key mapped back to the key as written, the two
    prefixes above turned into fixed tokens, and each string of four or more words replaced
    by the list of the numbers it contains, in order (so a translation must keep them);
    every other value, number, boolean, null and the whole structure enter the digest;
  - CSV: the number of rows and of cells per row, and, for each cell, the list of the
    numbers it contains, in order, after the same prefix tokens.
PAYLOAD_SHA256 lists the digest of each record as written, computed before publication.

The script then checks each entry of README_CLAIMS: the quoted text must appear in
README.md and the value at the given place of the record must be the quoted number.

Usage (from the repository root):  python3 verify_payload.py
"""
import csv
import hashlib
import io
import json
import pathlib
import re
import sys

ROOTS = {"${MOVEMENTGNN_DATA_ROOT}": "<root>", "${USER_HOME}": "<home>"}
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

KEYS_AS_WRITTEN = {
    "results/t12_rebuild_timing.json": {
        "protocol": "protocolo", "inputs": "entradas", "builder_script": "script_construtor",
        "official": "oficial", "execution": "execucao", "total_time_s": "tempo_total_s",
        "total_time_min": "tempo_total_min", "peak_rss_gb_sampled_2s": "pico_rss_gb_amostrado_2s",
        "machine": "maquina", "final_log": "log_final", "validation": "validacao",
        "generated_file_gb": "arquivo_gerado_gb", "dem_dem_edges_official": "dem_dem_edges_oficial",
        "dem_nodes_official": "dem_nodes_oficial", "note": "nota", "cleanup": "limpeza"},
    "results/t2_threshold_bands.json": {
        "methodology": "metodologia", "margins": "margens", "profiles": "perfis"},
    "results/t4_audit_results.json": {
        "pure_blocks": "blocos_puros", "pure_blocks_per_partition": "blocos_puros_por_particao",
        "n_blocks_with_assigned_nodes": "n_blocks_com_nos_atribuidos", "purity": "pureza"},
    "results/t4b_block_structure.json": {
        "date": "data", "vertical_boundaries": "fronteiras_verticais",
        "spacings_cells": "espacamentos_celulas", "horizontal_boundaries": "fronteiras_horizontais",
        "first_band_offset_rows": "offset_primeira_banda_linhas", "block_size": "tamanho_bloco",
        "n_blocks": "n_blocos", "blocks_per_partition": "blocos_por_particao", "purity": "pureza"},
}

README_CLAIMS = [
    ("12,067,692-node", "results/t12_rebuild_timing.json", ["official", "dem_nodes"], 12067692),
    ("12,067,692-node", "results/t4_audit_results.json", ["n_dem_nodes"], 12067692),
    ("184.1 s", "results/t12_rebuild_timing.json", ["execution", "total_time_s"], 184.1),
    ("42.36 GB", "results/t12_rebuild_timing.json", ["execution", "peak_rss_gb_sampled_2s"], 42.36),
    ("483 blocks", "results/t4b_block_structure.json", ["n_blocks"], 483),
    ("block purity 1.0", "results/t4b_block_structure.json", ["purity"], 1.0),
    ("8,454,917 / 1,847,936 / 1,764,839", "results/t4_audit_results.json", ["mask_stats", "train_mask", "count"], 8454917),
    ("8,454,917 / 1,847,936 / 1,764,839", "results/t4_audit_results.json", ["mask_stats", "val_mask", "count"], 1847936),
    ("8,454,917 / 1,847,936 / 1,764,839", "results/t4_audit_results.json", ["mask_stats", "test_mask", "count"], 1764839),
    ("271 input channels", "results/metrics_final_run.json", ["config", "in_channels"], 271),
    ("3 GATv2 layers", "results/metrics_final_run.json", ["config", "num_layers"], 3),
    ("4 attention heads", "results/metrics_final_run.json", ["config", "heads"], 4),
    ("52,364 parameters", "results/metrics_final_run.json", ["config", "n_params"], 52364),
    ("batches of 750,000 nodes", "results/metrics_final_run.json", ["config", "batch_size"], 750000),
    ("k = 8 neighbours", "results/metrics_final_run.json", ["config", "k_neighbors"], 8),
]


def root(s):
    for prefix, token in ROOTS.items():
        s = s.replace(prefix, token)
    return s


def is_free_text(s):
    return len(s.split()) >= 4


def _json_norm(o, keymap, root_fn):
    if isinstance(o, dict):
        return {keymap.get(k, k): _json_norm(v, keymap, root_fn) for k, v in o.items()}
    if isinstance(o, list):
        return [_json_norm(v, keymap, root_fn) for v in o]
    if isinstance(o, str):
        s = root_fn(o)
        return {"<text>": NUMBER.findall(s)} if is_free_text(s) else s
    return o


def json_payload(obj, keymap, root_fn=root):
    canon = json.dumps(_json_norm(obj, keymap, root_fn), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def csv_payload(text, root_fn=root):
    delim = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    canon = json.dumps([[NUMBER.findall(root_fn(c)) for c in r] for r in rows], separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def payload_sha256(path, rel):
    text = pathlib.Path(path).read_text(encoding="utf-8-sig")
    if rel.endswith(".csv"):
        return csv_payload(text)
    return json_payload(json.loads(text), KEYS_AS_WRITTEN.get(rel, {}))


def main():
    base = pathlib.Path(__file__).resolve().parent
    expected = {}
    for line in (base / "PAYLOAD_SHA256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    bad = 0
    for rel, digest in sorted(expected.items()):
        ok = payload_sha256(base / rel, rel) == digest
        bad += not ok
        if not ok:
            print(f"MISMATCH {rel}")
    print(f"{len(expected) - bad}/{len(expected)} records: payload identical to the record as written")
    readme = (base / "README.md").read_text(encoding="utf-8")
    bad_claims = 0
    for quoted, rel, keys, value in README_CLAIMS:
        o = json.loads((base / rel).read_text(encoding="utf-8"))
        for k in keys:
            o = o[k]
        ok = quoted in readme and o == value
        bad_claims += not ok
        if not ok:
            print(f"CLAIM FAILS: {quoted!r} vs {rel} {keys} = {o!r}")
    print(f"{len(README_CLAIMS) - bad_claims}/{len(README_CLAIMS)} README numbers found in their records")
    sys.exit(1 if bad or bad_claims else 0)


if __name__ == "__main__":
    main()
