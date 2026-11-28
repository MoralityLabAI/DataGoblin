"""Upload an immutable Stage-100M release or dev panel to Google Drive via the authenticated gdrive CLI.

Creates the folder hierarchy under ``MyDrive/aleph-10b-continuation-v1/<subdir>``, uploads
every file listed in ``MEMBERS.json`` (plus ``MEMBERS.json`` strictly last), and writes a
reproducible Drive ID + web URL + SHA-256 ledger JSON. Never touches existing checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

GDRIVE_BIN = shutil.which("gdrive") or "gdrive"
RE_DRIVE_ID = re.compile(r"\(ID:\s*([A-Za-z0-9_-]+)\)")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_gdrive(args: list[str], *, expect_json: bool = True, retries: int = 4) -> Any:
    cmd = [GDRIVE_BIN, *args]
    if expect_json:
        cmd.append("--json")
    last_err = ""
    for attempt in range(retries):
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            text = res.stdout.strip()
            if expect_json and text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    pass
            m = RE_DRIVE_ID.search(text + "\n" + res.stderr)
            if m:
                return {"id": m.group(1), "raw": text}
            return {"raw": text}
        last_err = res.stderr.strip() or res.stdout.strip()
        if attempt + 1 < retries:
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"gdrive {' '.join(args)} failed after {retries} attempts: {last_err}")


def find_or_create_folder(name: str, parent_id: str | None = None) -> str:
    ls_args = ["readonly", "ls"]
    if parent_id:
        ls_args.append(parent_id)
    listing = run_gdrive(ls_args, expect_json=True)
    items = listing if isinstance(listing, list) else listing.get("files") or listing.get("items") or []
    for it in items:
        if it.get("name") == name and "folder" in str(it.get("mimeType", "")):
            return str(it["id"])
    mk_args = ["mutate", "mkdir", name]
    if parent_id:
        mk_args.extend(["--parent", parent_id])
    created = run_gdrive(mk_args, expect_json=True)
    if "id" in created:
        return str(created["id"])
    # Fallback lookup by listing parent
    listing2 = run_gdrive(ls_args, expect_json=True)
    items2 = listing2 if isinstance(listing2, list) else listing2.get("files") or listing2.get("items") or []
    for it in items2:
        if it.get("name") == name and "folder" in str(it.get("mimeType", "")):
            return str(it["id"])
    raise RuntimeError(f"failed to resolve created folder ID for {name}: {created}")


def upload_one_file(
    local_path: Path, parent_id: str, existing: dict[str, dict[str, Any]], expected_sha: str
) -> dict[str, Any]:
    name = local_path.name
    sz = local_path.stat().st_size
    if name in existing and int(existing[name].get("size") or -1) == sz:
        fid = str(existing[name]["id"])
        return {
            "name": name,
            "file_id": fid,
            "url": f"https://drive.google.com/file/d/{fid}/view",
            "bytes": sz,
            "sha256": expected_sha,
            "status": "already_present_size_matched",
        }
    res = run_gdrive(["mutate", "upload", str(local_path), "--parent", parent_id], expect_json=True)
    fid = str(res.get("id") or "")
    if not fid:
        # Resolve by listing parent folder
        updated = list_folder_map(parent_id)
        if name in updated:
            fid = str(updated[name]["id"])
    if not fid:
        raise RuntimeError(f"failed to resolve Drive file ID after upload of {name}: {res}")
    return {
        "name": name,
        "file_id": fid,
        "url": f"https://drive.google.com/file/d/{fid}/view",
        "bytes": sz,
        "sha256": expected_sha,
        "status": "uploaded",
    }


def list_folder_map(folder_id: str) -> dict[str, dict[str, Any]]:
    listing = run_gdrive(["readonly", "ls", folder_id])
    items = listing if isinstance(listing, list) else listing.get("files") or listing.get("items") or []
    return {str(it["name"]): it for it in items if "folder" not in str(it.get("mimeType", ""))}


def stage_tree(
    local_root: Path,
    root_folder_name: str,
    sub_folder_name: str,
    *,
    root_folder_id: str | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    t0 = time.monotonic()
    root_dir = local_root.resolve()
    members_path = root_dir / "MEMBERS.json"
    members = json.loads(members_path.read_text(encoding="utf-8"))

    top_id = root_folder_id or find_or_create_folder(root_folder_name, parent_id=None)
    sub_id = find_or_create_folder(sub_folder_name, parent_id=top_id)

    # Pre-create every subdirectory and cache existing file maps
    dir_ids: dict[str, str] = {"": sub_id}
    rel_dirs = sorted({Path(entry["path"]).parent.as_posix() for entry in members["files"]} - {".", ""})
    for rel_dir in rel_dirs:
        cur_parent = sub_id
        cur_key = ""
        for part in rel_dir.split("/"):
            cur_key = f"{cur_key}/{part}" if cur_key else part
            if cur_key not in dir_ids:
                dir_ids[cur_key] = find_or_create_folder(part, parent_id=cur_parent)
            cur_parent = dir_ids[cur_key]

    existing_by_dir = {k: list_folder_map(fid) for k, fid in dir_ids.items()}

    uploaded: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        futs = {}
        for entry in members["files"]:
            rel = entry["path"]
            rel_parent = Path(rel).parent.as_posix()
            if rel_parent == ".":
                rel_parent = ""
            lp = root_dir / rel
            fut = pool.submit(
                upload_one_file,
                lp,
                dir_ids[rel_parent],
                existing_by_dir[rel_parent],
                entry["sha256"],
            )
            futs[fut] = rel
        for fut in as_completed(futs):
            rel = futs[fut]
            rec = fut.result()
            rec["path"] = rel
            uploaded.append(rec)
            print(f"[drive] {rec['status']}: {rel} ({rec['bytes'] / 1e6:.2f} MB) id={rec['file_id']}", flush=True)

    # Publish MEMBERS.json strictly last
    members_sha = sha256_file(members_path)
    m_rec = upload_one_file(members_path, sub_id, existing_by_dir[""], members_sha)
    m_rec["path"] = "MEMBERS.json"
    uploaded.append(m_rec)
    uploaded.sort(key=lambda r: r["path"])

    return {
        "schema": "aleph.stage100m.drive-ledger.v1",
        "drive_root_folder": {
            "name": root_folder_name,
            "id": top_id,
            "url": f"https://drive.google.com/drive/folders/{top_id}",
        },
        "drive_stage_folder": {
            "name": sub_folder_name,
            "id": sub_id,
            "url": f"https://drive.google.com/drive/folders/{sub_id}",
        },
        "members_ledger_sha256": members_sha,
        "file_count": len(uploaded),
        "total_bytes": sum(r["bytes"] for r in uploaded),
        "wall_seconds": round(time.monotonic() - t0, 2),
        "files": uploaded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("local_root", type=Path)
    parser.add_argument("--root-folder", default="aleph-10b-continuation-v1")
    parser.add_argument("--root-id", default=None)
    parser.add_argument("--sub-folder", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ledger-out", type=Path, required=True)
    args = parser.parse_args()

    ledger = stage_tree(
        args.local_root,
        args.root_folder,
        args.sub_folder,
        root_folder_id=args.root_id,
        workers=args.workers,
    )
    args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
    args.ledger_out.write_bytes(json.dumps(ledger, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
    print(json.dumps({k: v for k, v in ledger.items() if k != "files"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
