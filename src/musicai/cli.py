"""Kommandozeile: python -m musicai <befehl>"""
from __future__ import annotations

import argparse
import json
import sys
import time

from .app import App
from .web import serve


def wait(app: App, job_id: int) -> str:
    shown = 0
    while True:
        job = app.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        lines = (job["log"] or "").splitlines()
        for line in lines[shown:]: print(line, flush=True)
        shown = len(lines)
        if job["status"] in ("done", "error", "canceled", "interrupted"):
            print(f"-> {job['status']}"); return job["status"]
        time.sleep(1.5)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="musicai", description="MusicAI - Stil lernen und eigene Musik erzeugen")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve", help="WebUI starten"); p.add_argument("--host"); p.add_argument("--port", type=int)
    sub.add_parser("doctor", help="Umgebung pruefen")
    sub.add_parser("version", help="Version und Versionsverlauf anzeigen")
    sub.add_parser("creators", help="Creators auflisten")
    p = sub.add_parser("add-creator", help="Creator anlegen (ohne Ordner: input/<name>)"); p.add_argument("name"); p.add_argument("folder", nargs="?", default="")
    p = sub.add_parser("rights", help="Rechte-Bestaetigung setzen (Voraussetzung fuers Training)"); p.add_argument("creator"); p.add_argument("note", help="Grundlage der Rechte"); p.add_argument("--revoke", action="store_true")
    p = sub.add_parser("analyze", help="Titel eines Creators analysieren + Profil bauen"); p.add_argument("creator"); p.add_argument("--force", action="store_true"); p.add_argument("--limit", type=int)
    p = sub.add_parser("train", help="Stil-Adapter (LoRA) trainieren"); p.add_argument("creator")
    p = sub.add_parser("generate", help="Musik erzeugen"); p.add_argument("--style", action="append", default=[], metavar="CREATOR[=GEWICHT]", help="mehrfach angeben zum Mischen")
    p.add_argument("--prompt", default=""); p.add_argument("--seconds", type=float, default=30); p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--seed", type=int); p.add_argument("--no-profile", action="store_true")
    args = parser.parse_args(argv)
    from . import logging_setup
    logging_setup.setup(); print(f"Logs: {logging_setup.LOG_DIR}  (musicai.log, errors.log, analysis.log)", flush=True)
    app = App()

    def find(name: str) -> dict:
        from .pipeline import slugify
        row = app.db.one("SELECT * FROM creators WHERE slug=?", (slugify(name),))
        if not row: raise SystemExit(f"Creator nicht gefunden: {name}")
        return row

    if args.cmd == "serve": serve(app, args.host, args.port)
    elif args.cmd == "doctor":
        for key, value in app.doctor().items(): print(f"{key:48} {value if value not in (None, False, '') else 'FEHLT'}")
    elif args.cmd == "version":
        from .version import VERSION, CHANGELOG_PATH
        print(f"MusicAI {VERSION}")
        print(f"Versionsverlauf: {CHANGELOG_PATH}")
    elif args.cmd == "creators":
        for c in app.creators():
            print(f"{c['slug']:22} {c['tracks']:4} Titel {c['hours']:6.1f} h  analysiert {c['analyzed']:4}  Profil {'ja' if c['has_profile'] else 'nein':4} Rechte {'ja' if c['rights'] else 'nein':4} Adapter {'ja' if c['adapter'] else 'nein'}")
    elif args.cmd == "add-creator":
        c = app.add_creator(args.name, args.folder); print(f"{c['name']}: {c['tracks']} Titel in {c['source_dir']}")
    elif args.cmd == "rights":
        c = app.set_rights(find(args.creator)["id"], not args.revoke, args.note); print(f"{c['name']}: Rechte {'bestaetigt' if c['rights'] else 'zurueckgenommen'}")
    elif args.cmd == "analyze": return 0 if wait(app, app.start_analyze(find(args.creator)["id"], args.force, args.limit)) == "done" else 1
    elif args.cmd == "train": return 0 if wait(app, app.start_train(find(args.creator)["id"])) == "done" else 1
    elif args.cmd == "generate":
        styles = {}
        for item in args.style:
            name, _, weight = item.partition("="); styles[find(name)["slug"]] = float(weight or 1)
        piece = app.create_piece({"styles": styles, "prompt": args.prompt, "seconds": args.seconds, "strength": args.strength, "seed": args.seed, "use_profile": not args.no_profile})
        status = wait(app, piece["job"])
        print(json.dumps({"piece": piece["id"], "folder": str(app.config.tracks / str(piece["id"]))}))
        return 0 if status == "done" else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
