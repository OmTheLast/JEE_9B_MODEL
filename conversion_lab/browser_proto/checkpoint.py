"""Save a durable Markdown round log, then atomically refresh CURRENT.md."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[2]
RECOVERY = ROOT / "docs/jee/recovery"


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path, content):
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(note, directory=RECOVERY):
    if not note.strip():
        raise ValueError("Checkpoint note must not be empty")
    rounds = directory / "rounds"
    rounds.mkdir(parents=True, exist_ok=True)
    sync_directory(directory)
    now = datetime.now(timezone.utc)
    name = now.strftime("%Y%m%dT%H%M%S.%fZ-") + uuid.uuid4().hex[:8] + ".md"
    content = (f"<!-- Saved UTC: {now.isoformat()} | Round: {name} -->\n\n"
               + note.strip() + "\n")
    # Commit the round first: it survives even if updating CURRENT is interrupted.
    atomic_write(rounds / name, content)
    atomic_write(directory / "CURRENT.md", content)
    return rounds / name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("note", type=Path, help="Markdown file containing the round state")
    args = parser.parse_args()
    saved = save_checkpoint(args.note.read_text(encoding="utf-8"))
    print(f"Saved {saved}\nUpdated {RECOVERY / 'CURRENT.md'}")


if __name__ == "__main__":
    main()
