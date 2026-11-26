"""Extract RGB or depth ZIP shards to a sibling directory with progress and ETA."""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
import re
import shutil
import sys
from tempfile import NamedTemporaryFile
from threading import Event, Lock
from time import perf_counter
from zipfile import ZipFile


# Point this to the image_jpg or depth_exr directory that contains the ZIPs.
ZIP_DIR = Path(r"D:\UE_Render\UE_Rendered\Seq-01-CityPark\depth_exr")
WORKERS = 8
COPY_BUFFER_BYTES = 1024 * 1024
FILE_PATTERNS = {
    "image_jpg": re.compile(r"image_[0-9]+\.jpe?g", re.IGNORECASE),
    "depth_exr": re.compile(r"depth_[0-9]+\.exr", re.IGNORECASE),
}


def plan_extraction(source_dir, output_dir, kind):
    pattern = f"{kind}_*.zip"
    archives = sorted(source_dir.glob(pattern))
    if not archives:
        raise FileNotFoundError(f"No {pattern} archives in {source_dir}")
    file_pattern = FILE_PATTERNS[kind]

    jobs = []
    seen = set()
    skipped = 0
    for path in archives:
        members = []
        with ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name = member.filename
                if not file_pattern.fullmatch(name):
                    raise ValueError(f"Unexpected file in {path.name}: {name}")
                key = name.casefold()
                if key in seen:
                    raise ValueError(f"Duplicate output name in {path.name}: {name}")
                seen.add(key)

                target = output_dir / name
                if target.is_file() and target.stat().st_size == member.file_size:
                    skipped += 1
                else:
                    members.append(member)
        if members:
            jobs.append((path, members))
    return len(archives), jobs, skipped


def extract_archive(path, members, folder, stop, record):
    with ZipFile(path) as archive:
        for member in members:
            if stop.is_set():
                return
            target = folder / member.filename
            part = None
            try:
                with archive.open(member) as source, NamedTemporaryFile(
                    dir=folder, prefix=f".{target.stem}.", suffix=".part", delete=False
                ) as output:
                    part = Path(output.name)
                    shutil.copyfileobj(source, output, length=COPY_BUFFER_BYTES)
                part.replace(target)
            except Exception as error:
                raise RuntimeError(f"{path.name}: failed to extract {member.filename}") from error
            finally:
                if part is not None and part.exists():
                    part.unlink()
            record(member.file_size)


def main():
    if WORKERS < 1:
        raise ValueError("WORKERS must be at least 1")

    kind = ZIP_DIR.name.casefold()
    if kind not in FILE_PATTERNS:
        raise ValueError("ZIP_DIR must point to an image_jpg or depth_exr directory")
    output_dir = ZIP_DIR.with_name(f"{kind}_unzipped")
    print(f"Scanning ZIP archives in {ZIP_DIR}...", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    archive_count, jobs, skipped = plan_extraction(ZIP_DIR, output_dir, kind)
    total_files = sum(len(members) for _, members in jobs)
    total_bytes = sum(member.file_size for _, members in jobs for member in members)
    print(
        f"{archive_count} ZIPs, {total_files:,} files to extract "
        f"({total_bytes / 1e9:,.2f} GB), {skipped:,} already present",
        flush=True,
    )
    if not jobs:
        return
    space_path = output_dir if output_dir.is_dir() else output_dir.parent
    free_bytes = shutil.disk_usage(space_path).free
    if total_bytes > free_bytes:
        raise OSError(
            f"Need {total_bytes / 1e9:,.2f} GB to extract, "
            f"but only {free_bytes / 1e9:,.2f} GB are free."
        )

    output_dir.mkdir(exist_ok=True)
    worker_count = min(WORKERS, len(jobs))
    print(f"Extracting with {worker_count} worker(s)...", flush=True)
    stop = Event()
    lock = Lock()
    completed_files = completed_bytes = 0

    def record(size):
        nonlocal completed_files, completed_bytes
        with lock:
            completed_files += 1
            completed_bytes += size

    interactive = sys.stdout.isatty()
    start = last_update = perf_counter()
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        pending = {
            pool.submit(extract_archive, archive, members, output_dir, stop, record)
            for archive, members in jobs
        }
        try:
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()

                now = perf_counter()
                if pending and now - last_update < 1:
                    continue
                with lock:
                    files, size = completed_files, completed_bytes
                speed = size / (now - start)
                eta = "--:--:--"
                if speed:
                    seconds = round((total_bytes - size) / speed)
                    hours, remainder = divmod(seconds, 3600)
                    minutes, seconds = divmod(remainder, 60)
                    eta = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                progress = size / total_bytes if total_bytes else files / total_files
                width = int(progress * 16)
                status = (
                    f"[{'#' * width}{'-' * (16 - width)}] {progress:6.1%} "
                    f"{files:,}/{total_files:,} files | {speed / 1e6:,.1f} MB/s | ETA {eta}"
                )
                if interactive:
                    print(f"\r{status:<85}", end="" if pending else "\n", flush=True)
                else:
                    print(status, flush=True)
                last_update = now
        except BaseException:
            stop.set()
            for future in pending:
                future.cancel()
            if interactive:
                print()
            raise

    print(f"Done. Extracted {completed_files:,} files, skipped {skipped:,}.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped. Run the script again to resume.")
        sys.exit(130)
