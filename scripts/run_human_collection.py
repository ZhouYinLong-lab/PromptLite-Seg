"""Local-only human prompt collection interface.

Launches a Tkinter GUI that presents images WITHOUT mask/box/point overlays.
Records one point click (point task) and one drag rectangle (box task) per image.

Usage::

    python scripts/run_human_collection.py --participant P001

Raw annotations are stored under ``data/human_annotations/`` (Git-ignored).
No PII is collected — only the researcher-issued participant code, task ID,
image ID, task type, coordinates, and client-side elapsed time.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import tkinter as tk
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_OUTPUT_DIR = Path("data/human_annotations")
TASK_TIMEOUT_MS = 120_000  # 2 minutes per task
PARTICIPANT_CODE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


def _load_task_list(protocol_path: Path, data_dir: Path) -> list[dict[str, Any]]:
    """Build a deterministic task list from the protocol.

    Returns list of task dicts with keys: task_id, image_id, image_path,
    task_type, mask_path (for later validation, never shown to participant).
    """
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    design = protocol["design"]
    seed = design["sample"]["sampling_seed"]

    rng = np.random.default_rng(seed)

    # Collect all valid sample directories
    sample_dirs = sorted(
        p for p in data_dir.iterdir()
        if p.is_dir()
        and (p / "image.jpg").exists()
        and (p / "target_mask.png").exists()
    )

    # Stratified sampling: pick images_per_class per class
    from promptseg.dataset import VOC_CLASSES, load_sample
    from collections import defaultdict

    by_class: dict[str, list[Path]] = defaultdict(list)
    for sd in sample_dirs:
        try:
            sample = load_sample(sd)
            cls_name = sample.prompt.class_name
            by_class[cls_name].append(sd)
        except Exception:
            continue

    images_per_class = design["sample"]["images_per_class"]
    selected: list[Path] = []
    for cls_name in sorted(by_class):
        dirs = sorted(by_class[cls_name])
        if len(dirs) <= images_per_class:
            selected.extend(dirs)
        else:
            indices = rng.choice(len(dirs), size=images_per_class, replace=False)
            selected.extend(dirs[int(i)] for i in sorted(indices))

    # Build task list: one point + one box per image
    tasks = []
    task_idx = 0
    for sd in sorted(selected):
        sample_id = sd.name
        for task_type in ("point", "box"):
            tasks.append({
                "task_id": f"task_{task_idx:04d}",
                "image_id": sample_id,
                "image_path": str(sd / "image.jpg"),
                "mask_path": str(sd / "target_mask.png"),
                "task_type": task_type,
            })
            task_idx += 1

    return tasks


def _randomize_tasks(tasks: list[dict[str, Any]], participant_code: str) -> list[dict[str, Any]]:
    """Return a deterministic participant-specific task order."""
    digest = hashlib.sha256(participant_code.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    order = rng.permutation(len(tasks))
    return [tasks[int(index)] for index in order]


class CollectionInterface:
    """Tkinter-based local collection GUI.

    Shows images without any overlay. Records:
    - Point task: a single click
    - Box task: a drag rectangle (click + drag + release)
    """

    def __init__(
        self,
        tasks: list[dict],
        participant_code: str,
        output_dir: Path,
    ):
        self.tasks = tasks
        self.participant_code = participant_code
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task_index = 0
        self.annotations: list[dict] = []
        self.task_start_time = 0.0

        # Box drawing state
        self.box_start_x = 0
        self.box_start_y = 0
        self.box_rect_id = None
        self.drawing_box = False

        # Point task state
        self.point_marker_id = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"PromptLite-Seg Human Pilot — Participant {self.participant_code}")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")

        # Info bar
        self.info_var = tk.StringVar(value="Loading...")
        info_label = tk.Label(
            self.root, textvariable=self.info_var,
            font=("Segoe UI", 12), bg="#16213e", fg="#e94560",
            pady=8,
        )
        info_label.pack(fill=tk.X)

        # Canvas for image display
        self.canvas = tk.Canvas(
            self.root, bg="#0f3460",
            width=850, height=550,
            highlightthickness=0,
        )
        self.canvas.pack(pady=5)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(
            self.root, textvariable=self.status_var,
            font=("Segoe UI", 10), bg="#1a1a2e", fg="#a0a0b0",
            pady=4,
        )
        status_label.pack(fill=tk.X, side=tk.BOTTOM)

        self._show_current_task()

    def _show_current_task(self) -> None:
        """Display the current task's image."""
        if self.task_index >= len(self.tasks):
            self._finish()
            return

        task = self.tasks[self.task_index]
        self.canvas.delete("all")
        self.point_marker_id = None
        self.box_rect_id = None
        self.drawing_box = False

        # Load and display image
        img = Image.open(task["image_path"])
        # Resize to fit canvas while maintaining aspect ratio
        canvas_w, canvas_h = 850, 550
        img_w, img_h = img.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        self.display_scale = scale
        self.img_offset_x = (canvas_w - new_w) // 2
        self.img_offset_y = (canvas_h - new_h) // 2

        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img_resized)
        self.canvas.create_image(
            self.img_offset_x + new_w // 2,
            self.img_offset_y + new_h // 2,
            image=self.tk_image,
        )

        task_type = task["task_type"]
        task_id = task["task_id"]
        img_id = task["image_id"]
        progress = f"Task {self.task_index + 1}/{len(self.tasks)}"
        self.info_var.set(f"{progress} — {task_type.upper()} — Image: {img_id} "
                          f"— Participant: {self.participant_code}")
        self.status_var.set(
            f"{'Click on the target object' if task_type == 'point' else 'Click and drag to draw a box around the target object'}"
        )
        self.task_start_time = time.perf_counter()

    def _canvas_to_image(self, cx: int, cy: int) -> tuple[int, int]:
        """Convert canvas coordinates to original image coordinates."""
        ix = round((cx - self.img_offset_x) / self.display_scale)
        iy = round((cy - self.img_offset_y) / self.display_scale)
        return (ix, iy)

    def _on_mouse_down(self, event: tk.Event) -> None:
        task = self.tasks[self.task_index]
        if task["task_type"] == "point":
            self._record_point(event.x, event.y)
        else:
            # Start box drawing
            self.drawing_box = True
            self.box_start_x = event.x
            self.box_start_y = event.y
            if self.box_rect_id:
                self.canvas.delete(self.box_rect_id)
            self.box_rect_id = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="#e94560", width=2, dash=(4, 2),
            )

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self.drawing_box and self.box_rect_id:
            self.canvas.coords(
                self.box_rect_id,
                self.box_start_x, self.box_start_y,
                event.x, event.y,
            )

    def _on_mouse_up(self, event: tk.Event) -> None:
        if self.drawing_box:
            self.drawing_box = False
            self._record_box(event.x, event.y)

    def _record_point(self, cx: int, cy: int) -> None:
        elapsed_ms = (time.perf_counter() - self.task_start_time) * 1000
        if elapsed_ms > TASK_TIMEOUT_MS:
            self.status_var.set("TIMEOUT — Task skipped")
            self._store_annotation(timeout=True)
            return

        ix, iy = self._canvas_to_image(cx, cy)
        task = self.tasks[self.task_index]

        # Draw confirmation mark
        self.point_marker_id = self.canvas.create_oval(
            cx - 5, cy - 5, cx + 5, cy + 5,
            outline="#00ff88", width=2,
        )
        self.status_var.set(f"Point recorded: ({ix}, {iy}) → advancing in 500ms...")
        self.root.after(500, self._store_annotation)

        self._pending_annotation = {
            "participant_code": self.participant_code,
            "task_id": task["task_id"],
            "image_id": task["image_id"],
            "task_type": "point",
            "point_x": ix,
            "point_y": iy,
            "box_x0": "",
            "box_y0": "",
            "box_x1": "",
            "box_y1": "",
            "elapsed_time_ms": round(elapsed_ms, 1),
            "timeout": False,
        }

    def _record_box(self, ex: int, ey: int) -> None:
        elapsed_ms = (time.perf_counter() - self.task_start_time) * 1000
        if elapsed_ms > TASK_TIMEOUT_MS:
            self.status_var.set("TIMEOUT — Task skipped")
            self._store_annotation(timeout=True)
            return

        # Convert to image coordinates
        ix0, iy0 = self._canvas_to_image(self.box_start_x, self.box_start_y)
        ix1, iy1 = self._canvas_to_image(ex, ey)

        # Ensure valid box (x0 < x1, y0 < y1)
        x0, x1 = min(ix0, ix1), max(ix0, ix1)
        y0, y1 = min(iy0, iy1), max(iy0, iy1)

        task = self.tasks[self.task_index]
        self.status_var.set(f"Box recorded: ({x0}, {y0}) → ({x1}, {y1}) → advancing in 500ms...")
        self.root.after(500, self._store_annotation)

        self._pending_annotation = {
            "participant_code": self.participant_code,
            "task_id": task["task_id"],
            "image_id": task["image_id"],
            "task_type": "box",
            "point_x": "",
            "point_y": "",
            "box_x0": x0,
            "box_y0": y0,
            "box_x1": x1,
            "box_y1": y1,
            "elapsed_time_ms": round(elapsed_ms, 1),
            "timeout": False,
        }

    def _store_annotation(self, timeout: bool = False) -> None:
        task = self.tasks[self.task_index]
        if timeout:
            self.annotations.append({
                "participant_code": self.participant_code,
                "task_id": task["task_id"],
                "image_id": task["image_id"],
                "task_type": task["task_type"],
                "point_x": "",
                "point_y": "",
                "box_x0": "",
                "box_y0": "",
                "box_x1": "",
                "box_y1": "",
                "elapsed_time_ms": "",
                "timeout": True,
            })
        elif hasattr(self, "_pending_annotation"):
            self.annotations.append(self._pending_annotation)
            del self._pending_annotation

        self.task_index += 1
        self.root.after(200, self._show_current_task)

    def _finish(self) -> None:
        """Save all annotations and close."""
        # Save CSV
        csv_path = self.output_dir / f"annotations_{self.participant_code}.csv"
        fieldnames = [
            "participant_code", "task_id", "image_id", "task_type",
            "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
            "elapsed_time_ms", "timeout",
        ]
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.annotations)

        self.info_var.set("COLLECTION COMPLETE — Thank you!")
        self.status_var.set(f"Annotations saved to {csv_path}")
        self.canvas.delete("all")
        self.canvas.create_text(
            425, 275,
            text=f"Complete!\n{len(self.annotations)} tasks recorded.\n\nYou may close this window.",
            font=("Segoe UI", 18), fill="#00ff88", justify=tk.CENTER,
        )
        print(f"Annotations saved to {csv_path}")
        print(f"  Tasks recorded: {len(self.annotations)}")

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Synthetic/demo mode (no human needed)
# ---------------------------------------------------------------------------

def run_synthetic_demo(tasks: list[dict], participant_code: str, output_dir: Path) -> None:
    """Generate synthetic annotations for testing.

    All rows are marked ``is_synthetic: true`` and must never be presented
    as human results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    rows = []
    for task in tasks:
        img = Image.open(task["image_path"])
        w, h = img.size

        if task["task_type"] == "point":
            px = int(rng.integers(w // 4, 3 * w // 4))
            py = int(rng.integers(h // 4, 3 * h // 4))
            rows.append({
                "participant_code": participant_code,
                "task_id": task["task_id"],
                "image_id": task["image_id"],
                "task_type": "point",
                "point_x": px,
                "point_y": py,
                "box_x0": "",
                "box_y0": "",
                "box_x1": "",
                "box_y1": "",
                "elapsed_time_ms": round(rng.uniform(500, 3000), 1),
                "timeout": False,
                "is_synthetic": True,
            })
        else:
            bx0 = int(rng.integers(w // 10, w // 3))
            by0 = int(rng.integers(h // 10, h // 3))
            bx1 = int(rng.integers(2 * w // 3, 9 * w // 10))
            by1 = int(rng.integers(2 * h // 3, 9 * h // 10))
            rows.append({
                "participant_code": participant_code,
                "task_id": task["task_id"],
                "image_id": task["image_id"],
                "task_type": "box",
                "point_x": "",
                "point_y": "",
                "box_x0": bx0,
                "box_y0": by0,
                "box_x1": bx1,
                "box_y1": by1,
                "elapsed_time_ms": round(rng.uniform(800, 5000), 1),
                "timeout": False,
                "is_synthetic": True,
            })

    csv_path = output_dir / f"annotations_{participant_code}.csv"
    fieldnames = [
        "participant_code", "task_id", "image_id", "task_type",
        "point_x", "point_y", "box_x0", "box_y0", "box_x1", "box_y1",
        "elapsed_time_ms", "timeout", "is_synthetic",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Synthetic annotations saved to {csv_path}")
    print(f"  Tasks: {len(rows)}")
    print(f"  WARNING: These are synthetic/demo data — NOT human results.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local-only human prompt collection interface"
    )
    parser.add_argument(
        "--participant", type=str, required=True,
        help="Researcher-issued participant code (e.g., P001)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/voc_validation"),
        help="Directory containing sample subdirectories with images",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("protocol/human_pilot_protocol.json"),
        help="Human pilot protocol file",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for raw annotations (Git-ignored)",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Generate synthetic/demo annotations instead of launching GUI",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Limit number of images (for testing)",
    )
    args = parser.parse_args()

    # Validate participant code (no PII-like patterns)
    if not PARTICIPANT_CODE_PATTERN.fullmatch(args.participant):
        parser.error("Participant code must contain only letters, digits, '_' or '-' (1-20 chars)")

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if not args.synthetic:
        ethics = protocol.get("ethics_checkpoint", {})
        provenance = protocol.get("provenance", {})
        if ethics.get("status") != "APPROVED":
            raise SystemExit(
                "Human collection is locked: instructor/ethics approval is not recorded."
            )
        if not provenance.get("frozen_before_collection"):
            raise SystemExit("Human collection is locked: freeze the protocol before recruitment.")

    # Build task list
    tasks = _load_task_list(args.protocol, args.data_dir)
    if args.max_images:
        # Limit tasks proportionally
        max_tasks = args.max_images * 2  # point + box per image
        tasks = tasks[:max_tasks]
    tasks = _randomize_tasks(tasks, args.participant)

    if not tasks:
        print("ERROR: No valid sample images found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(tasks)} tasks from {len(set(t['image_id'] for t in tasks))} images")

    if args.synthetic:
        run_synthetic_demo(tasks, args.participant, args.output_dir)
    else:
        print(f"Starting collection interface for participant {args.participant}")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Output: {args.output_dir}")
        print()
        app = CollectionInterface(tasks, args.participant, args.output_dir)
        app.run()


if __name__ == "__main__":
    main()
