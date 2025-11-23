import argparse
import subprocess
from pathlib import Path
import sys

def main():
    p = argparse.ArgumentParser(description="Ultralytics YOLO training launcher (Windows friendly)")
    p.add_argument("--data", default=None, help="Path to dataset yaml (default: ./asl.yaml)")
    p.add_argument("--model", default="yolov8n.pt", help="Base model: yolov8n.pt, yolov8s.pt, etc.")
    p.add_argument("--imgsz", type=int, default=416, help="Image size (square). 416 is a fast starting point.")
    p.add_argument("--epochs", type=int, default=20, help="Epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--name", default="asl_yolo_win", help="Run name under runs/detect/")
    p.add_argument("--fraction", type=float, default=None, help="Fraction of data to use (e.g., 0.1 for 10%)")
    p.add_argument("--cache", default="disk", choices=["disk", "ram", "none"], help="Dataloader cache mode")
    p.add_argument("--workers", type=int, default=2, help="Num dataloader workers (2 is safe on Windows)")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint if found")
    p.add_argument("--device", default=None, help="Force device: 'cuda' or 'cpu' (auto if not set)")

    args = p.parse_args()
    here = Path(__file__).resolve().parent

    # Locate data yaml
    data_yaml = Path(args.data) if args.data else (here / "asl.yaml")
    if not data_yaml.exists():
        sys.exit(f"[ERROR] Dataset yaml not found: {data_yaml}")

    # Decide device (Windows: CUDA or CPU)
    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Build ultralytics command
    cmd = [
        "yolo", "detect", "train",
        f"data={data_yaml.as_posix()}",
        f"model={args.model}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"workers={args.workers}",
        f"cache={args.cache}",
        f"name={args.name}",
        f"device={device}",
        "verbose=False",
    ]
    if args.fraction is not None:
        cmd.append(f"fraction={args.fraction}")
    if args.resume:
        cmd.append("resume=True")

    print(">>> Launching:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=False)

if __name__ == "__main__":
    # Windows needs the spawn-safe guard for dataloading workers
    main()
