"""Run the Football Vision CPU MVP or its bundled synthetic demonstration."""
import argparse
from pathlib import Path
import tempfile

import cv2

from football_vision.pipeline import create_demo, run_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, nargs="?")
    parser.add_argument("--plays", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="new result directory")
    parser.add_argument("--source-kind", choices=["real", "synthetic"])
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    try:
        if args.demo:
            if args.video or args.plays or args.source_kind:
                parser.error("--demo cannot be combined with video, --plays or --source-kind")
            with tempfile.TemporaryDirectory(prefix="football-vision-demo-") as directory:
                video, labels = create_demo(Path(directory))
                result = run_workflow(video, labels, args.out, source_kind="synthetic",
                                      max_frames=args.max_frames)
        else:
            if args.video is None or args.plays is None or args.source_kind is None:
                parser.error("video, --plays and --source-kind are required without --demo")
            result = run_workflow(args.video, args.plays, args.out, source_kind=args.source_kind,
                                  max_frames=args.max_frames)
    except (ValueError, TypeError, KeyError, OSError, cv2.error) as exc:
        parser.exit(2, f"Workflow failed: {exc}\n")
    print(f"Processed {result['frames_processed']} frames and {len(result['plays'])} plays. Results: {args.out}")


if __name__ == "__main__":
    main()
