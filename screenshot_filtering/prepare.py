"""Prepare clip/frame manifests before Round 1 (put screenshots in correct order and map them to clips). MP4 export is optional."""
import argparse
import os

from screenshot_filtering.helpers import DATA_DIR, prepare_participant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--enrol", nargs="+")
    parser.add_argument("--videos", action="store_true", help="Also create missing MP4 files for viewing")
    args = parser.parse_args()
    participants = sorted(p for p in os.listdir(DATA_DIR)
                          if p.isdigit() and os.path.isdir(os.path.join(DATA_DIR, p))) if args.all else args.enrol
    if not participants:
        raise ValueError(f"No participants in {DATA_DIR}")
    for participant in participants:
        media = prepare_participant(os.path.join(DATA_DIR, participant), participant, videos=args.videos)
        print(f"Prepared {participant}: {len(media)} clips, {sum(c['n_frames'] for c in media)} screenshots", flush=True)


if __name__ == "__main__":
    main()
