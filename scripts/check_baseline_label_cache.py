#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIEW_ROOTS = {
    "RGB": PROJECT_ROOT / "yolo_views" / "rgb_v1",
    "IR": PROJECT_ROOT / "yolo_views" / "ir_v1",
    "Depth": PROJECT_ROOT / "yolo_views" / "depth8_v1",
}

TARGET_STEMS = {
    "000050",
    "003107",
    "003817",
}


def load_cache(path):

    cache = np.load(
        str(path),
        allow_pickle=True,
    ).item()

    return cache


def main():

    print("=" * 90)
    print("Baseline label-cache inspection")
    print("=" * 90)

    for modality, root in VIEW_ROOTS.items():

        print()
        print("=" * 90)
        print(modality)
        print("=" * 90)

        cache_files = sorted(
            root.rglob("*.cache")
        )

        if not cache_files:
            print("No .cache file found.")
            continue

        for cache_path in cache_files:

            print()
            print("Cache:", cache_path)

            try:
                cache = load_cache(
                    cache_path
                )
            except Exception as exc:
                print(
                    "Cannot load cache:",
                    repr(exc),
                )
                continue

            labels = cache.get(
                "labels",
                []
            )

            print(
                "Cached labels:",
                len(labels),
            )

            found = {
                stem: []
                for stem in TARGET_STEMS
            }

            for item in labels:

                im_file = Path(
                    item.get(
                        "im_file",
                        ""
                    )
                )

                stem = im_file.stem

                if stem not in TARGET_STEMS:
                    continue

                cls = item.get(
                    "cls"
                )

                bboxes = item.get(
                    "bboxes"
                )

                found[
                    stem
                ].append(
                    {
                        "im_file":
                            str(im_file),

                        "cls":
                            cls,

                        "bboxes":
                            bboxes,
                    }
                )

            for stem in sorted(
                TARGET_STEMS
            ):

                print()
                print(
                    f"Target: {stem}"
                )

                entries = found[
                    stem
                ]

                if not entries:

                    print(
                        "  NOT PRESENT in cache"
                    )
                    continue

                print(
                    f"  PRESENT: "
                    f"{len(entries)} entry"
                )

                for entry in entries:

                    print(
                        "  image:",
                        entry[
                            "im_file"
                        ],
                    )

                    print(
                        "  cls:"
                    )

                    print(
                        entry[
                            "cls"
                        ]
                    )

                    print(
                        "  bboxes:"
                    )

                    print(
                        entry[
                            "bboxes"
                        ]
                    )

    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
