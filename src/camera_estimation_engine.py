from __future__ import annotations

import multiprocessing
from typing import Optional, Tuple, List

import pycolmap

import sys
from pathlib import Path
_COLMAP_PY_PATH = Path(__file__).resolve().parent / "../colmap/scripts/python"
if _COLMAP_PY_PATH.exists():
    sys.path.append(str(_COLMAP_PY_PATH))
from read_write_model import read_images_binary, write_images_text

# ---------------------------------------------------------------------------
#                       CAMERA ESTIMATION ENGINE CLASS
# ---------------------------------------------------------------------------

class CameraEstimationEngine:
    _SIFT_MAX_FEATS: int = 8_192
    _MATCH_MAX_PAIRS: int = 32_768
    _NUM_THREADS: int = multiprocessing.cpu_count()
    _DEFAULT_WORK_DIR: Path = Path("./output")

    # ---------------------------------------------------------------------
    #                           public API
    # ---------------------------------------------------------------------

    def estimate(
        self,
        image_dir: Path | str,
        work_dir: Path | str | None = None,
    ) -> Tuple[Path, Path, List[pycolmap.Reconstruction]]:
        image_dir = Path(image_dir)
        work_dir = Path(work_dir) if work_dir else self._DEFAULT_WORK_DIR
        work_dir.mkdir(parents=True, exist_ok=True)
        db_path = work_dir / "database.db"

        self._extract_features(db_path, image_dir)
        self._match_features(db_path)
        models = self._incremental_mapping(db_path, image_dir, work_dir)
        self._export_images_text(work_dir)

        return work_dir, db_path, models

    # ----------------------- convenience wrapper ---------------------------

    @classmethod
    def run(
        cls,
        image_dir: Path | str,
        work_dir: Path | str | None = None,
        *,
        device: Optional[pycolmap.Device] = None,
    ):
        return cls().estimate(image_dir, work_dir)

    # ---------------------------------------------------------------------
    #                          internal helpers
    # ---------------------------------------------------------------------

    def _extract_features(self, db_path: Path, image_dir: Path) -> None:
        sift_opt = pycolmap.SiftExtractionOptions()
        sift_opt.max_num_features = self._SIFT_MAX_FEATS
        sift_opt.num_threads = self._NUM_THREADS
        pycolmap.extract_features(db_path, image_dir, sift_options=sift_opt)

    def _match_features(self, db_path: Path) -> None:
        match_opt = pycolmap.SiftMatchingOptions()
        match_opt.max_num_matches = self._MATCH_MAX_PAIRS
        match_opt.num_threads = self._NUM_THREADS
        match_opt.guided_matching = True
        pycolmap.match_exhaustive(db_path, sift_options=match_opt)

    @staticmethod
    def _incremental_mapping(
        db_path: Path, image_dir: Path, work_dir: Path
    ) -> List[pycolmap.Reconstruction]:
        return pycolmap.incremental_mapping(db_path, image_dir, work_dir)

    @staticmethod
    def _export_images_text(work_dir: Path) -> None:
        images = read_images_binary(work_dir / "0" / "images.bin")
        write_images_text(images, work_dir / "images.txt")

# ---------------------------------------------------------------------------
#                                 CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate camera poses from an image folder using COLMAP"
    )
    parser.add_argument("image_dir", type=Path, help="Folder containing input images")
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=CameraEstimationEngine._DEFAULT_WORK_DIR,
        help="Output directory for COLMAP database & sparse models",
    )
    cli_args = parser.parse_args()

    CameraEstimationEngine.run(cli_args.image_dir, cli_args.work_dir)
