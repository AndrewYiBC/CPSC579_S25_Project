"""
Light‑weight engine wrapper around TripoSR adopted from https://github.com/VAST-AI-Research/TripoSR/blob/main/run.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence, Union, Optional
import logging
import os
import tempfile
import time

import numpy as np
import rembg
import torch
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

_ImageLike = Union[str, Path, Image.Image]


# ---------------------------------------------------------------------------
#                               helper timer
# ---------------------------------------------------------------------------
class _Timer:
    def __init__(self, unit_scale: float = 1000.0):
        self._ts: dict[str, float] = {}
        self._scale = unit_scale

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._ts[name] = time.time()
        logger.info("%s ...", name)

    def end(self, name: str) -> None:
        if name not in self._ts:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - self._ts.pop(name)) * self._scale
        logger.info("%s finished in %.1f ms", name, dt)


# ---------------------------------------------------------------------------
#                           reconstruction engine
# ---------------------------------------------------------------------------
class ReconstructionEngine:
    _DEFAULT_MODEL = "stabilityai/TripoSR"
    _DEFAULT_OUTPUT_DIR = Path("./reconstruction_out")
    _DEFAULT_CHUNK_SIZE = 8192
    _DEFAULT_MC_RES = 256

    # ---------------------------------------------------------------------
    #                               init
    # ---------------------------------------------------------------------
    def __init__(
        self,
        *,
        device: Optional[str] = None,
        pretrained_model_name_or_path: str = _DEFAULT_MODEL,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
        mc_resolution: int = _DEFAULT_MC_RES,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.mc_resolution = mc_resolution
        self._timer = _Timer()

        self._timer.start("Loading TripoSR model")
        self.model = TSR.from_pretrained(
            pretrained_model_name_or_path,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        self.model.renderer.set_chunk_size(self.chunk_size)
        self.model.to(self.device)
        self._timer.end("Loading TripoSR model")

    # ---------------------------------------------------------------------
    #                               public API
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def reconstruct(
        self,
        images: Sequence[_ImageLike],
        *,
        output_dir: Union[str, Path, None] = None,
        remove_bg: bool = True,
        foreground_ratio: float = 0.85,
        model_save_format: str = "obj",
        bake_texture: bool = False,
        texture_resolution: int = 2048,
        render: bool = False,
    ) -> List[Path]:

        out_root = Path(output_dir or self._DEFAULT_OUTPUT_DIR)
        out_root.mkdir(parents=True, exist_ok=True)

        # ------------- load / (optionally) remove background -------------
        self._timer.start("Pre‑processing images")
        session = None if not remove_bg else rembg.new_session()
        proc_imgs: list[Image.Image] = []
        for idx, img_path in enumerate(images):
            if not remove_bg:
                img = np.array(Image.open(img_path).convert("RGB"))
            else:
                img = remove_background(Image.open(img_path), session)
                img = resize_foreground(img, foreground_ratio)
                arr = np.array(img).astype(np.float32) / 255.0
                arr = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
                img = Image.fromarray((arr * 255.0).astype(np.uint8))
                # save the cleaned input for reference
                (out_root / f"{idx}").mkdir(parents=True, exist_ok=True)
                img.save(out_root / f"{idx}/input.png")
            proc_imgs.append(img)
        self._timer.end("Pre‑processing images")

        # ---------------------------- TSR forward ------------------------
        saved_mesh_paths: list[Path] = []
        for idx, image in enumerate(proc_imgs):
            folder = out_root / f"{idx}"
            folder.mkdir(exist_ok=True)

            self._timer.start("Running TripoSR")
            with torch.no_grad():
                scene_codes = self.model([image], device=self.device)
            self._timer.end("Running TripoSR")

            if render:
                self._timer.start("Rendering views")
                frames = self.model.render(scene_codes, n_views=30, return_type="pil")[0]
                for fi, fimg in enumerate(frames):
                    fimg.save(folder / f"render_{fi:03d}.png")
                save_video(frames, folder / "render.mp4", fps=30)
                self._timer.end("Rendering views")

            # ---------------------- mesh extraction ---------------------
            self._timer.start("Extracting mesh")
            meshes = self.model.extract_mesh(scene_codes, not bake_texture, resolution=self.mc_resolution)
            self._timer.end("Extracting mesh")

            mesh_path = folder / f"mesh.{model_save_format}"
            if bake_texture:
                self._timer.start("Baking texture")
                bake_out = bake_texture(
                    meshes[0], self.model, scene_codes[0], texture_resolution
                )
                self._timer.end("Baking texture")

                self._timer.start("Exporting mesh + texture")
                xatlas.export(
                    mesh_path,
                    meshes[0].vertices[bake_out["vmapping"]],
                    bake_out["indices"],
                    bake_out["uvs"],
                    meshes[0].vertex_normals[bake_out["vmapping"]],
                )
                tex_path = folder / "texture.png"
                Image.fromarray(
                    (bake_out["colors"] * 255.0).astype(np.uint8)
                ).transpose(Image.FLIP_TOP_BOTTOM).save(tex_path)
                self._timer.end("Exporting mesh + texture")
            else:
                self._timer.start("Exporting mesh")
                meshes[0].export(mesh_path)
                self._timer.end("Exporting mesh")

            saved_mesh_paths.append(mesh_path)

        return saved_mesh_paths

    @classmethod
    def run(cls, image_paths: Iterable[_ImageLike], init_kwargs=None, **recon_kwargs) -> List[Path]:
        engine = cls(**(init_kwargs or {}))
        return engine.reconstruct(list(image_paths), **recon_kwargs)


# ---------------------------------------------------------------------------
#                                   CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Single‑image 3‑D reconstruction with TripoSR")
    p.add_argument("image", nargs="+", help="Input RGB or RGBA image(s)")
    p.add_argument("--output", dest="output_dir", default="reconstruction_out", help="Output directory")
    p.add_argument("--bake_texture", action="store_true", help="Bake a texture atlas")
    p.add_argument("--render", action="store_true", help="Save a 360° render video")
    args = p.parse_args()

    out_paths = ReconstructionEngine.run(args.image, output_dir=args.output_dir, bake_texture=args.bake_texture, render=args.render)
    print("Saved:")
    for pth in out_paths:
        print("  •", pth)