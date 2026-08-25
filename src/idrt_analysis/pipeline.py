"""End-to-end similarity and change analysis for the supplied IDRT imagery."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.color import rgb2lab
from skimage.filters import sobel
from skimage.metrics import structural_similarity
from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects


@dataclass(frozen=True)
class PairSpec:
    key: str
    label: str
    reference_file: str
    comparison_file: str
    reference_label: str
    comparison_label: str
    cloud_screen: bool = False


PAIRS = (
    PairSpec(
        key="maxar",
        label="Maxar pre-event vs post-Milton",
        reference_file="Maxar_pre_20231104_TreasureIsland_preview.png",
        comparison_file="Maxar_post_20241010_TreasureIsland_preview.png",
        reference_label="Pre-event | 2023-11-04",
        comparison_label="Post-Milton | 2024-10-10",
    ),
    PairSpec(
        key="noaa",
        label="NOAA post-Helene vs post-Milton",
        reference_file="NOAA_postHelene_20240930_TreasureIsland_preview.png",
        comparison_file="NOAA_postMilton_20241011_TreasureIsland_preview.png",
        reference_label="Post-Helene | 2024-09-30",
        comparison_label="Post-Milton | 2024-10-11",
    ),
    PairSpec(
        key="sentinel2",
        label="Sentinel-2 pre-event vs post-Milton",
        reference_file="S2_pre_20240919_preview.png",
        comparison_file="S2_post_20241014_preview.png",
        reference_label="Pre-event | 2024-09-19",
        comparison_label="Post-Milton | 2024-10-14",
        cloud_screen=True,
    ),
)

CHANGE_THRESHOLD = 0.65


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float RGB in [0, 1]."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def valid_data_mask(image: np.ndarray) -> np.ndarray:
    """Exclude black preview borders/no-data pixels."""
    return np.max(image, axis=2) > (8.0 / 255.0)


def conservative_cloud_mask(image: np.ndarray) -> np.ndarray:
    """Flag large bright cloud bodies and nearby dark shadows in RGB previews.

    This is intentionally a quality-control heuristic, not a replacement for a
    Sentinel-2 scene-classification or QA band.
    """
    hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0

    bright_neutral = (value > 0.76) & (saturation < 0.38)
    bright_neutral = remove_small_objects(bright_neutral, min_size=220)
    bright_neutral = binary_closing(bright_neutral, disk(5))
    cloud = binary_dilation(bright_neutral, disk(10))

    cloud_neighborhood = binary_dilation(cloud, disk(55))
    possible_shadow = (value < 0.24) & (saturation < 0.62) & cloud_neighborhood
    possible_shadow = remove_small_objects(possible_shadow, min_size=160)
    return binary_dilation(cloud | possible_shadow, disk(4))


def robust_color_match(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Map source channel medians/IQRs to target to reduce illumination bias."""
    matched = source.copy()
    for channel in range(3):
        source_values = source[..., channel][mask]
        target_values = target[..., channel][mask]
        s25, s50, s75 = np.percentile(source_values, [25, 50, 75])
        t25, t50, t75 = np.percentile(target_values, [25, 50, 75])
        scale = (t75 - t25) / max(s75 - s25, 1e-4)
        matched[..., channel] = (source[..., channel] - s50) * scale + t50
    return np.clip(matched, 0.0, 1.0)


def estimate_residual_shift(reference: np.ndarray, comparison: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """Estimate residual translation with phase correlation on edge images."""
    ref_gray = cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    cmp_gray = cv2.cvtColor((comparison * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    ref_edges = cv2.Laplacian(ref_gray, cv2.CV_32F)
    cmp_edges = cv2.Laplacian(cmp_gray, cv2.CV_32F)
    window = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
    weighted_window = window * mask.astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(ref_edges, cmp_edges, weighted_window)
    return float(dx), float(dy), float(response)


def masked_pearson(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    left_values = left[mask].astype(np.float64)
    right_values = right[mask].astype(np.float64)
    if left_values.size < 2 or left_values.std() < 1e-8 or right_values.std() < 1e-8:
        return float("nan")
    return float(np.corrcoef(left_values, right_values)[0, 1])


def compute_change_products(
    reference: np.ndarray, comparison: np.ndarray, mask: np.ndarray
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Return conventional metrics and a fused spatial change score."""
    normalized_reference = robust_color_match(reference, comparison, mask)
    difference = normalized_reference - comparison
    squared = np.square(difference)
    mae = float(np.mean(np.abs(difference)[mask]))
    rmse = float(np.sqrt(np.mean(squared[mask])))
    psnr = float(20.0 * math.log10(1.0 / max(rmse, 1e-12)))

    ssim_value, ssim_map_rgb = structural_similarity(
        normalized_reference,
        comparison,
        channel_axis=2,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        full=True,
    )
    del ssim_value
    ssim_map = np.mean(ssim_map_rgb, axis=2) if ssim_map_rgb.ndim == 3 else ssim_map_rgb
    safe_mask = cv2.erode(mask.astype(np.uint8), np.ones((11, 11), np.uint8), iterations=1).astype(bool)
    masked_ssim = float(np.mean(ssim_map[safe_mask]))

    ref_gray = cv2.cvtColor(normalized_reference, cv2.COLOR_RGB2GRAY)
    cmp_gray = cv2.cvtColor(comparison, cv2.COLOR_RGB2GRAY)
    gradient_correlation = masked_pearson(sobel(ref_gray), sobel(cmp_gray), safe_mask)

    delta_e = np.linalg.norm(rgb2lab(normalized_reference) - rgb2lab(comparison), axis=2)
    color_component = np.clip(delta_e / 35.0, 0.0, 1.0)
    structure_component = np.clip((1.0 - ssim_map) / 0.60, 0.0, 1.0)
    change_score = 0.55 * color_component + 0.45 * structure_component
    change_score[~safe_mask] = np.nan

    change_binary = np.nan_to_num(change_score, nan=0.0) >= CHANGE_THRESHOLD
    change_binary = remove_small_objects(change_binary, min_size=48)
    change_binary = binary_closing(change_binary, disk(3)) & safe_mask
    change_fraction = float(np.mean(change_binary[safe_mask]))

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "psnr_db": psnr,
        "ssim": masked_ssim,
        "gradient_correlation": gradient_correlation,
        "high_change_fraction": change_fraction,
    }
    return metrics, normalized_reference, change_score, change_binary


class DinoV2Comparator:
    """Tile-level DINOv2 perceptual similarity using the official torch hub model."""

    def __init__(self, model_name: str = "dinov2_vits14") -> None:
        self.model_name = model_name
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        # CPU is the reproducible default: shared machines can report CUDA as
        # available even when the device is occupied or has insufficient memory.
        self.device = torch.device("cpu")
        self.model.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def _embed(self, tiles: list[np.ndarray], batch_size: int = 8) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(tiles), batch_size):
            resized = [
                cv2.resize(tile, (224, 224), interpolation=cv2.INTER_AREA)
                for tile in tiles[start : start + batch_size]
            ]
            batch_np = np.stack(resized)
            batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).to(self.device)
            batch = (batch - self.mean) / self.std
            outputs.append(F.normalize(self.model(batch), dim=1).cpu())
        return torch.cat(outputs, dim=0)

    def compare_tiles(
        self, reference: np.ndarray, comparison: np.ndarray, mask: np.ndarray, grid_size: int = 6
    ) -> tuple[float, np.ndarray, int]:
        height, width = mask.shape
        ref_tiles: list[np.ndarray] = []
        cmp_tiles: list[np.ndarray] = []
        locations: list[tuple[int, int]] = []
        similarity_grid = np.full((grid_size, grid_size), np.nan, dtype=np.float32)

        for row in range(grid_size):
            y0, y1 = round(row * height / grid_size), round((row + 1) * height / grid_size)
            for col in range(grid_size):
                x0, x1 = round(col * width / grid_size), round((col + 1) * width / grid_size)
                tile_mask = mask[y0:y1, x0:x1]
                if float(np.mean(tile_mask)) < 0.55:
                    continue
                ref_tile = reference[y0:y1, x0:x1].copy()
                cmp_tile = comparison[y0:y1, x0:x1].copy()
                ref_tile[~tile_mask] = 0.0
                cmp_tile[~tile_mask] = 0.0
                ref_tiles.append(ref_tile)
                cmp_tiles.append(cmp_tile)
                locations.append((row, col))

        if not ref_tiles:
            return float("nan"), similarity_grid, 0
        ref_features = self._embed(ref_tiles)
        cmp_features = self._embed(cmp_tiles)
        similarities = torch.sum(ref_features * cmp_features, dim=1).numpy()
        for (row, col), similarity in zip(locations, similarities, strict=True):
            similarity_grid[row, col] = similarity
        return float(np.mean(similarities)), similarity_grid, len(similarities)


def top_change_regions(change_score: np.ndarray, grid_size: int = 4) -> list[dict[str, Any]]:
    height, width = change_score.shape
    row_names = ["north", "north-central", "south-central", "south"]
    col_names = ["west", "west-central", "east-central", "east"]
    regions: list[dict[str, Any]] = []
    for row in range(grid_size):
        y0, y1 = round(row * height / grid_size), round((row + 1) * height / grid_size)
        for col in range(grid_size):
            x0, x1 = round(col * width / grid_size), round((col + 1) * width / grid_size)
            values = change_score[y0:y1, x0:x1]
            if np.count_nonzero(np.isfinite(values)) < 0.25 * values.size:
                continue
            regions.append(
                {
                    "region": f"{row_names[row]}-{col_names[col]}",
                    "mean_change_score": float(np.nanmean(values)),
                }
            )
    return sorted(regions, key=lambda item: item["mean_change_score"], reverse=True)[:3]


def save_pair_figure(
    spec: PairSpec,
    reference: np.ndarray,
    comparison: np.ndarray,
    mask: np.ndarray,
    change_score: np.ndarray,
    change_binary: np.ndarray,
    metrics: dict[str, float],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.patch.set_facecolor("#f6f4ef")
    fig.suptitle(spec.label, fontsize=18, fontweight="bold", color="#17324d")

    axes[0, 0].imshow(reference)
    axes[0, 0].set_title(spec.reference_label, loc="left", fontweight="bold")
    axes[0, 1].imshow(comparison)
    axes[0, 1].set_title(spec.comparison_label, loc="left", fontweight="bold")

    heatmap = np.ma.masked_invalid(change_score)
    axes[1, 0].imshow(comparison, alpha=0.30)
    image = axes[1, 0].imshow(heatmap, cmap="magma", vmin=0.0, vmax=1.0, alpha=0.85)
    axes[1, 0].set_title("Fused photometric + structural change score", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.02)
    colorbar.set_label("Change score")

    overlay = comparison.copy()
    red = np.array([0.95, 0.15, 0.10], dtype=np.float32)
    overlay[change_binary] = 0.45 * overlay[change_binary] + 0.55 * red
    overlay[~mask] = 0.0
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title("High-confidence change candidates (red)", loc="left", fontweight="bold")
    axes[1, 1].text(
        0.02,
        0.02,
        f"SSIM {metrics['ssim']:.3f}  |  DINOv2 {metrics['dinov2_cosine']:.3f}\n"
        f"Candidate area {100 * metrics['high_change_fraction']:.1f}% of analyzed pixels",
        transform=axes[1, 1].transAxes,
        fontsize=10,
        color="white",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#17324d", "alpha": 0.88, "edgecolor": "none"},
    )

    for axis in axes.ravel():
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def save_metrics_figure(frame: pd.DataFrame, output_path: Path) -> None:
    labels = ["Maxar", "NOAA", "Sentinel-2"]
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), constrained_layout=True)
    fig.patch.set_facecolor("#f6f4ef")
    colors = ["#1f6f8b", "#e09f3e", "#9e2a2b"]
    panels = (
        ("ssim", "Structural similarity", (0, 1)),
        ("dinov2_cosine", "DINOv2 tile cosine", (0, 1)),
        ("high_change_fraction", "Candidate-area fraction", (0, max(0.35, frame["high_change_fraction"].max() * 1.25))),
    )
    for axis, (column, title, ylim) in zip(axes, panels, strict=True):
        bars = axis.bar(x, frame[column], color=colors, width=0.62)
        axis.set_xticks(x, labels)
        axis.set_ylim(*ylim)
        axis.set_title(title, fontweight="bold", color="#17324d")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    axes[2].yaxis.set_major_formatter(lambda value, _position: f"{100 * value:.0f}%")
    fig.suptitle("Pair-level comparison metrics", fontsize=17, fontweight="bold", color="#17324d")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def analyze_pair(spec: PairSpec, data_dir: Path, output_dir: Path, dino: DinoV2Comparator) -> dict[str, Any]:
    reference = load_rgb(data_dir / spec.reference_file)
    comparison = load_rgb(data_dir / spec.comparison_file)
    if reference.shape != comparison.shape:
        raise ValueError(f"Image dimensions differ for {spec.key}: {reference.shape} vs {comparison.shape}")

    valid_mask = valid_data_mask(reference) & valid_data_mask(comparison)
    cloud_mask = np.zeros(valid_mask.shape, dtype=bool)
    if spec.cloud_screen:
        cloud_mask = conservative_cloud_mask(reference) | conservative_cloud_mask(comparison)
    analysis_mask = valid_mask & ~cloud_mask

    dx, dy, phase_response = estimate_residual_shift(reference, comparison, analysis_mask)
    metrics, normalized_reference, change_score, change_binary = compute_change_products(
        reference, comparison, analysis_mask
    )
    dino_similarity, dino_grid, dino_tile_count = dino.compare_tiles(
        normalized_reference, comparison, analysis_mask
    )
    metrics.update(
        {
            "pair": spec.key,
            "label": spec.label,
            "width_px": reference.shape[1],
            "height_px": reference.shape[0],
            "valid_coverage": float(np.mean(valid_mask)),
            "cloud_screened_fraction": float(np.mean(cloud_mask & valid_mask)),
            "analysis_coverage": float(np.mean(analysis_mask)),
            "residual_shift_x_px": dx,
            "residual_shift_y_px": dy,
            "phase_correlation_response": phase_response,
            "dinov2_cosine": dino_similarity,
            "dinov2_tiles": dino_tile_count,
        }
    )

    figure_path = output_dir / "figures" / f"{spec.key}_comparison.png"
    save_pair_figure(spec, reference, comparison, analysis_mask, change_score, change_binary, metrics, figure_path)
    np.savez_compressed(
        output_dir / "metrics" / f"{spec.key}_change_arrays.npz",
        analysis_mask=analysis_mask,
        cloud_mask=cloud_mask,
        change_score=change_score,
        change_binary=change_binary,
        dino_similarity_grid=dino_grid,
    )
    return {**metrics, "top_change_regions": top_change_regions(change_score)}


def run(data_dir: Path, output_dir: Path) -> pd.DataFrame:
    missing = [spec.reference_file for spec in PAIRS if not (data_dir / spec.reference_file).exists()]
    missing += [spec.comparison_file for spec in PAIRS if not (data_dir / spec.comparison_file).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required image files: {sorted(set(missing))}")

    (output_dir / "metrics").mkdir(parents=True, exist_ok=True)
    dino = DinoV2Comparator()
    results = [analyze_pair(spec, data_dir, output_dir, dino) for spec in PAIRS]
    scalar_rows = [{key: value for key, value in row.items() if key != "top_change_regions"} for row in results]
    frame = pd.DataFrame(scalar_rows)
    frame.to_csv(output_dir / "metrics" / "pair_metrics.csv", index=False)
    with (output_dir / "metrics" / "analysis_results.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": {
                    "pairs": [asdict(pair) for pair in PAIRS],
                    "registration": "Phase-correlation residual translation diagnostic on Laplacian edge maps",
                    "photometric_control": "Per-channel median/IQR matching",
                    "similarity": "Masked SSIM, error metrics, gradient correlation, DINOv2 ViT-S/14 tile cosine",
                    "change_map": (
                        "0.55 normalized Lab delta-E + 0.45 normalized SSIM dissimilarity; "
                        f"high-confidence candidate threshold {CHANGE_THRESHOLD:.2f}"
                    ),
                    "cloud_control": "Conservative RGB preview heuristic; not an official Sentinel-2 QA mask",
                },
                "results": results,
            },
            handle,
            indent=2,
        )
    save_metrics_figure(frame, output_dir / "figures" / "metrics_overview.png")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run(args.data_dir.resolve(), args.output_dir.resolve())
    columns = ["pair", "ssim", "dinov2_cosine", "high_change_fraction", "analysis_coverage"]
    print(results[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
