# IDRT Disaster Image Analysis

Reproducible comparison of three satellite-image pairs supplied for the Institute for a Disaster Resilient Texas student research application. The workflow combines data-quality masks, registration diagnostics, classical image metrics, DINOv2 perceptual features, and spatial change-candidate maps.

> Important: a change candidate is not a damage label. Acquisition geometry, illumination, tide, water turbidity, season, sensor resolution, and clouds can all change image appearance. Results are screening evidence for follow-up review.

## Results at a glance

| Pair | Analyzed coverage | SSIM | DINOv2 cosine | Candidate area* |
|---|---:|---:|---:|---:|
| Maxar pre-event vs post-Milton | 81.6% | 0.391 | 0.773 | 32.9% |
| NOAA post-Helene vs post-Milton | 91.3% | 0.690 | 0.819 | 4.8% |
| Sentinel-2 pre-event vs post-Milton | 42.8% | 0.909 | 0.806 | 1.6% |

\* Fraction of analyzed pixels above the empirical high-confidence change-score threshold of 0.65.

The NOAA pair is the cleanest temporal comparison because the images share a platform and are only 11 days apart. The Maxar pair shows broad appearance change across a nearly one-year interval. The Sentinel-2 result applies only to the cloud-screened subset: the RGB heuristic excludes 55.9% of the canvas.

See [REPORT.md](REPORT.md) for methods, interpretation, and limitations. Machine-readable results are in [outputs/metrics](outputs/metrics), and figures are in [outputs/figures](outputs/figures).

## Repository structure

```text
data/raw/                 supplied PNG previews (ignored by Git)
src/idrt_analysis/        analysis implementation
scripts/run_analysis.py   convenient pipeline entry point
scripts/build_report.py   PDF report builder
tests/                    focused unit tests
outputs/figures/          generated comparison figures
outputs/metrics/          CSV and JSON results
application/              email draft and CV tailoring checklist
```

## Reproduce the analysis

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python scripts/run_analysis.py --data-dir data/raw --output-dir outputs
python -m pytest -q
python scripts/build_report.py --output IDRT_disaster_image_analysis_report.pdf
```

The first analysis run downloads the official `dinov2_vits14` checkpoint through PyTorch Hub. DINOv2 runs on CPU by default for predictable behavior on shared machines.

## Method summary

1. Pair images by sensor and date.
2. Remove preview no-data pixels; apply a conservative RGB cloud/cloud-shadow screen to Sentinel-2.
3. Confirm residual translation with phase correlation on Laplacian edge maps. All observed residuals are below one pixel, so no additional warp is applied.
4. Reduce global radiometric bias using per-channel median/IQR matching.
5. Calculate MAE, RMSE, PSNR, masked SSIM, gradient correlation, and 6 x 6 tile-level DINOv2 cosine similarity.
6. Fuse normalized Lab color difference (55%) and SSIM dissimilarity (45%) into a continuous change score. Morphologically cleaned pixels above 0.65 are displayed as high-confidence candidates.

## Data and sharing policy

Raw imagery is deliberately excluded from version control and is not redistributed. Generated figures should be reviewed against the imagery providers' terms before changing this repository from private to public.
