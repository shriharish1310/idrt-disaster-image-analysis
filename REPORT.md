# Satellite Image Similarity and Change Screening

**Institute for a Disaster Resilient Texas research application**  
**Study area:** Treasure Island / Tampa Bay, Florida  
**Analysis date:** August 25, 2026

## Executive summary

I compared three supplied satellite-image pairs using a quality-controlled, multi-method workflow. Black preview borders were excluded, the Sentinel-2 RGB previews received a conservative cloud/cloud-shadow screen, and subpixel registration was checked before comparison. I then combined traditional pixel/structural metrics with tile-level DINOv2 perceptual embeddings and generated continuous change maps.

| Pair | Analysis coverage | SSIM | DINOv2 cosine | Candidate area |
|---|---:|---:|---:|---:|
| Maxar pre-event vs post-Milton | 81.6% | 0.391 | 0.773 | 32.9% |
| NOAA post-Helene vs post-Milton | 91.3% | 0.690 | 0.819 | 4.8% |
| Sentinel-2 pre-event vs post-Milton | 42.8% | 0.909 | 0.806 | 1.6% |

The NOAA pair provides the most controlled sequential comparison. It retains moderate-to-high structural and perceptual similarity, while high-confidence changes are concentrated rather than scene-wide. The Maxar comparison has the lowest similarity and broadest candidate area, but its nearly one-year interval and strong radiometric/water-state differences mean the map should not be interpreted as a damage map. After screening clouds, the Sentinel-2 pair is structurally stable across the visible subset; however, only 42.8% of the canvas is analyzed, so absence of detected change elsewhere cannot be inferred.

## Methods

### 1. Quality control and registration

- Pixels near pure black were treated as preview no-data borders.
- For Sentinel-2 only, large bright neutral regions and nearby dark regions were conservatively screened as cloud/cloud shadow. This RGB heuristic is appropriate for preview-level quality control, but an operational analysis should use Sentinel-2 Scene Classification or QA layers.
- Phase correlation on Laplacian edge maps estimated residual shifts of less than one pixel for every pair. Because the previews already share a common canvas, no further warping was applied.

### 2. Similarity metrics

- **MAE, RMSE, and PSNR** quantify remaining photometric error after per-channel median/IQR normalization.
- **SSIM** measures local structural agreement.
- **Gradient correlation** compares edge structure and is less sensitive to uniform brightness shifts.
- **DINOv2 ViT-S/14 cosine similarity** compares learned visual features over a 6 x 6 tile grid. Tiles with less than 55% usable coverage are excluded.

### 3. Change-candidate maps

The continuous score combines 55% normalized CIE Lab color difference and 45% normalized SSIM dissimilarity. Connected regions above 0.65 are shown as high-confidence change candidates. This threshold is empirical and intentionally conservative; it has not been calibrated against labeled damage.

## Pair interpretations

### Maxar pre-event (2023-11-04) vs post-Milton (2024-10-10)

- Valid analyzed coverage: **81.6%**; the missing portion is mainly the post-image no-data strip.
- SSIM is **0.391**, DINOv2 cosine similarity is **0.773**, and high-confidence candidates cover **32.9%** of analyzed pixels.
- The largest differences follow the beach/dune system, nearshore water, canals, and many built-feature edges. The post image visibly contains darker inland water and a substantially different beach/wave state.
- Interpretation: this pair signals widespread appearance change, but it mixes storm effects with season, tide, sun/view geometry, water turbidity, and processing differences across an 11-month interval. Candidate areas should be reviewed at object level and validated against a closer-date pre-event Maxar image before being labeled as damage.

### NOAA post-Helene (2024-09-30) vs post-Milton (2024-10-11)

- Valid analyzed coverage: **91.3%**.
- SSIM is **0.690**, DINOv2 cosine similarity is **0.819**, and high-confidence candidates cover **4.8%** of analyzed pixels.
- Change candidates are concentrated along the Gulf shoreline/beach and appear more sparsely over developed areas. Large water regions also change in tone/texture, consistent with differing surface and turbidity conditions.
- Interpretation: because both images are NOAA products only 11 days apart, this is the strongest pair for isolating incremental change between the two post-storm states. Still, candidates require manual or labeled validation before being attributed specifically to Milton.

### Sentinel-2 pre-event (2024-09-19) vs post-Milton (2024-10-14)

- Nominal valid coverage is **98.7%**, but cloud screening removes **55.9%**, leaving **42.8%** for analysis and 10 usable DINOv2 tiles.
- Over this subset, SSIM is **0.909**, DINOv2 cosine similarity is **0.806**, and high-confidence candidates cover **1.6%** of analyzed pixels.
- Broad urban and coastal structure is stable where visible. Localized candidates occur, but the preview's scale and residual cloud/shadow uncertainty limit building-level interpretation.
- Interpretation: the result supports broad-scale stability only in the cloud-free subset. It does not support a conclusion of no damage across the full scene. A stronger workflow would use the original multispectral bands, Scene Classification Layer, co-registered cloud-free compositing, and water/vegetation indices.

## Conclusions and next steps

The three methods are complementary. SSIM is sensitive to local structure, DINOv2 provides a learned perceptual comparison that is less tied to exact pixel values, and the fused maps indicate where to inspect. The NOAA pair is the most defensible comparison for storm-to-storm change; the Maxar pair offers the spatial detail needed for follow-up but has stronger temporal and radiometric confounding; Sentinel-2 supplies regional context but is cloud-limited.

Recommended next steps are to acquire original georeferenced imagery and metadata, use provider QA/cloud layers, match acquisition dates and viewing conditions, segment buildings/roads/shoreline before comparison, and calibrate candidate scores using manually labeled damaged and unchanged samples. Performance should then be reported with precision-recall, F1, spatial cross-validation, and uncertainty intervals.

## References

1. Oquab, M. et al. (2023). *DINOv2: Learning Robust Visual Features without Supervision*. https://arxiv.org/abs/2304.07193
2. Wang, Z., Bovik, A. C., Sheikh, H. R., and Simoncelli, E. P. (2004). *Image Quality Assessment: From Error Visibility to Structural Similarity*. IEEE Transactions on Image Processing, 13(4), 600-612. https://ece.uwaterloo.ca/~z70wang/publications/ssim.html
3. OpenCV. *phaseCorrelate: Motion Analysis and Object Tracking*. https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html
4. ESA. *Sentinel-2 Level-2A Algorithm Theoretical Basis Document (Sen2Cor)*. https://step.esa.int/thirdparties/sen2cor/2.10.0/docs/S2-PDGS-MPC-L2A-ATBD-V2.10.0.pdf
