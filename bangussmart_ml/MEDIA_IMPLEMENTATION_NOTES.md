# Media implementation: measured phase checks

These checks were run on 17 October 2026. **No real labelled fish images were
available. No real-fish accuracy is claimed.** Synthetic cases are development
unit checks, not an independent real-world test set. Thresholds are configurable
ASSUMPTIONS and have not been selected or calibrated on real fish.

1. **Existing prototype:** 32 baseline tests passed and the three original demo
   cases completed. The final rerun retained their feed outputs: 18.17 kg,
   41.24 kg and 0.00 kg. The existing missed sudden-DO-drop demo remains a miss.
2. **Classical image measurement:** the initial seven synthetic geometry tests
   passed. Four single-fish shapes had the following results; full precision is
   preserved in `data/synthetic_media_checks/report.json`.

   | Drawn fork length cm | Rotation degrees | True pixels/cm | Measured cm | Measured pixels/cm |
   |---|---|---|---|---|
   | 10 | 0 | 24 | 10.03610610631618 | 24.01329733334801 |
   | 15 | 0 | 24 | 15.0333373625732 | 24.01329733334801 |
   | 18 | 15 | 24 | 18.01016668196963 | 24.01329733334801 |
   | 15 | -10 | 30 | 15.025578050527702 | 29.989723960747973 |

   On those four drawings, mean signed error/MAE was 0.02629705034667751 cm,
   sample SD of signed error 0.011640293810419896 cm and RMSE
   0.028163046478148747 cm. Errors ranged from 0.01016668196962911 to
   0.036106106316179165 cm. Missing-reference and blurred-fish cases were
   rejected. A later anisotropically distorted board check was also rejected.
   These numbers describe rasterized shapes and corner detection only.
3. **Weight, age range and feed:** 43 tests passed at this phase. The configured
   published juvenile relation gives 30.571889857715078 g at 15 cm. This is an
   equation calculation, not observed weight accuracy. Inverting the synthetic
   growth model at 110 g / 29 °C returned 63 days, range 49–77 days. Range width
   remains an ASSUMPTION, not a measured confidence interval. A requested 10%
   daily rate was capped at 4%: 110 g × 5,000 fish = 550 kg biomass, giving
   22 kg/day in Good water. Critical water gives zero under the existing policy.
4. **Video and CSV:** the phase suite had 13 media tests. A static three-second
   synthetic clip sampled three frames. Two were accepted, one lacked the board
   and was rejected (rejection fraction 0.33333333333333337). One track retained
   both accepted observations; median length was 15.036047917606599 cm. This is
   not a test of moving-fish identity or field counting. Tests also checked valid
   sensor/growth schemas, updated water/temperature settings, row errors and
   integer-hour validation.
5. **API, page, evaluation and regression:** final suite **53 passed**, with one
   existing Starlette/httpx deprecation warning. Invalid-video tests deliberately
   produce decoder diagnostics on stderr. API checks covered binary photo/video/
   CSV uploads, annotated artifact reads, saved JSON, missing scale, invalid
   options, known-age precedence, feed cap and user-count precedence. JavaScript
   syntax checks passed. A generated annotated silhouette was inspected and its
   endpoints matched the drawn snout and tail fork. No browser automation was
   available, so interactive browser layout/upload behavior was not visually
   verified. Evaluating the header-only label template returned
   `no_real_labelled_images` with all error scores null.

Reproduce geometry/video checks with `python check_synthetic_media.py`. Run the
full suite with `python -m pytest -q`. Evaluation is read-only and never calls the
parameter-fitting function. The optional local Ultralytics adapter was not run
with pretrained weights: none were supplied or downloaded. Default classical
segmentation and missing-model fallback were tested.

## Limits that remain material

- Classical silhouettes are unverified candidates, not a trained bangus species
  detector. Other elongated objects can pass; overlap/bending can corrupt fork
  landmarks. Manual review is necessary.
- Board detection establishes pixel scale only if the entered square size is
  correct and fish/board are coplanar. There is no depth calibration or arbitrary
  ruler detector. Strong board distortion rejects; perspective at a different
  fish depth cannot be detected reliably.
- Length-weight calibration defaults are for 9.2–18.6 cm juvenile fork lengths
  in the cited study, not this farm. No extrapolation is made outside that domain.
- Existing growth curves use synthetic data. Age ranges are assumed sensitivity
  envelopes. A date-derived primary age is a user record, not a visual finding.
- Position/length tracking is not ByteTrack or validated re-identification.
  Re-entry, occlusion and skipped frames can inflate/merge distinct tracks.
- Feeding rules remain assumptions. A visible sample might not represent the
  stock's size distribution; video tracks do not establish total surviving stock.
  No feed brand, pellet size or intervention-effect accuracy is inferred.

Before claiming real accuracy, collect paired photographs, manually measured
fork lengths/weights and independently recorded culture ages; measured scale and
capture conditions; sensor history and batch stock counts; and videos with true
fish identities/counts. Reserve fish/batch/session-disjoint test data before any
fitting. See `DATA_COLLECTION_MEDIA.md` for the procedure and label template.
