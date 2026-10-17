# Cooperative capture and measurement guide

There are no real labelled bangus images in this project. Synthetic silhouettes
test geometry only. They do not establish accuracy on farm fish.

1. Work with cooperative personnel on an approved fish-handling procedure. Use a
   shallow white/light tray or a clear tank, a plain background and even light.
   Photograph one fish at a time for the first labelled evaluation set. Keep fish
   straight, fully visible, separated from other objects and in side view.
2. Print the checkerboard from `/media/reference.svg`. It has **8 by 6 squares**
   and 7 by 5 internal corners. Measure the actual printed square width with a
   ruler. Do not trust printer scaling. Record `checkerboard:1.0cm` (or its actual
   square size) in `reference_object`. Keep the board flat, beside the fish,
   at exactly the same distance from the camera. A ruler alone is not detected by
   this version. The API rejects missing or strongly tilted boards.
3. Keep the camera perpendicular to the measurement plane. The fish's snout and
   the indentation of its forked tail must be visible. Avoid glare, bubbles,
   shadows, bending and overlap. Review the annotated endpoints after upload.
4. Use a ruler to measure **fork length**, from snout to tail fork, in cm, and a
   suitable scale to measure the same fish's body weight in g. Record instrument
   resolution/calibration in notes. Total length is not interchangeable with
   fork length. Write one row per fish/image in `data/template_image_labels.csv`.
5. Record capture date, batch and independently known days since stocking from
   the farm's stocking records. This is not age since hatching. Record known
   hatch date and age at stocking separately in farm records, if available.
6. Record water temperature, pH, dissolved oxygen, observation time, surviving
   batch count and the mean-temperature history. Retain raw sensor readings in
   the existing sensor CSV format. Retain weights by batch/day in growth CSVs.
7. For videos, keep camera and board stationary, show the checkerboard in every
   frame and film short clips (<=180 s). Independently label fish identities and
   entrances/exits/occlusion, plus a true distinct count. Never use repeated video
   frames as independent fish in an accuracy report.
8. Collect multiple sizes, batches, dates and realistic lighting conditions,
   including poor captures and missing references. Include rejection cases in
   evaluation; do not keep only easy images. Do not invent a fixed sample count
   as sufficient for accuracy: plan coverage and sample size with your adviser.
9. Split by **fish/batch/capture session**, not random frames. Reserve held-out
   test sessions before fitting. Keep all images/frames of the same fish in one
   split. Fit length-weight constants using training measurements only. Do not
   adjust segmentation thresholds or age-range assumptions after seeing test
   results. Record the configuration used for each evaluation.

The header-only template deliberately contains no invented real-fish labels.
Use fork-length labels in `measured_length_cm`. `known_age_days` is independently
recorded **days since stocking**. Evaluation reads `reference_object` using the
format `checkerboard:<square width>cm`. Capture mean temperature is an explicit
evaluation argument; evaluate temperature-homogeneous sessions separately or add
an optional `mean_temp_c` column to the label CSV. Add an optional `split` column
(`train`, `validation`, `test`) to guard against accidental test reuse.

Before claiming accuracy, evaluate length error, weight error, culture-age error,
range coverage and rejection rate on held-out real fish. Separately evaluate
video distinct-count/identity errors. Confirm feed tables, rations, pellet sizes,
water rules and intervention outcomes with cooperative personnel/advisers. Feed
output assists farm personnel; it does not replace their judgment.
