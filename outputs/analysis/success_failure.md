# Success and Failure Analysis

This analysis compares `robust_superpixel` against the `center_color` baseline.

## Largest Improvements

| Sample | Class | Baseline IoU | Robust IoU | Delta IoU |
| --- | --- | ---: | ---: | ---: |
| sample_024 | horse | 0.221775 | 0.609444 | 0.387669 |
| sample_006 | tvmonitor | 0.316638 | 0.545205 | 0.228567 |
| sample_000 | aeroplane | 0.262385 | 0.433684 | 0.171299 |
| sample_023 | horse | 0.386485 | 0.517178 | 0.130693 |
| sample_008 | horse | 0.343483 | 0.444881 | 0.101398 |

## Hardest Cases by Robust IoU

| Sample | Class | Robust IoU | Robust Dice |
| --- | --- | ---: | ---: |
| sample_017 | train | 0.140940 | 0.247059 |
| sample_015 | bicycle | 0.186586 | 0.314493 |
| sample_026 | diningtable | 0.192583 | 0.322968 |
| sample_002 | boat | 0.269366 | 0.424410 |
| sample_007 | person | 0.395683 | 0.567010 |

## Largest Regressions or Ties

| Sample | Class | Baseline IoU | Robust IoU | Delta IoU |
| --- | --- | ---: | ---: | ---: |
| sample_015 | bicycle | 0.188934 | 0.186586 | -0.002348 |
| sample_021 | bus | 0.643755 | 0.642317 | -0.001438 |
| sample_002 | boat | 0.269366 | 0.269366 | 0.000000 |
| sample_009 | person | 0.825239 | 0.825239 | 0.000000 |
| sample_012 | cow | 0.595955 | 0.595955 | 0.000000 |
