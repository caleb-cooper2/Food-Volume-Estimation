# Benchmark summary

Generated 2026-09-03 11:31 from `benchmark_results_simplefood45_v2.csv` (138 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning           144.6%  (n=23)  -> FAIL
  monocular-geometric     382.7%  (n=69)  -> FAIL
  multi-view              112.0%  (n=46)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  sizeprior   +188.4pp over 22 pairs, 6/22 better, sign p=0.052  -> FAIL

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning           23    144.6    121.2    144.6      0.12    3.1
monocular-geometric     69    382.7    361.4    393.5    -33.52   32.1
multi-view              46    112.0     36.7    112.0     -2.07   14.4
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none                23    144.6    121.2        -    3.1           -           -           -           -
monocular-geometric   checkerboard        23    413.0    413.0        -   29.0           -           -           -           -
monocular-geometric   sizeprior_nlp       23    469.1    465.4    501.6   39.7           -           -           -           -
monocular-geometric   sizeprior_notext    23    265.9    205.7        -   27.5           -           -           -           -
multi-view            checkerboard        23     45.1    -41.0        -   12.8           -           -           -           -
multi-view            sizeprior_notext    23    178.8    114.5        -   16.0           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
deep-learning         mild        14    138.6    115.5
deep-learning         oblique      8     43.6     17.0
monocular-geometric   mild        43    431.2    405.3
monocular-geometric   oblique     24    326.6    311.6
multi-view            mild        28    111.8     22.5
multi-view            oblique     16    113.6     78.7
======================================================

```
