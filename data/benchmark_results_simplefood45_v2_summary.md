# Benchmark summary

Generated 2026-09-10 10:15 from `benchmark_results_simplefood45_v2.csv` (252 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning           228.5%  (n=42)  -> FAIL
  monocular-geometric    6262.4%  (n=126)  -> FAIL
  multi-view              197.8%  (n=84)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  sizeprior   +346.9pp over 42 pairs, 14/42 better, sign p=0.044  -> FAIL

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning           42    228.5    207.6    228.6     -0.09    3.4
monocular-geometric    126   6262.4   6249.5   6381.2  -9354.41   33.6
multi-view              84    197.8    130.0    197.8     -1.78   14.1
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none                42    228.5    207.6        -    3.4           -           -           -           -
monocular-geometric   checkerboard        42  17469.4  17469.4        -   29.7           -           -           -           -
monocular-geometric   sizeprior_nlp       42    832.4    830.3   1188.1   42.6           -           -           -           -
monocular-geometric   sizeprior_notext    42    485.4    448.7        -   28.6           -           -           -           -
multi-view            checkerboard        42     45.6    -35.7        -   12.7           -           -           -           -
multi-view            sizeprior_notext    42    350.1    295.8        -   15.5           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
deep-learning         mild        27    232.1    215.1
deep-learning         oblique     12     80.8     45.8
monocular-geometric   mild        81    522.1    506.5
monocular-geometric   oblique     36    437.2    427.3
multi-view            mild        54    187.5    120.1
multi-view            oblique     24     96.7     36.3
======================================================

```
