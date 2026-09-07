# Benchmark summary

Generated 2026-09-03 12:00 from `benchmark_results_custom_v2.csv` (38 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning            33.4%  (n=7)  -> FAIL
  monocular-geometric      66.1%  (n=31)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  utensil       -1.4pp over 8 pairs, 8/8 better, sign p=0.008  -> PASS
  sizeprior     -1.0pp over 7 pairs, 5/7 better, sign p=0.453  -> PASS

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning            7     33.4    -33.4     33.5       nan    3.4
monocular-geometric     31     66.1     61.4     52.9       nan   47.1
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none                 7     33.4    -33.4        -    3.4        26.4           -        37.1        43.3
monocular-geometric   sizeprior_nlp        8     62.8     50.7     45.8   61.1       131.9           -        23.2        18.7
monocular-geometric   sizeprior_notext     7     69.4     62.4        -   35.8       133.3           -        23.5        15.5
monocular-geometric   utensil_nlp          8     65.5     65.5     31.9   55.5        90.4           -        46.2        57.3
monocular-geometric   utensil_notext       8     66.9     66.9        -   34.5        91.3           -        47.8        59.1
==============================================================================================================================

```
