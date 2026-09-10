# Benchmark summary

Generated 2026-09-10 12:07 from `benchmark_results_custom_v2.csv` (163 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning            38.0%  (n=30)  -> FAIL
  monocular-geometric     116.0%  (n=121)  -> FAIL
  multi-view              122.0%  (n=12)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  utensil     -266.0pp over 30 pairs, 17/30 better, sign p=0.585  -> PASS
  sizeprior     -2.2pp over 30 pairs, 10/30 better, sign p=0.099  -> PASS

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning           30     38.0    -38.0     38.0     -3.20    3.9
monocular-geometric    121    116.0     98.9    110.8   -824.28   49.1
multi-view              12    122.0     78.0    121.9    -41.62   25.0
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none                30     38.0    -38.0        -    3.9        32.1        32.9        40.9        47.7
monocular-geometric   sizeprior_nlp       30     39.2      8.7     31.4   59.2        62.4        34.2        27.4        27.1
monocular-geometric   sizeprior_notext    30     41.4      4.7        -   43.1        62.8        44.4        27.4        27.4
monocular-geometric   utensil_nlp         31     58.7     57.6     46.8   51.9        62.1        75.8        39.1        64.9
monocular-geometric   utensil_notext      30    326.4    325.9        -   42.0        62.3      1404.5        39.7        74.3
multi-view            sizeprior_notext     6     75.0     20.3        -   25.4           -           -           -           -
multi-view            utensil_notext       6    169.1    135.7        -   24.6           -           -           -           -
==============================================================================================================================

```
