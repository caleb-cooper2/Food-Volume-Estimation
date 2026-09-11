# Benchmark summary

Generated 2026-09-11 11:38 from `benchmark_results_custom_v2.csv` (1298 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning            55.4%  (n=236)  -> FAIL
  monocular-geometric     137.5%  (n=944)  -> FAIL
  multi-view              106.9%  (n=118)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  utensil     -120.4pp over 236 pairs, 112/236 better, sign p=0.474  -> PASS
  sizeprior     -0.3pp over 236 pairs, 102/236 better, sign p=0.043  -> PASS

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning          236     55.4     -4.6     55.5     -0.24    3.7
monocular-geometric    944    137.5     98.7    133.8    -18.97   11.7
multi-view             118    106.9     80.2    106.4     -6.26   22.4
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none               236     55.4     -4.6        -    3.7        58.8        56.9        53.6        52.3
monocular-geometric   sizeprior_nlp      236     86.2     29.0     74.9   16.0        79.7        83.7        87.2        94.0
monocular-geometric   sizeprior_notext   236     86.5     28.3        -    8.8        80.5        84.4        87.2        93.9
monocular-geometric   utensil_nlp        236    128.5    108.3    125.0   13.5       135.1       134.7       119.9       124.5
monocular-geometric   utensil_notext     236    248.9    229.1        -    8.4       154.5       596.6       120.2       124.2
multi-view            sizeprior_notext    59     47.9      5.0        -   22.7           -           -           -           -
multi-view            utensil_notext      59    165.9    155.4        -   22.0           -           -           -           -
==============================================================================================================================

```
