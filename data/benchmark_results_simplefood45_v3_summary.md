# Benchmark summary

Generated 2026-09-25 09:24 from `benchmark_results_simplefood45_v3.csv` (252 scored results)

```
RQ2 - volume MAPE vs ground truth (success < 20%):
  deep-learning           204.9%  (n=42)  -> FAIL
  monocular-geometric    1122.8%  (n=126)  -> FAIL
  multi-view              261.3%  (n=84)  -> FAIL

RQ3 - NLP text vs image-only, paired per image (success: text reduces error):
  sizeprior     -0.3pp over 42 pairs, 28/42 better, sign p=0.044  -> PASS

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning           42    204.9    181.9    204.9      0.02    1.9
monocular-geometric    126   1122.8   1122.8   1226.3    -77.66    1.9
multi-view              84    261.3    246.3    261.3     -2.38    8.1
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none                42    204.9    181.9        -    1.9           -           -           -           -
monocular-geometric   checkerboard        42   1454.8   1454.8        -    1.6           -           -           -           -
monocular-geometric   sizeprior_nlp       42    956.6    956.6   1267.2    2.0           -           -           -           -
monocular-geometric   sizeprior_notext    42    956.9    956.9        -    2.1           -           -           -           -
multi-view            checkerboard        42     22.9     -7.0        -    7.7           -           -           -           -
multi-view            sizeprior_notext    42    499.7    499.6        -    8.5           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
deep-learning         mild        27    210.0    191.3
deep-learning         oblique     12     80.6     42.0
monocular-geometric   mild        81    556.2    556.2
monocular-geometric   oblique     36    434.4    434.4
multi-view            mild        54    232.6    218.4
multi-view            oblique     24    133.5    121.4
======================================================

```
