# Benchmark summary

Generated 2026-08-13 10:12 from `benchmark_results_simplefood45.csv` (210 scored results)

```
RQ1 - volume MAPE vs ground truth (success < 20%):
  deep-learning           206.2%  (n=42)  -> FAIL
  monocular-geometric    1207.0%  (n=84)  -> FAIL
  multi-view              261.4%  (n=84)  -> FAIL

RQ2 - NLP text vs image-only, paired per image (success: text reduces error):
  no paired nlp/notext arms in this results file

======================================================================
approach                 n  V-MAPE%  V-bias%  M-MAPE%  R2(mass)  lat_s
----------------------------------------------------------------------
deep-learning           42    206.2    183.1    206.2      0.01    2.5
monocular-geometric     84   1207.0   1207.0   1207.0    -84.98    5.8
multi-view              84    261.4    246.2    261.4     -2.28   11.5
======================================================================

==============================================================================================================================
approach              arm                  n  V-MAPE%  V-bias%  M-MAPE%  lat_s    overhead        tilt   side_left  side_right
------------------------------------------------------------------------------------------------------------------------------
deep-learning         none                42    206.2    183.1        -    2.5           -           -           -           -
monocular-geometric   checkerboard        42   1456.0   1456.0        -    5.8           -           -           -           -
monocular-geometric   sizeprior_notext    42    957.9    957.9        -    5.9           -           -           -           -
multi-view            checkerboard        42     22.2     -6.4        -   11.3           -           -           -           -
multi-view            sizeprior_notext    42    500.7    498.7        -   11.7           -           -           -           -
==============================================================================================================================

======================================================
approach              tilt         n  V-MAPE%  V-bias%
------------------------------------------------------
deep-learning         mild        27    210.0    191.1
deep-learning         oblique     12     82.5     44.1
monocular-geometric   mild        54    426.0    426.0
monocular-geometric   oblique     24    336.0    336.0
multi-view            mild        54    234.6    223.4
multi-view            oblique     24    128.7    108.7
======================================================

```
