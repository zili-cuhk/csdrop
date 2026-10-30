# csdrop

Code for the paper

> **Simultaneous Margin-Specific Variable Selection for Competing Risks with Dependent Dropout and Administrative Censoring**

The proposed method fits cause-specific Cox models for the competing events and a Cox model for dropout, links the first-event time and the dropout time by a survival copula, and treats administrative censoring as a separate, independent censoring mechanism. Covariates are selected separately for each margin with the Gaussian seamless-L0 (GSELO) penalty. The method is then run over a sensitivity set of Kendall's τ values, and it reports robust supports, possible supports, and bounds on the cumulative incidence functions (CIFs).

The repository contains the estimator, the code for all simulation studies (Experiments A to H and the larger-dimension cells P01–P36), the lung adenocarcinoma (TCGA-LUAD) analysis, and the scripts that produce the figures and tables of the paper and the Web Appendix. In the paper the proposed method is called copHCR, and the same method under the independence copula is called HDCR.

## Requirements

| Component | Used for | Version used |
|---|---|---|
| MATLAB | proposed method, two-margin comparator, Wald refits | R2025b (no toolboxes required) |
| R | data generation, calibration, lasso comparators | 4.4.3 with `glmnet` (≥ 5.1), `survival` |
| Python 3 | pseudo-true values, TCGA analysis, summaries, figures | `numpy`, `scipy`, `pandas`, `matplotlib` |

The simulation studies were run as LSF job arrays on a computing cluster (`lsf/`). Each job script records the number of tasks and the queue that was used, and the scripts can be adapted to other schedulers.

## Repository layout

```
matlab/csm/        proposed estimator (likelihood, GSELO path, Algorithm S1, sensitivity set, bootstrap)
matlab/csm/test/   reference dataset and values for csm_selftest
matlab/            simulation drivers (run_*_fits.m), comparator adapter, true CIFs, partitions
matlab/app/        MATLAB checks of the TCGA-LUAD analysis
R/                 scenarios, data generation, calibration, lasso and Fine-Gray lasso comparators
python/            Python reference implementation of the estimator (csm.py) and data generation
pseudotrue/        pseudo-true parameters on large samples (support leakage, robust targets)
application/       TCGA-LUAD cohort construction, analysis, bootstrap, and the cohort patient list
analysis/          summaries of the simulation output
tables/            LaTeX tables of the paper and the Web Appendix
figures/           figures of the paper and the Web Appendix
lsf/               cluster job scripts
scenarios.csv      simulation scenarios with the calibrated dropout and administrative constants
fit_plan.csv       which fits are run for each experiment
```

## Quick check of the estimator

In MATLAB, from the repository root,

```matlab
addpath(genpath('matlab'));
csm_selftest
```

This checks the copula derivatives and the analytic scores against finite differences, checks the factorization under the independence copula, and compares the fit on a reference dataset with the values in `matlab/csm/test/ref_values.json`. The reference values were computed with the Python implementation (`python/mkref.py`).

## Simulation studies

Run all commands from the repository root. The experiment names below are those of the paper. Some file names use earlier internal names, which are listed in the last column.

| Paper | Content | Job scripts | Internal name |
|---|---|---|---|
| Experiment A | selection and estimation, n ∈ {400, 600, 1000}, p ∈ {10, 40} | `gen`, `gen2`, `comp`, `comp2`, `prop`, `propA2`, `propliu`, `thr`, `liu` | cells A01–A36 |
| Experiment B | one factor at a time, n = 1000 with p = 100 | `cal`, `gen2`, `comp2`, `propSH`, `liu` | cells B01–B08, S01–S07, H01 ("Study 2") |
| Experiment C | analysis over the Frank grid on the Experiment A datasets | `path1`, `path2` | Exp C |
| Experiment D | misspecified copula family and event-type-dependent dropout | `pathB` | Exp C-B |
| Experiment E | τ-dependent dropout effect | `expD`, `expDm` | Exp D |
| Experiment F | coverage of the Imbens–Manski and envelope intervals | `expF` | Exp F |
| Experiment G | near-tie grid {0.4, 0.45, 0.5}, Theorem 4(v) | `expG` | Exp G |
| Experiment H | grid refinement 0:0.025:0.5, Theorem 5 | `expH` | Exp H |
| Larger dimension | n ∈ {400, 600, 1000}, p ∈ {100, 200}; copHCR, HDCR, lasso, oracle | `genP`, `compP`, `propP`, `liuP` | cells P01–P36 |

Each experiment has a smoke-test script (`*test.lsf`; `largeptest` and `expGHtest` for the newest ones). The order of a full run is

1. `bsub < lsf/cal.lsf` calibrates the dropout multiplier and the administrative time of the Experiment B cells and updates `scenarios.csv`. The values used in the paper are already in `scenarios.csv`, so this step can be skipped.
2. `bsub < lsf/gen.lsf` and `bsub < lsf/gen2.lsf` write the datasets to `data/<cell>/rep_<r>.csv`. Every method reads the same files.
3. The comparators (`comp`, `comp2`), the proposed method and the two-margin method (`prop`, `propA2`, `propSH`, `propliu`), and the thresholded unpenalized estimator (`thr`) write to `results/comp/` and `results/prop/`.
4. After step 3, `bsub < lsf/liu.lsf` refits the selected models with Wald standard errors (`results/liu/`).
5. `path1`, `path2`, `pathB`, `expD`, `expDm`, and `expF` write to `results/path/`, `results/pathB/`, `results/expD/`, `results/expDm/`, and `results/expF/`.
6. `expG` and `expH` rerun the Experiment F driver (`matlab/run_expF_fits.m`) with other grids on the Experiment A datasets and write to `results/expG/` and `results/expH/`.
7. The larger-dimension cells are run in the order `genP`, then `compP` and `propP`, then `liuP`. Their datasets take about 20 GB (covariates stored with 8 significant digits). Job arrays are limited to 1000 tasks, so `expH` (1600 tasks) is submitted twice with an offset:

   ```bash
   for o in 0 800; do OFFSET=$o bsub < lsf/expH.lsf; done
   ```

The comparators follow their original published procedures, including their own tuning. The cause-specific lasso uses `glmnet` with BIC on unpenalized refits or 10-fold cross-validation. The two-margin method of Liu et al. (2026) is fitted once for each cause with the same MATLAB code (`matlab/fit_prop_adapter.m`).

### Summaries

```bash
python3 analysis/agg_liu.py . results/summary/liu          # Experiment A tables
python3 analysis/agg_study2.py . results/summary/study2    # Experiment B
awk 'FNR==1 && NR!=1 {next} {print}' results/path/*.csv  | gzip > results/path_all.csv.gz
awk 'FNR==1 && NR!=1 {next} {print}' results/pathB/*.csv | gzip > results/pathB_all.csv.gz
awk 'FNR==1 && NR!=1 {next} {print}' results/expD/*.csv  | gzip > results/expD_all.csv.gz
python3 analysis/agg_path.py   results/path_all.csv.gz   results/summary/path    # Experiment C
python3 analysis/agg_pathB.py  results/pathB_all.csv.gz  results/summary/pathB   # Experiment D
python3 analysis/agg_expD.py   results/expD_all.csv.gz   results/pseudotrue/expD_targets.json results/summary/expD   # Experiment E
python3 analysis/agg_expF.py   results/expF results/summary/expF                 # Experiment F
python3 analysis/agg_expF.py   results/expG results/summary/expG                 # Experiment G
python3 analysis/agg_expH.py   results/expH results/summary/expH                 # Experiment H
python3 analysis/agg_largep.py . results/summary/largep                          # cells P01-P36
python3 analysis/agg_largep_gamma.py .                                         # cells P01-P36, EBIC_0.25 vs EBIC_0.5
python3 analysis/agg_largep_ebic.py . prop_gamma                               # after lsf/propPg.lsf: EBIC_1 and the Chen-Chen gamma_E
python3 analysis/agg_sandwich.py results/path_all.csv.gz results/expD_all.csv.gz results/pseudotrue/expD_targets.json results/summary/sandwich   # Corollary 2
```

The pseudo-true values that define the targets of Experiments C to E and the support-leakage figure are computed in `pseudotrue/` from samples of 40,000 observations. Run the scripts from `results/pseudotrue/`, for example

```bash
mkdir -p results/pseudotrue && cd results/pseudotrue
python3 ../../pseudotrue/path.py strong 0.5 40000 10      # Web Figure 1 (support leakage)
python3 ../../pseudotrue/path.py moderate 0.5 40000 10
python3 ../../pseudotrue/firstorder.py strong 0.5 40000 10
python3 ../../pseudotrue/expD_targets.py                   # Experiment E, strong signals
python3 ../../pseudotrue/expEm_targets.py                  # Experiment E, moderate signals
for s in strong moderate; do for m in 1 2 4 8; do for mode in cov nocov; do
  python3 ../../pseudotrue/info_tau.py fit $s 40000 $m $mode info; done; done; done
python3 ../../pseudotrue/info_tau.py table info            # Web Table of the profile curvature (Proposition 1)
```

### Figures and tables

The scripts in `figures/` and `tables/` read the summaries from `results/` (or from the directory in the environment variable `CSDROP_RESULTS`) and write to `output/figures/` and `output/tables/`.

| Script | Output |
|---|---|
| `figures/fig_expA.py` | Figure 1 |
| `figures/fig_expB.py` | Figure 2 |
| `figures/fig_sens.py` | Figure 3 |
| `figures/fig_tcga.py` | Figure 4 |
| `figures/fig_leak_supp.py`, `figures/fig_expE_supp.py` | Web Figures 1 and 2 |
| `tables/build.py`, `tables/extra_tables.py` | Table 1 and the Experiment A and B Web Tables |

## Lung adenocarcinoma application

The patient-level data are public and are not redistributed here. Download

* the patient table `data_clinical_patient.txt` of the cBioPortal study `luad_tcga_pan_can_atlas_2018` (https://www.cbioportal.org), and
* `TCGA.LUAD.sampleMap/LUAD_clinicalMatrix` from the UCSC Xena browser (https://xenabrowser.net), saved as `LUAD_clinicalMatrix.tsv`,

and place both files in `application/data/`. Then

```bash
cd application
python3 make_clinical.py data/data_clinical_patient.txt data/clinical.csv   # 498 patients of data/cohort_patient_ids.txt
python3 export_cohorts.py              # analysis cohorts (433 and 474 patients) for MATLAB
python3 run_orig.py                    # supports, hazard ratios, and CIFs of the ten analyses
python3 run_orig_curve.py              # CIF curves for Figure 4
python3 boot_path.py 0 200             # bootstrap over the sensitivity set
python3 boot_an.py                     # selection frequencies, bounds, Imbens-Manski and envelope intervals (Table 2)
```

Outputs are written to `application/out/`. The same analyses can be run in MATLAB with `matlab/app/tcga_check.m` and `matlab/app/tcga_path_check.m`, and the cluster bootstrap with `lsf/tcgaboot.lsf`, summarized by `lsf/tcgaagg.lsf` (`matlab/agg_tcga_boot.m`, which also reports the intervals with standard errors from the normalized interquartile range of the bootstrap replicates). The expected output is listed in `matlab/app/*_expected_python.txt`. `python3 make_boot_idx.py` writes `application/data/tcga_boot_idx.csv`, the bootstrap resampling indices used in the paper and read by `matlab/run_tcga_boot.m`.

## License

MIT License. See `LICENSE`.
