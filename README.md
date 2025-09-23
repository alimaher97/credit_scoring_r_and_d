# Customer Segmentation for Credit Risk

End-to-end notebooks for preparing credit data, creating supervised customer segments (decision-tree clusters), and producing score-based risk bands for both banked and unbanked populations. Outputs include ready-to-use datasets, profiles, statistical validation, and exportable artifacts.

## Repository layout

- Data prep
  - [customer_banked_datapreperation.ipynb](customer_banked_datapreperation.ipynb)
  - [customer_unbanked_datapreperation.ipynb](customer_unbanked_datapreperation.ipynb)
- Supervised clustering (Decision Tree)
  - [supervised_clusting_banked_dtree.ipynb](supervised_clusting_banked_dtree.ipynb)
  - [supervised_clusting_unbanked_dtree.ipynb](supervised_clusting_unbanked_dtree.ipynb)
- Risk bands (Logistic Regression)
  - [supervised_clusting_banked_risk_bands.ipynb](supervised_clusting_banked_risk_bands.ipynb)
  - [supervised_clusting_unbanked_risk_bands.ipynb](supervised_clusting_unbanked_risk_bands.ipynb)
- Analysis
  - [analize_result.ipynb](analize_result.ipynb)
- Utilities
  - [utiles.py](utiles.py) (feature prep, profiling, tests, readable rules)
- Inputs and notes
  - [credit_report (1).xlsx](credit_report (1).xlsx)
  - [good-bad_customer_bucket.txt](good-bad_customer_bucket.txt)
  - [notes_for_update_two.txt](notes_for_update_two.txt)
- Outputs (created by notebooks)
  - customer_segmentation_baked_dtree_v1.0/
  - customer_segmantation_unbaked_dtree_v1.0/
  - customer_segmentation_baked_logreg_bands_v1.0/
  - customer_segmantation_unbaked_logreg_bands_v1.0/
  - outputs/

## Workflow

1) Data preparation
- Banked: filter Banked, exclude special programs, validate business rules, deduplicate, cap/impute selective fields, and derive binary target from customer_bucket. Saves:
  - customer_segmentation_baked_dtree_v1.0/data/banked_customer_segmentation_final.csv
  - Documentation and summary report
  See [customer_banked_datapreperation.ipynb](customer_banked_datapreperation.ipynb).

- Unbanked: similar pipeline; additionally keeps Rank 1–2 only. Saves:
  - customer_segmantation_unbaked_dtree_v1.0/data/unbanked_customer_segmentation_final.csv
  - Documentation and summary report
  See [customer_unbanked_datapreperation.ipynb](customer_unbanked_datapreperation.ipynb).

Target definition uses [good-bad_customer_bucket.txt](good-bad_customer_bucket.txt) mapping (good=g, bad=b, remove=r).

2) Supervised clustering (Decision Tree)
- Build CART trees on prepared features; leaves are ranked by bad rate and remapped to ordered clusters. Exports customers with cluster labels, profiles, leaf stats, pairwise significance, tree plots, and readable rules.
  - Banked: [supervised_clusting_banked_dtree.ipynb](supervised_clusting_banked_dtree.ipynb) → customer_segmentation_baked_dtree_v1.0/
  - Unbanked: [supervised_clusting_unbanked_dtree.ipynb](supervised_clusting_unbanked_dtree.ipynb) → customer_segmantation_unbaked_dtree_v1.0/

3) Risk bands (Logistic Regression)
- Train logistic regression, compute PD scores, and bin into statistically distinct bands (adjacent-bin merges via two-proportion z-tests with FDR). Exports customers with bands, profiles, tests, plots, model, encoder, and band edges.
  - Banked: [supervised_clusting_banked_risk_bands.ipynb](supervised_clusting_banked_risk_bands.ipynb) → customer_segmentation_baked_logreg_bands_v1.0/
  - Unbanked: [supervised_clusting_unbanked_risk_bands.ipynb](supervised_clusting_unbanked_risk_bands.ipynb) → customer_segmantation_unbaked_logreg_bands_v1.0/

4) Inspection
- Quick slicing/filtering of exported clusters: [analize_result.ipynb](analize_result.ipynb)

## Utilities (API)

- Feature prep and encoding:
  - [`utiles.prepare_features`](utiles.py)
- Decision tree clustering:
  - [`utiles.fit_tree_for_clusters`](utiles.py)
  - [`utiles.evaluate_leaf_bad_rates`](utiles.py)
  - [`utiles.build_readable_rules`](utiles.py)
- Profiling and stats:
  - [`utiles.compute_cluster_profiles`](utiles.py)
  - [`utiles.proportion_ci_wilson`](utiles.py)
  - [`utiles.chi_square_test_cluster_target`](utiles.py)
  - [`utiles.pairwise_bad_rate_ztests`](utiles.py)

## How to run

- Environment: Python 3.10 (kernel “credit_risk” in notebooks).
- Dependencies: pandas, numpy, scikit-learn, statsmodels, scipy, seaborn, matplotlib, jupyter.
- In VS Code:
  - Open a notebook (e.g., [customer_unbanked_datapreperation.ipynb](customer_unbanked_datapreperation.ipynb)) and run all cells top-to-bottom.
  - Proceed to the corresponding clustering and/or risk-band notebook for that population.
- Inputs must include [credit_report (1).xlsx](credit_report (1).xlsx) in the repo root.

## Outputs (key artifacts)

- Data (per population): customers_with_clusters.csv, cluster_profiles.csv, leaf_validation_stats.csv, tree_rules[_readable].txt, model_decision_tree.pkl, encoder_ordinal.pkl.
- Risk bands (per population): customers_with_risk_bands.csv, band_profiles.csv, band_pairwise_bad_rate_tests.csv, model_logreg.pkl, encoder_ordinal.pkl, band_edges.txt, plots.

## Next steps

See [notes_for_update_two.txt](notes_for_update_two.txt) for pending updates: address-category refresh, retrain with new data, and alternative models beyond
