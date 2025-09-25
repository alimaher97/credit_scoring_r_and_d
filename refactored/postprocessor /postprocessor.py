from scipy import stats
import numpy as np
import pandas as pd

class CreditDataPostprocessor:
    def __init__(self, df: pd.DataFrame = None, all_proba: np.ndarray = None):
        self.df = df
        self.all_proba = all_proba
    
    def __two_prop_pval(self, bad1: int, n1: int, bad2: int, n2: int) -> float:
        # Pooled two-proportion z-test, two-sided
        if n1 == 0 or n2 == 0:
            return 1.0
        p1, p2 = bad1 / n1, bad2 / n2
        p_pool = (bad1 + bad2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
        if se == 0:
            # If both perfectly 0 or 1, treat as no evidence if equal else strong difference
            return 1.0 if p1 == p2 else 0.0
        z = (p1 - p2) / se
        return float(2 * (1 - stats.norm.cdf(abs(z))))
    
    def __compute_bins(self, df: pd.DataFrame, score_col: str, target_col: str, edges: np.ndarray) -> pd.DataFrame:
        cats = pd.cut(df[score_col], bins=edges, right=False, include_lowest=True)
        agg = df.groupby(cats)[target_col].agg(["sum", "count"]).rename(columns={"sum": "bad", "count": "n"})
        out = agg.reset_index()
        out["left"] = out[score_col].apply(lambda iv: iv.left)
        out["right"] = out[score_col].apply(lambda iv: iv.right)
        out["rate"] = out["bad"] / out["n"].where(out["n"] > 0, np.nan)
        return out[["left", "right", "bad", "n", "rate"]]
    
    def __make_significant_pd_bands(
    self,
    df: pd.DataFrame,
    score_col: str = "pd_score",
    target_col: str = "target",
    *,
    init_bins: int = 20,
    max_bands: int = 8,
    min_bin_size: float = 0.05,  # fraction of total rows per band
    alpha: float = 0.05,
) -> np.ndarray:
        """
        Create PD score bands by iteratively merging adjacent bins until:
        - All adjacent bad-rate differences are statistically significant (two-proportion z-test, p < alpha),
        - Each band has at least `min_bin_size` fraction of the portfolio, and
        - The number of bands <= `max_bands`.

        Returns the bin edges array suitable for pd.cut(..., bins=edges, right=False).
        """
        s = df[score_col].astype(float).values
        n_total = len(df)
        if n_total == 0:
            return np.array([0.0, 1.0000001])

        # Initial edges via quantiles (equal-frequency), dropping duplicates
        n_unique = int(pd.Series(s).nunique(dropna=True))
        nb = max(1, min(init_bins, n_unique))
        qs = np.linspace(0, 1, nb + 1)
        edges = np.quantile(s, qs)
        edges = np.unique(edges)

        # Ensure at least a single interval and open-right top edge
        smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
        if edges[0] > smin:
            edges[0] = smin
        if edges[-1] < smax:
            # bump top slightly for right-open binning
            eps = np.nextafter(smax, np.inf) - smax
            edges[-1] = smax + max(eps, 1e-12)

        # If degenerate, return a single band
        if len(edges) < 2:
            return np.array([smin, smax + 1e-12])

        min_n = max(1, int(np.ceil(min_bin_size * n_total)))

        # Iteratively merge
        while True:
            bins = self.__compute_bins(df, score_col, target_col, edges)
            # Drop any empty bins created by duplicate edges
            bins = bins[bins["n"] > 0].reset_index(drop=True)
            # Rebuild edges from non-empty bins to keep consistency
            if len(bins) >= 1:
                new_edges = [bins.loc[0, "left"]]
                for i in range(len(bins)):
                    new_edges.append(bins.loc[i, "right"])
                edges = np.array(new_edges)
            B = len(edges) - 1
            if B <= 1:
                break

            changed = False

            # Enforce minimum bin size first
            small_idx = bins.index[bins["n"] < min_n].tolist()
            if small_idx:
                i = small_idx[0]
                # choose neighbor with closest bad rate
                if i == 0:
                    k = 1  # merge boundary between bin 0 and 1 -> remove edges[1]
                elif i == B - 1:
                    k = B - 1  # merge boundary between last-1 and last -> remove edges[B-1]
                else:
                    dl = abs(bins.loc[i, "rate"] - bins.loc[i - 1, "rate"]) if bins.loc[i - 1, "n"] > 0 else np.inf
                    dr = abs(bins.loc[i, "rate"] - bins.loc[i + 1, "rate"]) if bins.loc[i + 1, "n"] > 0 else np.inf
                    k = i if dl <= dr else i + 1
                edges = np.delete(edges, k)
                changed = True
            else:
                # Compute adjacency p-values
                pvals = []
                for i in range(B - 1):
                    pvals.append(
                        self.__two_prop_pval(
                            int(bins.loc[i, "bad"]), int(bins.loc[i, "n"]),
                            int(bins.loc[i + 1, "bad"]), int(bins.loc[i + 1, "n"]),
                        )
                    )
                max_p = max(pvals) if pvals else 0.0
                idx_max = int(np.argmax(pvals)) if pvals else 0

                # Merge by significance or to reduce count to max_bands
                if (B > max_bands) or (max_p >= alpha):
                    # remove boundary between idx_max and idx_max+1
                    k = idx_max + 1
                    edges = np.delete(edges, k)
                    changed = True

            if not changed:
                break

        return edges
    
    def create_risk_bands(self):
        # Attach probabilities to df and define risk bands via data-driven edges
        # Step 1: attach PD scores
        df = df.copy()
        df['pd_score'] = self.all_proba

        # Step 2: compute significant bands (data-driven)
        edges = self.__make_significant_pd_bands(
            df,
            score_col='pd_score',
            target_col='target',
            init_bins=20,
            max_bands=7,
            min_bin_size=0.05,  # each band at least 5% of portfolio
            alpha=0.05,
        )

        # Step 3: assign band labels 1..K (1=best/lowest risk)
        K = len(edges) - 1
        cat = pd.cut(df['pd_score'], bins=edges, right=False, include_lowest=True)
        # codes in [0..K-1], -1 for NaN
        codes = cat.cat.codes.to_numpy()
        missing = int((codes < 0).sum())
        if missing:
            print(f"Warning: {missing} rows fell outside band edges; they will be imputed to nearest band.")
            # Impute: values < edges[0] -> 0, values >= edges[-1] -> K-1
            vals = df['pd_score'].to_numpy()
            codes = np.where(vals < edges[0], 0, codes)
            codes = np.where(vals >= edges[-1], K - 1, codes)
            codes = np.where(codes < 0, 0, codes)

        df['risk_band'] = codes + 1
        print(f"Formed {K} risk bands with edges: {np.round(edges, 4)}")
        
        return self.df, edges