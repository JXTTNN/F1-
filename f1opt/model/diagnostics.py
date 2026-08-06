from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class RegressionMetrics:
    mae: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    r2: float = 0.0
    mape: float = 0.0
    max_error: float = 0.0
    explained_variance: float = 0.0


@dataclass
class ResidualDiagnostics:
    mean: float = 0.0
    std: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    durbin_watson: float = 0.0
    jarque_bera: float = 0.0
    jb_pvalue: float = 0.0


class ModelDiagnostics:

    @staticmethod
    def compute_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> RegressionMetrics:
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        residuals = y_true - y_pred
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))

        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            apei = np.abs(residuals / np.where(y_true == 0, 1e-8, y_true))
            mape = float(np.mean(apei[apei < 1e6])) * 100.0

        return RegressionMetrics(
            mae=mae,
            mse=mse,
            rmse=np.sqrt(mse),
            r2=r2,
            mape=mape,
            max_error=float(np.max(np.abs(residuals))),
            explained_variance=float(1.0 - np.var(residuals) / np.var(y_true)) if np.var(y_true) > 0 else 0.0,
        )

    @staticmethod
    def analyze_residuals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ResidDiagnostics:
        residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
        n = len(residuals)

        from scipy import stats as sp_stats

        mean_r = float(np.mean(residuals))
        std_r = float(np.std(residuals, ddof=1))
        skew = float(sp_stats.skew(residuals))
        kurt = float(sp_stats.kurtosis(residuals, fisher=True))

        dw_num = np.sum(np.diff(residuals) ** 2)
        dw_den = np.sum(residuals ** 2)
        dw = float(dw_num / dw_den) if dw_den > 0 else 2.0

        jb_stat, jb_p = sp_stats.jarque_bera(residuals)

        return ResidDiagnostics(
            mean=mean_r,
            std=std_r,
            skewness=skew,
            kurtosis=kurt,
            durbin_watson=dw,
            jarque_bera=float(jb_stat),
            jb_pvalue=float(jb_p),
        )

    @staticmethod
    def report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
        m = ModelDiagnostics.compute_regression_metrics(y_true, y_pred)
        r = ModelDiagnostics.analyze_residuals(y_true, y_pred)

        lines = [
            "Model Diagnostics Report",
            "=" * 32,
            f"MAE:  {m.mae:.4f}",
            f"RMSE: {m.rmse:.4f}",
            f"R2:   {m.r2:.4f}",
            f"MAPE: {m.mape:.2f}%",
            f"Max Error: {m.max_error:.4f}",
            "",
            "Residual Analysis",
            "-" * 32,
            f"Mean: {r.mean:.6f}",
            f"Std:  {r.std:.6f}",
            f"Skewness: {r.skewness:.4f}",
            f"Kurtosis: {r.kurtosis:.4f}",
            f"DW:  {r.durbin_watson:.4f}",
            f"JB:  {r.jarque_bera:.4f} (p={r.jb_pvalue:.4f})",
        ]
        return "\n".join(lines)
