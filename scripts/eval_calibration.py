from __future__ import annotations

import argparse
import pandas as pd
from agri_reliability.metrics.calibration import expected_calibration_error, maximum_calibration_error, error_detection_auroc
from agri_reliability.metrics.risk_coverage import risk_coverage_curve, risk_coverage_auc
from agri_reliability.reporting.plots import plot_reliability_diagram, plot_risk_coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True, help='CSV with columns confidence, correct')
    parser.add_argument('--out-prefix', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    conf = df['confidence'].values
    correct = df['correct'].astype(int).values

    metrics = {
        'ece': expected_calibration_error(conf, correct),
        'mce': maximum_calibration_error(conf, correct),
        'error_detection_auroc': error_detection_auroc(conf, correct),
        'risk_coverage_auc': risk_coverage_auc(conf, correct),
    }
    pd.DataFrame([metrics]).to_csv(args.out_prefix + '_calibration_metrics.csv', index=False)
    coverage, risk = risk_coverage_curve(conf, correct)
    pd.DataFrame({'coverage': coverage, 'risk': risk}).to_csv(args.out_prefix + '_risk_coverage.csv', index=False)
    plot_reliability_diagram(conf, correct, args.out_prefix + '_reliability.pdf')
    plot_risk_coverage(coverage, risk, args.out_prefix + '_risk_coverage.pdf')
    print(metrics)


if __name__ == '__main__':
    main()
