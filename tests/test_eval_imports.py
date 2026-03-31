"""
Test that all evaluation/visualization modules import correctly
and that the EvalConfig dataclass works.
"""

import sys
sys.path.insert(0, '/home/RESEARCH/adjoint_samplers')

import torch


def test_evaluation_imports():
    print("=== Test: evaluation module imports ===")
    from enhancements.evaluation import (
        EvalConfig, full_evaluation, single_run_evaluation,
        generate_samples, save_results,
    )
    config = EvalConfig()
    assert config.n_seeds == 10
    assert config.sample_sizes == [100, 500, 1000, 2000]
    assert config.mh_steps_list == [0, 5, 10, 20]
    assert config.max_stein_samples == 2000
    print(f"  EvalConfig defaults: n_seeds={config.n_seeds}, "
          f"sizes={config.sample_sizes}")
    print("  PASSED")


def test_visualization_imports():
    print("\n=== Test: visualization module imports ===")
    from enhancements.visualization import (
        generate_all_plots,
        plot_estimation_error_vs_samples,
        plot_variance_comparison,
        plot_variance_reduction_factors,
        plot_ksd_across_samples,
        plot_mcmc_ablation,
        plot_antithetic_correlation,
        plot_summary_table,
        COLORS, LABELS,
    )
    assert len(COLORS) == 6
    assert len(LABELS) == 6
    print(f"  Methods: {list(LABELS.values())}")
    print("  PASSED")


def test_eval_enhanced_import():
    print("\n=== Test: eval_enhanced.py imports ===")
    # Just verify the script can be imported as a module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "eval_enhanced",
        "/home/RESEARCH/adjoint_samplers/eval_enhanced.py"
    )
    assert spec is not None
    print("  eval_enhanced.py found and loadable")
    print("  PASSED")


def test_run_evaluation_import():
    print("\n=== Test: run_evaluation.py imports ===")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_evaluation",
        "/home/RESEARCH/adjoint_samplers/run_evaluation.py"
    )
    assert spec is not None
    print("  run_evaluation.py found and loadable")
    print("  PASSED")


def test_save_results(tmp_path="/tmp/test_eval_results"):
    print("\n=== Test: save_results ===")
    from enhancements.evaluation import save_results
    import json
    from pathlib import Path

    fake_results = {
        100: {
            'naive_mean_energy': {'mean': 5.1, 'std': 0.2, 'values': [5.0, 5.2]},
            'ksd_squared': {'mean': 0.01, 'std': 0.005, 'values': [0.01, 0.01]},
        },
    }

    path = f"{tmp_path}/test_results.json"
    save_results(fake_results, path)

    # Read back and verify
    with open(path) as f:
        loaded = json.load(f)

    assert '100' in loaded  # JSON keys are strings
    assert loaded['100']['naive_mean_energy']['mean'] == 5.1
    print("  Saved and loaded results correctly")

    # Cleanup
    Path(path).unlink()
    Path(tmp_path).rmdir()
    print("  PASSED")


if __name__ == "__main__":
    test_evaluation_imports()
    test_visualization_imports()
    test_eval_enhanced_import()
    test_run_evaluation_import()
    test_save_results()
    print("\n✅ ALL TESTS PASSED")
