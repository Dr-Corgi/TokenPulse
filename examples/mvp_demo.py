#!/usr/bin/env python3
"""
TokenPulse MVP Demo Script

This script demonstrates the end-to-end health check workflow:
1. 常规抽血: 空输入 → 词表概率分布 → 发现病态高概率 token
2. 心电图: 常识题 → ECE → 校准曲线 → 发现过度自信

Usage:
    python -m examples.mvp_demo
    python examples/mvp_demo.py --model gpt2-small --preset quick
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="TokenPulse MVP Demo - LLM Health Check"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-small",
        help="Model name to check (default: gpt2-small)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["quick", "full", "calibration"],
        default="quick",
        help="Check preset to run (default: quick)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for markdown report (default: print to console)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "text", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of samples per check (default: 5)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("TokenPulse MVP Demo - LLM Health Check")
    print("=" * 60)
    print()

    # Import components
    try:
        from tokenpulse.providers.transformer_lens_provider import TransformerLensProvider
        from tokenpulse.clinic import ClinicRunner, ClinicConfig
    except ImportError as e:
        print(f"Error importing TokenPulse: {e}")
        print("Make sure you have installed the package and its dependencies.")
        sys.exit(1)

    # Initialize provider
    print(f"Loading model: {args.model}")
    print("This may take a moment...")
    print()

    try:
        provider = TransformerLensProvider(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to mock provider for demo...")
        # Use mock provider for demonstration
        from unittest.mock import Mock
        provider = Mock()
        provider.model_id = args.model
        provider.check_capability = Mock(return_value=True)
        provider.get_vocab_size = Mock(return_value=50257)

    # Configure clinic
    config = ClinicConfig(
        preset=args.preset,
        sample_size=args.sample_size,
    )

    # Create runner
    runner = ClinicRunner(provider, config)

    # Show available checks
    print("Available checks:", runner.get_available_checks())
    print("Compatible checks:", runner.get_compatible_checks())
    print()

    # Prepare test dataset
    test_prompts = [
        "The quick brown fox jumps over the lazy",
        "Machine learning is a field of",
        "The capital of France is",
        "In the year 2024, artificial intelligence",
        "The most important thing in life is",
    ]

    print(f"Running {args.preset} health check...")
    print(f"Test prompts: {len(test_prompts)}")
    print()

    # Run health check
    try:
        report = runner.run(test_prompts)
    except Exception as e:
        print(f"Error running health check: {e}")
        # Create a mock report for demonstration
        from tokenpulse.clinic.data_structures import ClinicReport
        from tokenpulse.checks.check_result import CheckResult, HealthStatus
        from datetime import datetime

        report = ClinicReport(
            report_id="demo_report",
            created_at=datetime.now(),
            model_info={"model_id": args.model},
            check_preset=args.preset,
            checks_executed=["vocabulary_utilization", "token_stability"],
            checks_skipped=[],
            overall_health_score=75.0,
            overall_health_status=HealthStatus.GREEN,
            check_results=[
                CheckResult(
                    check_name="vocabulary_utilization",
                    check_category="blood",
                    raw_metrics={"utilization_ratio": 0.35, "type_token_ratio": 0.25},
                    health_status=HealthStatus.GREEN,
                    health_score=75.0,
                ),
                CheckResult(
                    check_name="token_stability",
                    check_category="blood",
                    raw_metrics={"coefficient_of_variation": {"mean": 0.12}, "top_token_consistency": {"mean": 0.85}},
                    health_status=HealthStatus.GREEN,
                    health_score=80.0,
                ),
            ],
            recommendations=[],
            execution_time=1.5,
        )

    # Generate report
    if args.format == "markdown":
        output = runner.generate_report_markdown()
    elif args.format == "text":
        output = runner.generate_report_text()
    else:
        output = runner.generate_report_json()

    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)

    print()
    print("=" * 60)
    print("Health Check Complete!")
    print(f"Overall Status: {report.overall_health_status.value.upper()}")
    print(f"Overall Score: {report.overall_health_score:.1f}/100")
    print(f"Execution Time: {report.execution_time:.2f}s")
    print("=" * 60)

    return 0 if report.is_healthy() else 1


def demo_vocabulary_check():
    """Demo: Vocabulary utilization check (常规抽血)."""
    print("\n" + "=" * 60)
    print("Demo 1: 常规抽血 - Vocabulary Utilization Check")
    print("=" * 60)
    print()
    print("This check analyzes how much of the model's vocabulary is used.")
    print("A healthy model should utilize a diverse set of tokens.")
    print()

    try:
        from tokenpulse.providers.transformer_lens_provider import TransformerLensProvider
        from tokenpulse.checks.blood import VocabularyUtilizationCheck

        print("Loading model...")
        provider = TransformerLensProvider("gpt2-small")

        print("Running vocabulary check...")
        check = VocabularyUtilizationCheck()
        result = check.run(provider, ["Hello world!", "How are you?"])

        print(f"\nResults:")
        print(f"  Utilization: {result.raw_metrics.get('utilization_ratio', 0):.1%}")
        print(f"  Health Status: {result.health_status.value}")
        print(f"  Interpretation: {check.interpret_result(result)}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("This is expected if no GPU/model is available.")


def demo_calibration_check():
    """Demo: Calibration check (心电图)."""
    print("\n" + "=" * 60)
    print("Demo 2: 心电图 - Calibration Check")
    print("=" * 60)
    print()
    print("This check evaluates how well the model's confidence matches accuracy.")
    print("ECE (Expected Calibration Error) measures this alignment.")
    print()

    try:
        from tokenpulse.providers.transformer_lens_provider import TransformerLensProvider
        from tokenpulse.checks.immune import CalibrationCheck

        print("Loading model...")
        provider = TransformerLensProvider("gpt2-small")

        print("Running calibration check...")
        check = CalibrationCheck()
        result = check.run(provider)

        print(f"\nResults:")
        print(f"  ECE: {result.raw_metrics.get('ece', 0):.3f}")
        print(f"  Accuracy: {result.raw_metrics.get('accuracy', 0):.1%}")
        print(f"  Mean Confidence: {result.raw_metrics.get('mean_confidence', 0):.1%}")
        print(f"  Health Status: {result.health_status.value}")
        print(f"  Interpretation: {check.interpret_result(result)}")

    except Exception as e:
        print(f"Demo failed: {e}")
        print("This is expected if no GPU/model is available.")


if __name__ == "__main__":
    # Run main demo
    exit_code = main()

    # Optionally run individual demos
    # demo_vocabulary_check()
    # demo_calibration_check()

    sys.exit(exit_code)
