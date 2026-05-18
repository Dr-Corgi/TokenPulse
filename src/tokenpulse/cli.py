"""
Command-line interface for TokenPulse.
"""

import argparse
import sys


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tokenpulse",
        description="LLM Model Diagnostic Tool - Health Check for Large Language Models"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Run health check on a model"
    )
    check_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model ID or path (e.g., gpt2-small, meta-llama/Llama-3-8b)"
    )
    check_parser.add_argument(
        "--preset", "-p",
        choices=["quick", "full"],
        default="quick",
        help="Check preset (default: quick)"
    )
    check_parser.add_argument(
        "--output", "-o",
        help="Output file for report (default: stdout)"
    )
    check_parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Report format (default: markdown)"
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show model information"
    )
    info_parser.add_argument(
        "--model", "-m",
        required=True,
        help="Model ID or path"
    )

    # List checks command
    list_parser = subparsers.add_parser(
        "list-checks",
        help="List available check items"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "check":
        _run_check(args)
    elif args.command == "info":
        _show_info(args)
    elif args.command == "list-checks":
        _list_checks()


def _run_check(args):
    """Run health check on a model."""
    print(f"TokenPulse Health Check")
    print(f"Model: {args.model}")
    print(f"Preset: {args.preset}")
    print("-" * 40)
    print()
    print("P0 phase: Core infrastructure only.")
    print("Please install dependencies and use the Python API for full functionality.")
    print()
    print("Example:")
    print("  from tokenpulse import TransformerLensProvider")
    print(f"  provider = TransformerLensProvider('{args.model}')")
    print("  output = provider.generate('Hello, world!', return_hidden_states=True)")


def _show_info(args):
    """Show model information."""
    print(f"Model: {args.model}")
    print()
    print("Use Python API to get detailed model information:")
    print("  from tokenpulse import TransformerLensProvider")
    print(f"  provider = TransformerLensProvider('{args.model}')")
    print("  print(f'Vocab size: {provider.get_vocab_size()}')")
    print("  print(f'Layers: {provider.get_num_layers()}')")


def _list_checks():
    """List available check items."""
    print("Available Check Items:")
    print()
    print("P1 Phase (MVP):")
    print("  - vocabulary_utilization: Vocabulary utilization check")
    print("  - token_stability: Token probability stability check")
    print("  - confidence_calibration: Confidence calibration check")
    print()
    print("P2 Phase (Specialized):")
    print("  - perturbation_sensitivity: Perturbation sensitivity check")
    print("  - attention_sink: Attention sink check")
    print("  - representation_collapse: Representation collapse check")


if __name__ == "__main__":
    main()
