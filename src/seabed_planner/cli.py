"""Command-line interface; multiprocessing is launched only from explicit commands."""

import argparse
import json
from pathlib import Path

from .data import generate_mission, load_mission


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Open-source planner accompanying An et al. (2017)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hardware", help="Report CPU, memory and optional CUDA hardware")
    generate = commands.add_parser("generate", help="Generate a seeded paper-style mission")
    generate.add_argument("--targets", type=int, default=100)
    generate.add_argument("--seed", type=int, default=2017)
    generate.add_argument("--output", default="outputs/mission.json")
    run = commands.add_parser("plan", help="Select anchors and plan every target")
    run.add_argument("input", help="JSON, CSV or original MATLAB .mat")
    run.add_argument("--iterations", type=int, default=500)
    run.add_argument("--ants", type=int, default=30)
    run.add_argument("--seed", type=int, default=2017)
    run.add_argument("--workers", type=int, default=None, help="Omit for automatic CPU count")
    run.add_argument("--serial", action="store_true")
    run.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    run.add_argument("--speed", type=float)
    run.add_argument("--endurance", type=float)
    run.add_argument("--capacity", type=int)
    run.add_argument("--alpha", type=float, default=1.0)
    run.add_argument("--beta", type=float, default=0.5)
    run.add_argument("--dive-penalty", type=float, default=2.0)
    run.add_argument("--evaporation", type=float, default=0.5)
    run.add_argument("--output", default="outputs/plan.json")
    run.add_argument("--plot", default=None, help="Optional PNG/PDF/SVG figure path")
    validate = commands.add_parser(
        "validate", help="Independently recompute saved route constraints"
    )
    validate.add_argument("input")
    args = parser.parse_args(argv)
    try:
        if args.command == "hardware":
            from .hardware import detect_hardware

            print(json.dumps(detect_hardware(), indent=2, ensure_ascii=False))
        elif args.command == "generate":
            _save(args.output, generate_mission(args.targets, args.seed).as_dict())
            print(f"Saved {args.targets} targets to {args.output}")
        elif args.command == "validate":
            from .planner import validate_plan

            report = json.loads(Path(args.input).read_text(encoding="utf-8"))
            validate_plan(report)
            print("Valid: every target visited once; every dive satisfies endurance and capacity.")
        elif args.command == "plan":
            from .planner import plan_mission

            mission = load_mission(
                args.input, speed=args.speed, endurance=args.endurance, capacity=args.capacity
            )
            report = plan_mission(
                mission,
                iterations=args.iterations,
                ants=args.ants,
                seed=args.seed,
                workers=args.workers,
                parallel=not args.serial,
                backend=args.backend,
                alpha=args.alpha,
                beta=args.beta,
                dive_penalty=args.dive_penalty,
                evaporation=args.evaporation,
            )
            _save(args.output, report)
            if args.plot:
                from .plotting import plot_plan

                plot_plan(report, args.plot)
            print(
                f"{len(report['groups'])} anchors; {report['total_dives']} dives; "
                f"distance {report['total_distance']:.3f} {mission.distance_unit}; "
                f"{report['execution']['elapsed_seconds']:.3f}s; "
                f"backend={report['execution']['backend']}; "
                f"workers={report['execution']['workers']}. Saved {args.output}"
            )
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
