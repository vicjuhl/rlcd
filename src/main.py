import argparse
from typing import Optional, Sequence, Dict, Any

from rlcd.search import search
from rlcd.gen_data import gen_data

from rlcd.model import QNetwork

def parse_args(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(prog="rlcd")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    parsed = parser.parse_args(argv)
    return vars(parsed)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main entry point (argv-only).

    Parses CLI arguments from `argv` when provided, otherwise from the real
    command-line. This keeps `main` signature simple and focused on CLI usage.
    """
    args = parse_args(argv)
    if args["verbose"]:
        print("verbose")
    df = gen_data()



    # policy_net = QNetwork(...) #arg here TODO
    search(df)

if __name__ == "__main__":
    main()