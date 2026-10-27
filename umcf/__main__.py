"""python -m umcf <command>"""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: python -m umcf {generate,train,test,infer,experiment,released-check}\n"
            "       python -m umcf generate --quick --out data/processed/demo.npz\n"
            "       python -m umcf train --data ... --out checkpoints/umcf.pt\n"
            "       python -m umcf test  --data ... --ckpt checkpoints/umcf.pt\n"
            "       python -m umcf infer --data ... --ckpt checkpoints/umcf.pt --index 0\n"
            "       python -m umcf experiment --quick --ablation --baselines\n"
            "       python -m umcf released-check --quick"
        )
        sys.exit(1)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    sys.argv = [sys.argv[0]] + rest
    if cmd in ("generate", "gen", "data"):
        from .generate import main as _m
    elif cmd == "train":
        from .train import main as _m
    elif cmd == "test":
        from .test import main as _m
    elif cmd == "infer":
        from .infer import main as _m
    elif cmd in ("experiment", "exp", "run"):
        from .experiment import main as _m
    elif cmd in ("released-check", "released_check", "check"):
        from .released_check import main as _m
    else:
        raise SystemExit(f"Unknown command: {cmd}")
    _m()


if __name__ == "__main__":
    main()
