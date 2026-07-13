import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# 你要跑的参数列表
# LESS_FRACS = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.7]
LESS_FRACS = [0.02, 0.03, 0.04, 0.05, 0.07]
LOG_DIR = "twitter_logs_less_frac_aug"

def frac_to_tag(x: float) -> str:
    # 0.5 -> 0p50, 0.05 -> 0p05
    return f"{x:.2f}".replace(".", "p")

def run_one(less_frac: float):
    os.makedirs(LOG_DIR, exist_ok=True)

    tag = f"less{frac_to_tag(less_frac)}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"{tag}_{ts}.log")

    llm_model = os.environ.get("SHARP_LLM_MODEL")
    prompt_seed = os.environ.get(
        "SHARP_PROMPT_SEED",
        str(REPO_ROOT / "data" / "twitter15_16" / "prompt_seed" / "clippool_16_release.xlsx"),
    )
    if not llm_model:
        raise RuntimeError("Set SHARP_LLM_MODEL before running the sweep.")
    if not os.path.isfile(prompt_seed):
        raise RuntimeError(f"Twitter15/16 prompt seed not found: {prompt_seed}")

    cmd = [
        sys.executable,
        "-u",
        "model_adjust.py",
        "--less-frac",
        str(less_frac),
        "--llm-model",
        llm_model,
        "--prompt-seed",
        prompt_seed,
    ]

    print(f"\n[SWEEP] Running less_frac={less_frac}")
    print(f"[SWEEP] CMD: {' '.join(cmd)}")
    print(f"[SWEEP] LOG: {log_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"[CMD] {' '.join(cmd)}\n")
        f.write(f"[START] {datetime.now().isoformat()}\n\n")
        f.flush()

        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in p.stdout:
            f.write(line)
            f.flush()
            print(line, end="")  # 同时在终端显示（想静默就删掉这一行）

        rc = p.wait()
        f.write(f"\n[END] {datetime.now().isoformat()} returncode={rc}\n")
        f.flush()

    if rc != 0:
        print(f"[SWEEP] ❌ Failed less_frac={less_frac}. See: {log_path}")
    else:
        print(f"[SWEEP] ✅ Done less_frac={less_frac}. Log: {log_path}")

def main():
    for lf in LESS_FRACS:
        run_one(lf)

if __name__ == "__main__":
    main()
