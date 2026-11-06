"""Stand-in for accelerate + the patched train_network.py: writes the same files and names and prints the same
progress/halt lines, so the app can be driven end to end without a GPU. FAKE_SIDE_LOG gets one line per step."""
import sys, time, toml, os, json
from pathlib import Path
from PIL import Image
argv = sys.argv[1:]
cfg = toml.load(argv[argv.index("--config_file") + 1])
out = Path(cfg["output_dir"]); name = cfg["output_name"]; total = cfg["max_train_steps"]
save_every = cfg.get("save_every_n_steps"); sample_every = cfg.get("sample_every_n_steps")
halt = Path(cfg["halt_request_file"]); side = open(os.environ["FAKE_SIDE_LOG"], "a", buffering=1)
(Path(os.environ["FAKE_SIDE_LOG"]).parent / "last_argv.json").write_text(json.dumps(argv))
prompts = [l for l in open(cfg["sample_prompts"], encoding="utf-8").read().splitlines() if l.strip() and not l.startswith("#")]
delay = float(os.environ.get("FAKE_DELAY", "0.05"))
start = 0
if "--resume" in argv:
    assert "--skip_until_initial_step" in argv
    start = json.load(open(Path(argv[argv.index("--resume") + 1]) / "train_state.json"))["current_step"]
    print(f"INFO load train state: current_step={start}", flush=True)
def save_state(dirname, step):
    st = out / dirname; st.mkdir(exist_ok=True)
    (st / "train_state.json").write_text(json.dumps({"current_epoch": 1, "current_step": step}))
    (st / "random_states_0.pkl").write_bytes(b"rng")
print("INFO loading model ...", flush=True)
for step in range(start + 1, total + 1):
    loss = 0.3 / (1 + step / 100)
    for tick in range(2):
        sys.stdout.write(f"\rsteps:  {step*100//total}%|##  | {step}/{total} [00:01<01:00,  0.05s/it, avr_loss={loss + tick*1e-4:.4f}]"); sys.stdout.flush()
    side.write(f"{time.time():.2f} step {step}\n")
    if sample_every and step % sample_every == 0:
        print(f"\ngenerating sample images at step {step}", flush=True)
        (out / "sample").mkdir(exist_ok=True)
        for i in range(len(prompts)):
            ts = time.strftime("%Y%m%d%H%M%S")
            Image.new("RGB", (64, 96), (step % 255, 40 * i, 90)).save(out / "sample" / f"{name}_{step:06d}_{i:02d}_{ts}_42.png")
    if save_every and step % save_every == 0:
        (out / f"{name}-step{step:08d}.safetensors").write_bytes(b"lora")
        if cfg.get("save_state"): save_state(f"{name}-step{step:08d}-state", step)
    if step < total and halt.exists():
        if not (save_every and step % save_every == 0 and cfg.get("save_state")):
            (out / f"{name}-step{step:08d}.safetensors").write_bytes(b"lora"); save_state(f"{name}-step{step:08d}-state", step)
        halt.unlink(); print(f"\nINFO HALTED at step {step}", flush=True); sys.exit(0)
    time.sleep(delay)
if cfg.get("save_state"): save_state(f"{name}-state", total)
(out / f"{name}.safetensors").write_bytes(b"final")
print("\nINFO model saved.", flush=True)
