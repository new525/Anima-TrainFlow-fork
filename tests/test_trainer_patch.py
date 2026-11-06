"""The sd-scripts patch, without a GPU: the resume arithmetic and the state-folder cleanup are executed from the real
source of train_network.py / train_util.py (extracted verbatim), driven with the dataset sizes TrainFlow generates."""
import argparse, ast, math, os, re, shutil, sys, tempfile, textwrap, types
from pathlib import Path
from _harness import check, finish, REPO

TN = (REPO / "training" / "sd-scripts" / "train_network.py").read_text(encoding="utf-8")
TU = (REPO / "training" / "sd-scripts" / "library" / "train_util.py").read_text(encoding="utf-8")

def block(src, start, end_marker):
    i = src.index(start); j = src.index(end_marker, i) + len(end_marker)
    return textwrap.dedent(src[i:j])

def function_source(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise KeyError(name)

# --- resume / Train More with gradient accumulation -----------------------------------------------------------------
CALC = block(TN, "        # calculate steps to skip when resuming", "        global_step = 0\n")
SKIP = block(TN, "        if initial_step > 0:  # only if skip_until_initial_step", "            global_step = resumed_global_step\n")
quiet = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)

def resumed_run_ends_at(n_images, steps_for_repeats, max_steps, grad_accum, resume_step, calc=None):
    repeats = max(1, math.ceil(steps_for_repeats * grad_accum / n_images))   # app.py create_dataset_toml, batch 1
    dataloader = range(n_images * repeats)
    args = types.SimpleNamespace(initial_epoch=None, initial_step=None, max_train_steps=max_steps,
                                 skip_until_initial_step=True, resume="state", gradient_accumulation_steps=grad_accum)
    ns = dict(args=args, steps_from_state=resume_step, logger=quiet, math=math, train_dataloader=dataloader)
    exec(calc or CALC, ns); exec(SKIP, ns)
    epochs = math.ceil(max_steps / math.ceil(len(dataloader) / grad_accum))
    step, skip = ns["global_step"], ns["initial_step"]
    for _epoch in range(ns["epoch_to_start"], epochs):
        acc = 0
        for i in range(len(dataloader)):
            if skip > 0:
                skip -= 1; continue
            acc += 1
            if acc % grad_accum == 0 or i == len(dataloader) - 1:
                step += 1
            if step >= max_steps: break
        if step >= max_steps: break
    return step

for g in (1, 2, 4):
    for s in (300, 900, 1500, 2300):
        end = resumed_run_ends_at(37, 2400, 2400, g, s)
        check(f"resume from {s}/2400 with gradient accumulation {g} trains to the end", end == 2400, end)
    end = resumed_run_ends_at(40, 600, 700, g, 600)
    check(f"Train More 600 -> 700 with gradient accumulation {g} trains the extra steps", end == 700, end)

# the check above has teeth: upstream's formula fails it
UPSTREAM = "epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)"
fixed = re.search(r"epoch_to_start = initial_step // len\(train_dataloader\)", CALC)
check("patched formula is in place", fixed is not None)
if fixed:
    old = CALC[:fixed.start()] + UPSTREAM + CALC[fixed.end():]
    end = resumed_run_ends_at(37, 2400, 2400, 2, 1500, calc=old)
    check("(upstream formula would stop a resumed run at once: step 1500 of 2400)", end == 1500, end)

# --- halt states are cleaned up once superseded -----------------------------------------------------------------------
ns = {"argparse": argparse, "os": os, "shutil": shutil, "re": re, "logger": quiet, "huggingface_util": None,
      "STEP_STATE_NAME": "{}-step{:08d}-state", "DEFAULT_STEP_NAME": "at",
      "default_if_none": lambda v, d: d if v is None else v}
exec(function_source(TU, "save_and_remove_state_stepwise"), ns)
ns["train_util"] = types.SimpleNamespace(default_if_none=ns["default_if_none"], DEFAULT_STEP_NAME="at")
exec(function_source(TN, "remove_superseded_halt_states"), ns)
save_stepwise, cleanup = ns["save_and_remove_state_stepwise"], ns["remove_superseded_halt_states"]

def simulate(save_state, halts, max_steps=2400, every=300):
    out = Path(tempfile.mkdtemp())
    args = types.SimpleNamespace(output_name="kw", output_dir=str(out), save_state_to_huggingface=False,
                                 save_last_n_steps_state=every if save_state else None, save_last_n_steps=None,
                                 save_every_n_steps=every, save_state=save_state)
    acc = types.SimpleNamespace(save_state=lambda p: os.makedirs(p, exist_ok=True))
    snapshots = {}
    for step in range(1, max_steps + 1):
        # mirrors the patched loop: periodic save, then the halt check
        if step % every == 0 and save_state:
            save_stepwise(args, acc, step); cleanup(args, step)
        if step in halts:
            if not (step % every == 0 and save_state):
                save_stepwise(args, acc, step)
            cleanup(args, step)
            snapshots[step] = sorted(p.name for p in out.iterdir())
    if save_state:
        os.makedirs(out / "kw-state", exist_ok=True)       # save_state_on_train_end
    cleanup(args, max_steps + 1)
    final = sorted(p.name for p in out.iterdir())
    shutil.rmtree(out)
    return snapshots, final

snaps, final = simulate(True, {750, 1630})
check("resume points on: a halt state exists right after the halt", "kw-step00000750-state" in snaps[750], snaps[750])
check("resume points on: the halt state goes once the next save is written",
      "kw-step00000750-state" not in snaps[1630], snaps[1630])
check("resume points on: finished run keeps only the newest periodic states + end state",
      final == ["kw-state", "kw-step00002100-state", "kw-step00002400-state"], final)
snaps, final = simulate(False, {750, 900, 1630})
check("resume points off: the second halt replaces the first", snaps[900] == ["kw-step00000900-state"], snaps[900])
check("resume points off: only the newest halt state is kept", snaps[1630] == ["kw-step00001630-state"], snaps[1630])
check("resume points off: a finished run leaves no halt states behind", final == [], final)
finish()
