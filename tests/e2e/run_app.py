"""Start the real app.py with torch stubbed, U2Net replaced and the trainer command swapped for fake_trainer.py.
usage: run_app.py <app.py inside a test tree> <port>"""
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)
sys.modules["torch"] = torch
APP, PORT = Path(sys.argv[1]), int(sys.argv[2])
src = APP.read_text(encoding="utf-8").replace('if __name__ == "__main__":', "if False:")
marker = "# A trainer that kept running while the app was closed is picked up again here."
assert marker in src
src = src.replace(marker, "build_train_command = __fake_cmd__\n" + marker)


def fake_cmd(training_toml, dataset_toml, extra_args=()):
    return [sys.executable, str(HERE / "fake_trainer.py"), "--config_file", str(training_toml), "--dataset_config", str(dataset_toml), *extra_args]


g = {"__name__": "app", "__file__": str(APP), "__fake_cmd__": fake_cmd}
exec(compile(src, str(APP), "exec"), g)
g["smart_cropper"].load_model = lambda: "FAKE (no U2Net in tests)"
g["smart_cropper"].process_image = lambda img, tw, th: __import__("cv2").resize(img, (tw, th))
# same file-serving rule as the real launch: previews and the grid live under training/output
g["ui"].launch(server_port=PORT, theme=g["gr"].themes.Soft(), css=g["CSS"], prevent_thread_lock=False,
               allowed_paths=[str(g["OUTPUT_BASE"].resolve())])
