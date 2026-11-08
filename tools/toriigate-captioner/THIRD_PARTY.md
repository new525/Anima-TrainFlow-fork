# Third-party components

## Bundled

### `prompts.py`: ToriiGate 0.5 prompts

| | |
|---|---|
| **From** | [Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5), file `scripts/prompts.py` |
| **Author** | Minthy |
| **Licence** | MIT, as declared for the model repository |
| **Changes** | None. The file is included byte for byte, and `tests/test_toriigate_captioner.py` checks its checksum. |

MIT licence text:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
> documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
> rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of
> the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
> WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
> COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
> OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Downloaded on first use (not included in this repository)

| Component | Source | Licence |
|---|---|---|
| llama.cpp `b12211`, Windows CUDA 12.4 build and its CUDA runtime | [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases/tag/b12211) | llama.cpp: MIT. The CUDA runtime DLLs are NVIDIA's, redistributed in that release. |
| ToriiGate 0.5 GGUF model and vision projector | [SleepVeryHard/ToriiGate-0.5_GGUF](https://huggingface.co/SleepVeryHard/ToriiGate-0.5_GGUF), a conversion of [Minthy/ToriiGate-0.5](https://huggingface.co/Minthy/ToriiGate-0.5) | See the model repositories |
