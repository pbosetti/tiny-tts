# tiny-tts (C++20)

`tiny-tts` is a minimal C++20 text-to-speech engine for converting `std::string` to speech with the repository ONNX models.

## What changed

- Migrated runtime implementation to C++20.
- No Python runtime dependency is required.
- Inference uses ONNX Runtime C++ API and the existing ONNX model set in `onnx/`:
  - `text_encoder.onnx`
  - `duration_predictor.onnx`
  - `flow.onnx`
  - `decoder.onnx`
- Added a reusable C++ API and a minimal CLI.

## Build requirements

- CMake >= 3.16
- C++20 compiler
- ONNX Runtime C/C++ package (headers + library)

Set `ONNXRUNTIME_ROOT` to your ONNX Runtime installation root (must contain `include/` and `lib/` or `lib64/`).

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
```

## Run

```bash
./build/tiny-tts-cli --text "Hello world from tiny tts" --output output.wav --checkpoint onnx --speaker MALE --speed 1.0 --device cpu
```

CLI options (semantically aligned with the previous Python CLI):

- `-t, --text`: text input (default sentence is provided)
- `-c, --checkpoint`: ONNX model directory
- `-o, --output`: output WAV path
- `-s, --speaker`: speaker name (`MALE`) or `all`
- `--speed`: speech speed (`1.0` default)
- `--device`: `cuda` or `cpu` device preference

## C++ API

```cpp
#include "tiny_tts/tiny_tts.hpp"

tiny_tts::TinyTTS tts("onnx");

tiny_tts::SynthesisOptions opts;
opts.speaker = "MALE";
opts.speed = 1.0f;
opts.device = "cpu";

tts.synthesize_to_file("Hello world", "output.wav", opts);
```

## Notes and assumptions

- Current frontend targets English and uses `resources/cmudict.rep`.
- Unknown words fallback to symbol-level pronunciation when not found in CMU dictionary.
- Output WAV is mono 44.1kHz 16-bit PCM.
