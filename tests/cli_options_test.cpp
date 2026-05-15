#include <cstdlib>
#include <vector>

#include "cli_options.hpp"

int main() {
  const auto opts = tiny_tts::parse_cli_options(
      {"--text", "Hello world", "-o", "out.wav", "-c", "onnx", "-s", "MALE", "--speed", "1.25", "--device", "cpu"});

  if (opts.text != "Hello world" || opts.output != "out.wav" || opts.model_dir != "onnx" ||
      opts.speaker != "MALE" || opts.speed != 1.25f || opts.device != "cpu") {
    return EXIT_FAILURE;
  }

  const auto all_opts = tiny_tts::parse_cli_options({"--speaker", "all"});
  if (all_opts.speaker != "all") {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
