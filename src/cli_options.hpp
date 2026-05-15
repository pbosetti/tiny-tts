#pragma once

#include <string>
#include <vector>

namespace tiny_tts {

struct CliOptions {
  std::string text{"The weather is nice today, and I feel very relaxed."};
  std::string output{"output.wav"};
  std::string model_dir{"onnx"};
  std::string speaker{"MALE"};
  std::string device{"cuda"};
  float speed{1.0f};
  bool show_help{false};
};

CliOptions parse_cli_options(const std::vector<std::string>& args);
std::string cli_usage();

}  // namespace tiny_tts
