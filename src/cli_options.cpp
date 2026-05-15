#include "cli_options.hpp"

#include <stdexcept>

namespace tiny_tts {

std::string cli_usage() {
  return "Usage: tiny-tts-cli [options]\n"
         "  -t, --text <text>              Text to synthesize\n"
         "  -o, --output <wav>             Output audio file path (default: output.wav)\n"
         "  -c, --checkpoint <onnx_dir>    ONNX model directory (default: onnx)\n"
         "      --model-dir <onnx_dir>     Alias for --checkpoint\n"
         "  -s, --speaker <name|all>       Speaker ID/name (default: MALE)\n"
         "      --speed <float>            Speech speed (default: 1.0)\n"
         "      --device <cuda|cpu>        Device preference (default: cuda)\n"
         "  -h, --help                     Show this help\n";
}

CliOptions parse_cli_options(const std::vector<std::string>& args) {
  CliOptions options;

  for (std::size_t i = 0; i < args.size(); ++i) {
    const auto& arg = args[i];

    auto require_value = [&](const char* option_name) -> const std::string& {
      if (i + 1 >= args.size()) {
        throw std::invalid_argument(std::string("Missing value for ") + option_name);
      }
      return args[++i];
    };

    if (arg == "-h" || arg == "--help") {
      options.show_help = true;
    } else if (arg == "-t" || arg == "--text") {
      options.text = require_value(arg.c_str());
    } else if (arg == "-o" || arg == "--output") {
      options.output = require_value(arg.c_str());
    } else if (arg == "-c" || arg == "--checkpoint" || arg == "--model-dir") {
      options.model_dir = require_value(arg.c_str());
    } else if (arg == "-s" || arg == "--speaker") {
      options.speaker = require_value(arg.c_str());
    } else if (arg == "--speed") {
      options.speed = std::stof(require_value(arg.c_str()));
    } else if (arg == "--device") {
      options.device = require_value(arg.c_str());
    } else {
      throw std::invalid_argument("Unknown argument: " + arg);
    }
  }

  if (options.speed <= 0.0f) {
    throw std::invalid_argument("--speed must be greater than zero");
  }

  return options;
}

}  // namespace tiny_tts
