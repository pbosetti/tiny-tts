#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "cli_options.hpp"
#include "tiny_tts/tiny_tts.hpp"

int main(int argc, char** argv) {
  try {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc > 1 ? argc - 1 : 0));
    for (int i = 1; i < argc; ++i) {
      args.emplace_back(argv[i]);
    }

    const auto options = tiny_tts::parse_cli_options(args);
    if (options.show_help) {
      std::cout << tiny_tts::cli_usage();
      return EXIT_SUCCESS;
    }

    tiny_tts::TinyTTS tts(options.model_dir, {}, options.device);
    tiny_tts::SynthesisOptions synth_options;
    synth_options.speaker = options.speaker;
    synth_options.device = options.device;
    synth_options.speed = options.speed;

    if (options.speaker == "all" || options.speaker == "ALL") {
      for (const auto& speaker_name : tts.available_speakers()) {
        auto output_path = std::filesystem::path(options.output);
        const auto stem = output_path.stem().string();
        const auto ext = output_path.extension().string().empty() ? ".wav" : output_path.extension().string();
        const auto parent = output_path.parent_path();
        const auto full_name = stem + "_spk" + speaker_name + ext;
        const auto final_path = parent.empty() ? std::filesystem::path(full_name) : parent / full_name;

        synth_options.speaker = speaker_name;
        tts.synthesize_to_file(options.text, final_path.string(), synth_options);
        std::cout << "Saved: " << final_path.string() << '\n';
      }
    } else {
      tts.synthesize_to_file(options.text, options.output, synth_options);
      std::cout << "Saved: " << options.output << '\n';
    }
  } catch (const std::exception& ex) {
    std::cerr << "tiny-tts error: " << ex.what() << '\n';
    std::cerr << tiny_tts::cli_usage();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
