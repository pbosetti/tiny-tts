#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include "tiny_tts/tiny_tts.hpp"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Usage: tiny-tts-cli \"text\" [output.wav] [model_dir]\n";
    return EXIT_FAILURE;
  }

  const std::string text = argv[1];
  const std::string output = argc > 2 ? argv[2] : "output.wav";
  const std::string model_dir = argc > 3 ? argv[3] : "onnx";

  try {
    tiny_tts::TinyTTS tts(model_dir);
    tts.synthesize_to_file(text, output);
    std::cout << "Saved: " << output << '\n';
  } catch (const std::exception& ex) {
    std::cerr << "tiny-tts error: " << ex.what() << '\n';
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
