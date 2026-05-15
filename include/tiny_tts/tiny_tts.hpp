#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tiny_tts {

struct SynthesisOptions {
  std::string speaker{"MALE"};
  std::string device{"cuda"};
  float noise_scale{0.667f};
  float noise_scale_w{0.8f};
  float speed{1.0f};
};

class TinyTTS {
 public:
  explicit TinyTTS(std::string model_dir = "onnx", std::string cmudict_path = {},
                   std::string device = "cuda");
  ~TinyTTS();
  TinyTTS(const TinyTTS&) = delete;
  TinyTTS& operator=(const TinyTTS&) = delete;
  TinyTTS(TinyTTS&&) = delete;
  TinyTTS& operator=(TinyTTS&&) = delete;

  std::vector<float> synthesize(const std::string& text,
                                const SynthesisOptions& options = {}) const;

  void synthesize_to_file(const std::string& text, const std::string& output_path,
                          const SynthesisOptions& options = {}) const;
  std::vector<std::string> available_speakers() const;

 private:
  class Impl;
  Impl* impl_;
};

}  // namespace tiny_tts
