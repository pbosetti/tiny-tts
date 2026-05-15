#include "tiny_tts/tiny_tts.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "onnx_engine.hpp"
#include "text_frontend.hpp"
#include "wav_writer.hpp"

namespace tiny_tts {

class TinyTTS::Impl {
 public:
  Impl(std::string model_dir, std::string cmudict_path, std::string device)
      : frontend(std::move(cmudict_path)),
        engine(std::move(model_dir), device == "cuda" || device == "gpu") {}

  static std::int64_t speaker_id_for(const std::string& speaker_name) {
    static const std::unordered_map<std::string, std::int64_t> kSpeakerMap = {
        {"MALE", 0}, {"FEMALE", 0}};
    auto normalized = speaker_name;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    const auto it = kSpeakerMap.find(normalized);
    if (it != kSpeakerMap.end()) {
      return it->second;
    }
    std::cerr << "Warning: Speaker " << speaker_name << " not found, using ID 0\n";
    return 0;
  }

  TextFrontend frontend;
  OnnxEngine engine;
};

TinyTTS::TinyTTS(std::string model_dir, std::string cmudict_path, std::string device)
    : impl_(new Impl(std::move(model_dir), std::move(cmudict_path), std::move(device))) {}

TinyTTS::~TinyTTS() { delete impl_; }

std::vector<float> TinyTTS::synthesize(const std::string& text,
                                       const SynthesisOptions& options) const {
  if (options.speed <= 0.0f) {
    throw std::invalid_argument("speed must be greater than zero");
  }
  const auto ids = impl_->frontend.text_to_ids(text);

  OnnxInferenceInput input{.phone_ids = ids.phone_ids,
                           .tone_ids = ids.tone_ids,
                           .lang_ids = ids.lang_ids,
                           .noise_scale = options.noise_scale,
                           .noise_scale_w = options.noise_scale_w,
                           .length_scale = 1.0f / options.speed,
                           .speaker_id = Impl::speaker_id_for(options.speaker)};
  return impl_->engine.run(input);
}

void TinyTTS::synthesize_to_file(const std::string& text, const std::string& output_path,
                                 const SynthesisOptions& options) const {
  auto audio = synthesize(text, options);
  const auto parent = std::filesystem::path(output_path).parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  write_wav_16bit(output_path, audio, 44100);
}

std::vector<std::string> TinyTTS::available_speakers() const { return {"MALE"}; }

}  // namespace tiny_tts
