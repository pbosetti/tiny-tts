#include "tiny_tts/tiny_tts.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

#include "onnx_engine.hpp"
#include "text_frontend.hpp"
#include "wav_writer.hpp"

namespace tiny_tts {

class TinyTTS::Impl {
 public:
  Impl(std::string model_dir, std::string cmudict_path)
      : frontend(std::move(cmudict_path)), engine(std::move(model_dir)) {}

  TextFrontend frontend;
  OnnxEngine engine;
};

TinyTTS::TinyTTS(std::string model_dir, std::string cmudict_path)
    : impl_(new Impl(std::move(model_dir), std::move(cmudict_path))) {}

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
                           .speaker_id = options.speaker_id};
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

}  // namespace tiny_tts
