#pragma once

#include <cstdint>
#include <array>
#include <random>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"

namespace tiny_tts {

struct OnnxInferenceInput {
  std::vector<std::int64_t> phone_ids;
  std::vector<std::int64_t> tone_ids;
  std::vector<std::int64_t> lang_ids;
  float noise_scale;
  float noise_scale_w;
  float length_scale;
  std::int64_t speaker_id;
};

class OnnxEngine {
 public:
  explicit OnnxEngine(const std::string& model_dir, bool use_gpu = false);

  std::vector<float> run(const OnnxInferenceInput& input) const;

 private:
  static std::vector<std::int64_t> shape_of(const Ort::Value& tensor);

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  Ort::Session encoder_;
  Ort::Session duration_predictor_;
  Ort::Session flow_;
  Ort::Session decoder_;
  mutable std::mt19937 rng_;
};

}  // namespace tiny_tts
