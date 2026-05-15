#include "onnx_engine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <numeric>
#include <stdexcept>

#if __has_include(<cuda_provider_factory.h>)
#include <cuda_provider_factory.h>
#define TINY_TTS_HAS_ORT_CUDA 1
#else
#define TINY_TTS_HAS_ORT_CUDA 0
#endif

namespace tiny_tts {
namespace {

using Shape = std::vector<std::int64_t>;

std::size_t product(const Shape& shape) {
  return static_cast<std::size_t>(std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                                                  std::multiplies<std::int64_t>()));
}

std::filesystem::path model_path(const std::string& model_dir, const char* model_name) {
  return std::filesystem::path(model_dir) / model_name;
}

}  // namespace

OnnxEngine::OnnxEngine(const std::string& model_dir, bool use_gpu)
    : env_(ORT_LOGGING_LEVEL_WARNING, "tiny_tts"),
      encoder_(nullptr),
      duration_predictor_(nullptr),
      flow_(nullptr),
      decoder_(nullptr),
      rng_(std::random_device{}()) {
  session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.SetIntraOpNumThreads(0);
#if TINY_TTS_HAS_ORT_CUDA
  if (use_gpu) {
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options_, 0);
  }
#else
  (void)use_gpu;
#endif

  auto enc_path = model_path(model_dir, "text_encoder.onnx");
  auto dp_path = model_path(model_dir, "duration_predictor.onnx");
  auto flow_path = model_path(model_dir, "flow.onnx");
  auto dec_path = model_path(model_dir, "decoder.onnx");

  if (!std::filesystem::exists(enc_path) || !std::filesystem::exists(dp_path) ||
      !std::filesystem::exists(flow_path) || !std::filesystem::exists(dec_path)) {
    throw std::runtime_error("Missing ONNX model(s) in: " + model_dir);
  }

  encoder_ = Ort::Session(env_, enc_path.c_str(), session_options_);
  duration_predictor_ = Ort::Session(env_, dp_path.c_str(), session_options_);
  flow_ = Ort::Session(env_, flow_path.c_str(), session_options_);
  decoder_ = Ort::Session(env_, dec_path.c_str(), session_options_);
}

std::vector<std::int64_t> OnnxEngine::shape_of(const Ort::Value& tensor) {
  auto info = tensor.GetTensorTypeAndShapeInfo();
  return info.GetShape();
}

std::vector<float> OnnxEngine::run(const OnnxInferenceInput& input) const {
  (void)input.noise_scale_w;
  const auto seq_len = static_cast<std::int64_t>(input.phone_ids.size());
  if (seq_len == 0) {
    throw std::runtime_error("Empty phoneme sequence");
  }

  if (input.tone_ids.size() != input.phone_ids.size() ||
      input.lang_ids.size() != input.phone_ids.size()) {
    throw std::runtime_error("Input id vectors must have same length");
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::array<std::int64_t, 2> seq_shape{1, seq_len};
  std::array<std::int64_t, 3> bert_shape{1, 1024, seq_len};
  std::array<std::int64_t, 3> ja_bert_shape{1, 768, seq_len};

  std::vector<float> bert(static_cast<std::size_t>(1024 * seq_len), 0.0f);
  std::vector<float> ja_bert(static_cast<std::size_t>(768 * seq_len), 0.0f);
  std::array<std::int64_t, 1> len_shape{1};
  std::array<std::int64_t, 1> scalar_shape{1};

  auto x = Ort::Value::CreateTensor<std::int64_t>(mem, const_cast<std::int64_t*>(input.phone_ids.data()),
                                                   input.phone_ids.size(), seq_shape.data(), seq_shape.size());
  std::array<std::int64_t, 1> x_lengths_data{seq_len};
  auto x_lengths = Ort::Value::CreateTensor<std::int64_t>(mem, x_lengths_data.data(), x_lengths_data.size(),
                                                           len_shape.data(), len_shape.size());

  auto sid = Ort::Value::CreateTensor<std::int64_t>(mem, const_cast<std::int64_t*>(&input.speaker_id), 1,
                                                     scalar_shape.data(), scalar_shape.size());
  auto tone = Ort::Value::CreateTensor<std::int64_t>(mem, const_cast<std::int64_t*>(input.tone_ids.data()),
                                                      input.tone_ids.size(), seq_shape.data(), seq_shape.size());
  auto language = Ort::Value::CreateTensor<std::int64_t>(
      mem, const_cast<std::int64_t*>(input.lang_ids.data()), input.lang_ids.size(), seq_shape.data(), seq_shape.size());
  auto bert_tensor = Ort::Value::CreateTensor<float>(mem, bert.data(), bert.size(), bert_shape.data(), bert_shape.size());
  auto ja_bert_tensor =
      Ort::Value::CreateTensor<float>(mem, ja_bert.data(), ja_bert.size(), ja_bert_shape.data(), ja_bert_shape.size());

  const char* enc_input_names[] = {"phone_ids", "phone_lengths", "tone_ids", "language_ids", "bert", "ja_bert", "speaker_id"};
  const char* enc_output_names[] = {"x", "m_p", "logs_p", "x_mask", "g"};

  std::array<Ort::Value, 7> enc_inputs = {std::move(x), std::move(x_lengths), std::move(tone), std::move(language),
                                          std::move(bert_tensor), std::move(ja_bert_tensor), std::move(sid)};

  auto enc_outputs = encoder_.Run(Ort::RunOptions{nullptr}, enc_input_names, enc_inputs.data(), enc_inputs.size(),
                                  enc_output_names, 5);

  float* x_enc = enc_outputs[0].GetTensorMutableData<float>();
  float* m_p = enc_outputs[1].GetTensorMutableData<float>();
  float* logs_p = enc_outputs[2].GetTensorMutableData<float>();
  float* x_mask = enc_outputs[3].GetTensorMutableData<float>();
  float* g = enc_outputs[4].GetTensorMutableData<float>();

  const auto x_enc_shape = shape_of(enc_outputs[0]);
  const auto m_p_shape = shape_of(enc_outputs[1]);
  const auto logs_p_shape = shape_of(enc_outputs[2]);
  const auto x_mask_shape = shape_of(enc_outputs[3]);
  const auto g_shape = shape_of(enc_outputs[4]);

  const char* dp_input_names[] = {"x", "x_mask", "g"};
  const char* dp_output_names[] = {"logw"};

  auto x_enc_tensor = Ort::Value::CreateTensor<float>(mem, x_enc, product(x_enc_shape), x_enc_shape.data(), x_enc_shape.size());
  auto x_mask_tensor =
      Ort::Value::CreateTensor<float>(mem, x_mask, product(x_mask_shape), x_mask_shape.data(), x_mask_shape.size());
  auto g_tensor = Ort::Value::CreateTensor<float>(mem, g, product(g_shape), g_shape.data(), g_shape.size());

  std::array<Ort::Value, 3> dp_inputs = {std::move(x_enc_tensor), std::move(x_mask_tensor), std::move(g_tensor)};
  auto dp_outputs = duration_predictor_.Run(Ort::RunOptions{nullptr}, dp_input_names, dp_inputs.data(), dp_inputs.size(),
                                            dp_output_names, 1);

  float* logw = dp_outputs[0].GetTensorMutableData<float>();
  auto logw_shape = shape_of(dp_outputs[0]);

  const auto tx = logw_shape.at(2);
  const auto channels = m_p_shape.at(1);

  std::vector<float> w(logw, logw + product(logw_shape));
  std::vector<float> w_ceil(w.size(), 0.0f);

  std::int64_t y_len = 0;
  for (std::int64_t i = 0; i < tx; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const float scaled = std::exp(w[idx]) * x_mask[idx] * input.length_scale;
    w_ceil[idx] = std::ceil(scaled);
    y_len += static_cast<std::int64_t>(w_ceil[idx]);
  }
  y_len = std::max<std::int64_t>(y_len, 1);

  std::vector<float> y_mask(static_cast<std::size_t>(y_len), 1.0f);
  std::vector<float> attn(static_cast<std::size_t>(y_len * tx), 0.0f);

  std::int64_t frame_start = 0;
  for (std::int64_t i = 0; i < tx; ++i) {
    const auto dur = static_cast<std::int64_t>(w_ceil[static_cast<std::size_t>(i)]);
    const std::int64_t frame_end = std::min<std::int64_t>(frame_start + dur, y_len);
    for (std::int64_t t = frame_start; t < frame_end; ++t) {
      attn[static_cast<std::size_t>(t * tx + i)] = y_mask[static_cast<std::size_t>(t)] * x_mask[static_cast<std::size_t>(i)];
    }
    frame_start += dur;
    if (frame_start >= y_len) {
      break;
    }
  }

  std::vector<float> m_p_exp(static_cast<std::size_t>(channels * y_len), 0.0f);
  std::vector<float> logs_p_exp(static_cast<std::size_t>(channels * y_len), 0.0f);

  for (std::int64_t c = 0; c < channels; ++c) {
    for (std::int64_t t = 0; t < y_len; ++t) {
      float m_sum = 0.0f;
      float logs_sum = 0.0f;
      for (std::int64_t x_idx = 0; x_idx < tx; ++x_idx) {
        const auto attn_value = attn[static_cast<std::size_t>(t * tx + x_idx)];
        const auto src_index = static_cast<std::size_t>(c * tx + x_idx);
        m_sum += attn_value * m_p[src_index];
        logs_sum += attn_value * logs_p[src_index];
      }
      const auto dst_index = static_cast<std::size_t>(c * y_len + t);
      m_p_exp[dst_index] = m_sum;
      logs_p_exp[dst_index] = logs_sum;
    }
  }

  std::normal_distribution<float> normal_dist(0.0f, 1.0f);
  std::vector<float> z_p(static_cast<std::size_t>(channels * y_len), 0.0f);
  for (std::size_t i = 0; i < z_p.size(); ++i) {
    z_p[i] = m_p_exp[i] + normal_dist(rng_) * std::exp(logs_p_exp[i]) * input.noise_scale;
  }

  std::array<std::int64_t, 3> z_p_shape{1, channels, y_len};
  std::array<std::int64_t, 3> y_mask_shape{1, 1, y_len};

  auto z_p_tensor = Ort::Value::CreateTensor<float>(mem, z_p.data(), z_p.size(), z_p_shape.data(), z_p_shape.size());
  auto y_mask_tensor =
      Ort::Value::CreateTensor<float>(mem, y_mask.data(), y_mask.size(), y_mask_shape.data(), y_mask_shape.size());
  auto g_flow_tensor = Ort::Value::CreateTensor<float>(mem, g, product(g_shape), g_shape.data(), g_shape.size());

  const char* flow_input_names[] = {"z_p", "y_mask", "g"};
  const char* flow_output_names[] = {"z"};
  std::array<Ort::Value, 3> flow_inputs = {std::move(z_p_tensor), std::move(y_mask_tensor), std::move(g_flow_tensor)};

  auto flow_outputs = flow_.Run(Ort::RunOptions{nullptr}, flow_input_names, flow_inputs.data(), flow_inputs.size(),
                                flow_output_names, 1);

  auto z_shape = shape_of(flow_outputs[0]);
  float* z_data = flow_outputs[0].GetTensorMutableData<float>();

  std::vector<float> z_masked(z_data, z_data + product(z_shape));
  for (std::int64_t c = 0; c < channels; ++c) {
    for (std::int64_t t = 0; t < y_len; ++t) {
      z_masked[static_cast<std::size_t>(c * y_len + t)] *= y_mask[static_cast<std::size_t>(t)];
    }
  }

  auto z_masked_tensor =
      Ort::Value::CreateTensor<float>(mem, z_masked.data(), z_masked.size(), z_shape.data(), z_shape.size());
  auto g_dec_tensor = Ort::Value::CreateTensor<float>(mem, g, product(g_shape), g_shape.data(), g_shape.size());

  const char* dec_input_names[] = {"z", "g"};
  const char* dec_output_names[] = {"audio"};
  std::array<Ort::Value, 2> dec_inputs = {std::move(z_masked_tensor), std::move(g_dec_tensor)};

  auto dec_outputs = decoder_.Run(Ort::RunOptions{nullptr}, dec_input_names, dec_inputs.data(), dec_inputs.size(),
                                  dec_output_names, 1);

  auto audio_shape = shape_of(dec_outputs[0]);
  float* audio_data = dec_outputs[0].GetTensorMutableData<float>();
  const auto sample_count = static_cast<std::size_t>(audio_shape.at(2));

  return std::vector<float>(audio_data, audio_data + sample_count);
}

}  // namespace tiny_tts
