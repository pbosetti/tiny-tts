#pragma once

#include <string>
#include <vector>

namespace tiny_tts {

void write_wav_16bit(const std::string& output_path, const std::vector<float>& audio,
                     int sample_rate = 44100);

}  // namespace tiny_tts
