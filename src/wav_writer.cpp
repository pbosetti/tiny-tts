#include "wav_writer.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace tiny_tts {

namespace {

void write_u16(std::ofstream& out, std::uint16_t value) {
  out.put(static_cast<char>(value & 0xFF));
  out.put(static_cast<char>((value >> 8) & 0xFF));
}

void write_u32(std::ofstream& out, std::uint32_t value) {
  out.put(static_cast<char>(value & 0xFF));
  out.put(static_cast<char>((value >> 8) & 0xFF));
  out.put(static_cast<char>((value >> 16) & 0xFF));
  out.put(static_cast<char>((value >> 24) & 0xFF));
}

}  // namespace

void write_wav_16bit(const std::string& output_path, const std::vector<float>& audio,
                     int sample_rate) {
  std::ofstream out(output_path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("Unable to open output file: " + output_path);
  }

  const std::uint16_t channels = 1;
  const std::uint16_t bits_per_sample = 16;
  const std::uint16_t block_align = channels * bits_per_sample / 8;
  const std::uint32_t byte_rate = static_cast<std::uint32_t>(sample_rate) * block_align;
  const std::uint32_t data_size = static_cast<std::uint32_t>(audio.size() * block_align);

  out.write("RIFF", 4);
  write_u32(out, 36u + data_size);
  out.write("WAVE", 4);

  out.write("fmt ", 4);
  write_u32(out, 16);
  write_u16(out, 1);
  write_u16(out, channels);
  write_u32(out, static_cast<std::uint32_t>(sample_rate));
  write_u32(out, byte_rate);
  write_u16(out, block_align);
  write_u16(out, bits_per_sample);

  out.write("data", 4);
  write_u32(out, data_size);

  for (float sample : audio) {
    const float clipped = std::clamp(sample, -1.0f, 1.0f);
    const auto pcm = static_cast<std::int16_t>(clipped * 32767.0f);
    write_u16(out, static_cast<std::uint16_t>(pcm));
  }
}

}  // namespace tiny_tts
