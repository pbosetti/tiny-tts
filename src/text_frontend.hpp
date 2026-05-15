#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tiny_tts {

struct TextIds {
  std::vector<std::int64_t> phone_ids;
  std::vector<std::int64_t> tone_ids;
  std::vector<std::int64_t> lang_ids;
};

class TextFrontend {
 public:
  explicit TextFrontend(const std::string& cmudict_path = {});

  TextIds text_to_ids(const std::string& text) const;

 private:
  using Pronunciation = std::vector<std::string>;

  static std::string to_lower(std::string value);
  static std::string to_upper(std::string value);
  static std::string trim(const std::string& value);
  static bool is_word_char(char c);

  static std::pair<std::string, int> parse_phone(const std::string& phone);
  std::vector<std::string> resolve_word(const std::string& word) const;

  void load_cmudict(const std::string& cmudict_path);
  std::string map_phoneme(const std::string& phoneme) const;

  std::unordered_map<std::string, Pronunciation> cmu_dict_;
  std::unordered_map<std::string, std::int64_t> symbol_to_id_;
  std::int64_t unk_id_{0};
};

}  // namespace tiny_tts
