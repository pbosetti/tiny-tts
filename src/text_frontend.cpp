#include "text_frontend.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tiny_tts {
namespace {

constexpr std::int64_t kLanguageId = 2;
constexpr std::int64_t kToneOffset = 7;

const std::vector<std::string> kSymbols = {
    "_", "\"", "(", ")", "*", "/", ":", "AA", "E", "EE", "En", "N", "OO", "Q", "V", "[", "\\", "]", "^", "a", "a:", "aa", "ae", "ah", "ai", "an", "ang", "ao", "aw", "ay", "b", "by", "c", "ch", "d", "dh", "dy", "e", "e:", "eh", "ei", "en", "eng", "er", "ey", "f", "g", "gy", "h", "hh", "hy", "i", "i0", "i:", "ia", "ian", "iang", "iao", "ie", "ih", "in", "ing", "iong", "ir", "iu", "iy", "j", "jh", "k", "ky", "l", "m", "my", "n", "ng", "ny", "o", "o:", "ong", "ou", "ow", "oy", "p", "py", "q", "r", "ry", "s", "sh", "t", "th", "ts", "ty", "u", "u:", "ua", "uai", "uan", "uang", "uh", "ui", "un", "uo", "uw", "v", "van", "ve", "vn", "w", "x", "y", "z", "zh", "zy", "~", "æ", "ç", "ð", "ø", "ŋ", "œ", "ɐ", "ɑ", "ɒ", "ɔ", "ɕ", "ə", "ɛ", "ɜ", "ɡ", "ɣ", "ɥ", "ɦ", "ɪ", "ɫ", "ɬ", "ɭ", "ɯ", "ɲ", "ɵ", "ɸ", "ɹ", "ɾ", "ʁ", "ʃ", "ʊ", "ʌ", "ʎ", "ʏ", "ʑ", "ʒ", "ʝ", "ʲ", "ʸ", "ˈ", "ˌ", "ː", "̃", "̩", "β", "θ", "ᄀ", "ᄁ", "ᄂ", "ᄃ", "ᄄ", "ᄅ", "ᄆ", "ᄇ", "ᄈ", "ᄉ", "ᄊ", "ᄋ", "ᄌ", "ᄍ", "ᄎ", "ᄏ", "ᄐ", "ᄑ", "ᄒ", "ᅡ", "ᅢ", "ᅣ", "ᅤ", "ᅥ", "ᅦ", "ᅧ", "ᅨ", "ᅩ", "ᅪ", "ᅫ", "ᅬ", "ᅭ", "ᅮ", "ᅯ", "ᅰ", "ᅱ", "ᅲ", "ᅳ", "ᅴ", "ᅵ", "ᆨ", "ᆫ", "ᆮ", "ᆯ", "ᆷ", "ᆸ", "ᆼ", "ㄸ", "!", "?", "…", ",", ".", "'", "-", "¿", "¡", "SP", "UNK"};

std::vector<std::string> split_words(const std::string& text) {
  std::istringstream iss(text);
  std::vector<std::string> out;
  std::string token;
  while (iss >> token) {
    out.push_back(token);
  }
  return out;
}

}  // namespace

TextFrontend::TextFrontend(const std::string& cmudict_path) {
  for (std::size_t i = 0; i < kSymbols.size(); ++i) {
    symbol_to_id_[kSymbols[i]] = static_cast<std::int64_t>(i);
  }
  unk_id_ = symbol_to_id_.at("UNK");
  load_cmudict(cmudict_path);
}

TextIds TextFrontend::text_to_ids(const std::string& text) const {
  std::vector<std::string> phones;
  std::vector<int> tones;

  const auto words = split_words(to_lower(trim(text)));
  for (const auto& word : words) {
    std::size_t lead = 0;
    while (lead < word.size() && !is_word_char(word[lead])) {
      const std::string mapped = map_phoneme(std::string(1, word[lead]));
      phones.push_back(mapped);
      tones.push_back(0);
      ++lead;
    }

    std::size_t tail = word.size();
    while (tail > lead && !is_word_char(word[tail - 1])) {
      --tail;
    }

    const std::string core = word.substr(lead, tail - lead);

    if (!core.empty()) {
      const auto resolved = resolve_word(core);
      if (resolved.empty()) {
        for (char c : core) {
          if (c == '\'') {
            continue;
          }
          phones.push_back(map_phoneme(std::string(1, static_cast<char>(std::tolower(static_cast<unsigned char>(c))))));
          tones.push_back(0);
        }
      } else {
        for (const auto& p : resolved) {
          auto parsed = parse_phone(p);
          phones.push_back(map_phoneme(parsed.first));
          tones.push_back(parsed.second);
        }
      }
    }

    for (std::size_t i = tail; i < word.size(); ++i) {
      const std::string mapped = map_phoneme(std::string(1, word[i]));
      phones.push_back(mapped);
      tones.push_back(0);
    }
  }

  phones.insert(phones.begin(), "_");
  phones.push_back("_");
  tones.insert(tones.begin(), 0);
  tones.push_back(0);

  TextIds ids;
  ids.phone_ids.reserve(phones.size() * 2 + 1);
  ids.tone_ids.reserve(tones.size() * 2 + 1);
  ids.lang_ids.reserve(phones.size() * 2 + 1);

  ids.phone_ids.push_back(0);
  ids.tone_ids.push_back(0);
  ids.lang_ids.push_back(0);

  for (std::size_t i = 0; i < phones.size(); ++i) {
    const auto it = symbol_to_id_.find(phones[i]);
    const std::int64_t phone_id = it == symbol_to_id_.end() ? unk_id_ : it->second;
    ids.phone_ids.push_back(phone_id);
    ids.tone_ids.push_back(static_cast<std::int64_t>(tones[i]) + kToneOffset);
    ids.lang_ids.push_back(kLanguageId);

    ids.phone_ids.push_back(0);
    ids.tone_ids.push_back(0);
    ids.lang_ids.push_back(0);
  }

  return ids;
}

std::string TextFrontend::to_lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

std::string TextFrontend::to_upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

std::string TextFrontend::trim(const std::string& value) {
  std::size_t start = 0;
  while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
    ++start;
  }
  std::size_t end = value.size();
  while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(start, end - start);
}

bool TextFrontend::is_word_char(char c) {
  const unsigned char uc = static_cast<unsigned char>(c);
  return std::isalnum(uc) || c == '\'';
}

std::pair<std::string, int> TextFrontend::parse_phone(const std::string& phone) {
  if (!phone.empty() && std::isdigit(static_cast<unsigned char>(phone.back()))) {
    return {to_lower(phone.substr(0, phone.size() - 1)), (phone.back() - '0') + 1};
  }
  return {to_lower(phone), 0};
}

std::vector<std::string> TextFrontend::resolve_word(const std::string& word) const {
  const std::string upper_word = to_upper(word);
  const auto it = cmu_dict_.find(upper_word);
  if (it != cmu_dict_.end()) {
    return it->second;
  }

  if (word.find('\'') != std::string::npos) {
    std::vector<std::string> merged;
    std::size_t pos = 0;
    bool ok = true;
    bool first = true;
    while (pos <= word.size()) {
      std::size_t next = word.find('\'', pos);
      std::string part = next == std::string::npos ? word.substr(pos) : word.substr(pos, next - pos);
      if (!part.empty()) {
        auto pit = cmu_dict_.find(to_upper(part));
        if (pit == cmu_dict_.end()) {
          ok = false;
          break;
        }
        if (!first) {
          merged.push_back("'");
        }
        merged.insert(merged.end(), pit->second.begin(), pit->second.end());
        first = false;
      }
      if (next == std::string::npos) {
        break;
      }
      pos = next + 1;
    }
    if (ok && !merged.empty()) {
      return merged;
    }
  }

  return {};
}

void TextFrontend::load_cmudict(const std::string& cmudict_path) {
  std::string path = cmudict_path;
  if (path.empty()) {
    path = "resources/cmudict.rep";
  }

  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Unable to open CMU dictionary: " + path);
  }

  std::string line;
  std::size_t line_num = 0;
  while (std::getline(in, line)) {
    ++line_num;
    if (line_num < 49 || line.empty()) {
      continue;
    }

    const std::size_t sep = line.find("  ");
    if (sep == std::string::npos) {
      continue;
    }

    std::string word = line.substr(0, sep);
    std::string pronunciation = line.substr(sep + 2);
    auto variant_pos = word.find('(');
    if (variant_pos != std::string::npos) {
      word = word.substr(0, variant_pos);
    }

    if (cmu_dict_.contains(word)) {
      continue;
    }

    std::vector<std::string> phones;
    std::size_t start = 0;
    while (start < pronunciation.size()) {
      std::size_t syllable_end = pronunciation.find(" - ", start);
      std::string syllable = syllable_end == std::string::npos
                                 ? pronunciation.substr(start)
                                 : pronunciation.substr(start, syllable_end - start);
      std::istringstream iss(syllable);
      std::string ph;
      while (iss >> ph) {
        phones.push_back(ph);
      }

      if (syllable_end == std::string::npos) {
        break;
      }
      start = syllable_end + 3;
    }

    if (!phones.empty()) {
      cmu_dict_[word] = std::move(phones);
    }
  }

  if (cmu_dict_.empty()) {
    throw std::runtime_error("CMU dictionary is empty: " + path);
  }
}

std::string TextFrontend::map_phoneme(const std::string& phoneme) const {
  static const std::unordered_map<std::string, std::string> replacements = {
      {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."}, {"！", "!"},
      {"？", "?"}, {"\n", "."}, {"·", ","}, {"、", ","}, {"...", "…"},
      {"v", "V"}};

  auto it = replacements.find(phoneme);
  const std::string mapped = it == replacements.end() ? phoneme : it->second;
  return symbol_to_id_.contains(mapped) ? mapped : "UNK";
}

}  // namespace tiny_tts
