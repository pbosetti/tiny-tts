#include <cstdlib>
#include <string>

#include "text_frontend.hpp"

int main() {
  tiny_tts::TextFrontend frontend(std::string(TINY_TTS_SOURCE_DIR) + "/resources/cmudict.rep");
  const auto ids = frontend.text_to_ids("Hello world!");
  if (ids.phone_ids.empty() || ids.tone_ids.size() != ids.phone_ids.size() ||
      ids.lang_ids.size() != ids.phone_ids.size()) {
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
