import re

modes = {"prosody": "p",
         "vocalization": "v"}

situation_pattern = re.compile(r'^sit\d+$')

# Compile the regex pattern
version_pattern = re.compile(r'^ver\d+$')

video_id_pattern = re.compile(r'^[A-Z]\d{1,4}$')

intensity_levels = {"below_medium": "1",
                    "medium": "2",
                    "high": "3",
                    "extremely_high": "4"
                    }

long_emotion_names = {"positive_surprise": ["pos", "sur"],
                      "negative_surprise": ["neg", "sur"]}


error = "e"

mix = "mix"

proportion_component_patterns = ["30", "50", "70"]
proportion_patterns = ["3070", "5050", "7030"]

neu = "neu"

