LANGUAGES = ["arabic", "english", "greek", "hindi", "japanese"]

# ISO-639 codes for Google Translate API
# see https://g.co/cloud/translate/v2/translate-reference#supported_languages
LANG2ISOCODE = {"arabic": "ar",
                "english": "en",
                "greek": "el",
                "hindi": "hi",
                "japanese": "ja"}

# language codes and voice names for Google Text-to-Speech API
# see https://cloud.google.com/text-to-speech/docs/voices
LANG2LANGCODE = {"arabic": "ar-XA",
                 "english": "en-US",
                 "greek": "el-GR",
                 "hindi": "hi-IN",
                 "japanese": "ja-JP"}
LANG2VOICES = {
    "arabic": ["ar-XA-Standard-A", "ar-XA-Standard-B", "ar-XA-Standard-C", "ar-XA-Standard-D"],
    "english": ["en-US-Standard-A", "en-US-Standard-B", "en-US-Standard-C", "en-US-Standard-D"],
    "greek": ["el-GR-Standard-A"],
    "hindi": ["hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D"],
    "japanese": ["ja-JP-Standard-A", "ja-JP-Standard-B", "ja-JP-Standard-C", "ja-JP-Standard-D"]
}

LANG2FLAGFILE = {"arabic": "lebanon.png",
                 "english": "usa.png",
                 "greek": "greece.png",
                 "hindi": "india.png",
                 "japanese": "japan.png"}