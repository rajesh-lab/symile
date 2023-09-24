LANGUAGES = ["arabic", "english", "greek", "hindi", "japanese"]

# ISO-639 codes for Google Translate API
# see https://g.co/cloud/translate/v2/translate-reference#supported_languages
ISOCODES = ["ar", "en", "el", "hi", "ja"]

# language codes and voice names for Google Text-to-Speech API
# see https://cloud.google.com/text-to-speech/docs/voices
ISO2LANGCODE = {"ar": "ar-XA",
                "en": "en-US",
                "el": "el-GR",
                "hi": "hi-IN",
                "ja": "ja-JP"}
ISO2VOICES = {
    "ar": ["ar-XA-Standard-A", "ar-XA-Standard-B", "ar-XA-Standard-C", "ar-XA-Standard-D"],
    "en": ["en-US-Standard-A", "en-US-Standard-B", "en-US-Standard-C", "en-US-Standard-D"],
    "el": ["el-GR-Standard-A"],
    "hi": ["hi-IN-Standard-A", "hi-IN-Standard-B", "hi-IN-Standard-C", "hi-IN-Standard-D"],
    "ja": ["ja-JP-Standard-A", "ja-JP-Standard-B", "ja-JP-Standard-C", "ja-JP-Standard-D"]
}

ISO2FLAGFILE = {"ar": "lebanon.png",
                "en": "usa.png",
                "el": "greece.png",
                "hi": "india.png",
                "ja": "japan.png"}

FLAGFILE2ISO = {"lebanon.png": "ar",
                "usa.png": "en",
                "greece.png": "el",
                "india.png": "hi",
                "japan.png": "ja"}