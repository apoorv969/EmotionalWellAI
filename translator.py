from googletrans import Translator

translator = Translator()

def translate_to_english(text, lang):
    if lang == "en":
        return text
    try:
        return translator.translate(text, src=lang, dest="en").text
    except:
        return text