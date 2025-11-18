
from transformers import pipeline
import pandas as pd


translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-de")

text = """
Dear Amazon, last week I ordered an Optimus Prime action figure from your
online store in India. Unfortunately when I opened the package, I discovered to
my horror that I had been sent an action figure of Megatron instead!
"""


outputs = translator(text)
outputs

[{'translation_text': 'Sehr geehrter Amazon, letzte Woche habe ich eine Optimus Prime Action Figur in Ihrem Online-Shop in Indien bestellt. Leider als ich das Paket öffnete, entdeckte ich zu meinem Entsetzen, dass ich stattdessen eine Action Figur von Megatron geschickt worden war!'}]
