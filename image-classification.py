!pip install -U transformers
!pip install -U sentencepiece
!pip install -U sacremoses


"""## Image Classification pipeline"""

from PIL import Image
import requests
from transformers import pipeline

url = "https://media.istockphoto.com/id/992637094/fr/photo/chat-british-poil-court-et-golden-retriever.jpg?s=2048x2048&w=is&k=20&c=caKnidFBwH0Fct3MA8Pv3CVBWjcSOrniuQKVgCgyT20="

image = Image.open(requests.get(url, stream=True).raw)

classifier = pipeline("image-classification")

output = classifier(image)
output

url = "https://imgs.search.brave.com/ZSTgS0eaNczlmky2u5Jh7rWlO4Ir0MGJV8DQsT1zr5U/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9zdGF0/aWMudmVjdGVlenku/Y29tL3N5c3RlbS9y/ZXNvdXJjZXMvdGh1/bWJuYWlscy8wMDMv/MzYzLzQ0MS9zbWFs/bC9wb3J0cmFpdC1v/Zi1oYW5kc29tZS1n/dXktc21pbGluZy1p/bi10aGUtYXV0dW1u/LXBhcmstZnJlZS1w/aG90by5qcGc"

image = Image.open(requests.get(url, stream=True).raw)

classifier = pipeline("image-classification", model ="nateraw/vit-age-classifier")

output = classifier(image)
output

