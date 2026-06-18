
from docling.document_converter import DocumentConverter

converter = DocumentConverter()

result = converter.convert(
    "pdfs/Cir_2026_01_fr.pdf"
)

markdown = result.document.export_to_markdown()

with open("output.md","w",encoding="utf-8") as f:
    f.write(markdown)

print("Done")