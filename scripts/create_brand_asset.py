"""Render the small document-shaped brand mark as a local bitmap asset."""

from pathlib import Path

from PIL import Image, ImageDraw


def main():
    root = Path(__file__).resolve().parents[1]
    image = Image.new("RGB", (160, 160), "#e5f3ef")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 25, 124, 136), radius=8, fill="#087f78")
    draw.rectangle((53, 49, 105, 56), fill="#ffffff")
    draw.rectangle((53, 70, 94, 77), fill="#bde0d3")
    draw.rectangle((53, 91, 83, 98), fill="#bde0d3")
    draw.ellipse((99, 104, 131, 136), fill="#df947b")
    image.save(root / "web/mark.png")


if __name__ == "__main__":
    main()
