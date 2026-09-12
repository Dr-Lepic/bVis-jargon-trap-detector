"""Generate a small, synthetic dataset for exercising the batch workflow."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


CASES = [
    ("darier", "Darier's Disease", "Lichen Planus", "Clusters of tan-brown keratotic papules with a greasy surface are visible in a seborrhoeic distribution."),
    ("nevus", "Melanocytic Nevus", "Melanoma", "The lesion is symmetric and uniformly brown, with a smooth regular border and homogeneous pigmentation."),
    ("psoriasis", "Plaque Psoriasis", "Tinea Corporis", "A sharply defined erythematous plaque has diffuse adherent silvery scale across its surface."),
    ("eczema", "Nummular Eczema", "Basal Cell Carcinoma", "A coin-shaped erythematous patch shows fine scale and mild excoriation without a pearly border."),
    ("acne", "Acne Vulgaris", "Rosacea", "Open and closed comedones occur alongside scattered inflammatory papules on the facial skin."),
    ("vitiligo", "Vitiligo", "Tinea Versicolor", "A well-demarcated depigmented patch is present without scale, erythema, or surface textural change."),
]


def create_samples(output_dir: Path | None = None) -> Path:
    root = output_dir or Path(__file__).resolve().parent
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, (slug, true_name, wrong_name, report) in enumerate(CASES, start=1):
        filename = f"{slug}.png"
        image = Image.new("RGB", (640, 480), (235, 200 - index * 7, 180 + index * 5))
        draw = ImageDraw.Draw(image)
        draw.ellipse((175, 95, 465, 385), fill=(150 + index * 5, 78, 70), outline=(90, 45, 40), width=7)
        draw.text((18, 18), f"Synthetic demo: {true_name}", fill=(30, 30, 30))
        image.save(images / filename)
        rows.append({
            "case_id": f"sample-{index:02d}", "image": filename,
            "disease_true": true_name, "disease_wrong": wrong_name,
            "report_plain": report, "report_jargon": "",
        })
    csv_path = root / "sample_cases.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    print(create_samples())
