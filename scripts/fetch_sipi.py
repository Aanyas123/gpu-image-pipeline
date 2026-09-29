"""Downloads a few classic test images from the USC-SIPI Image Database.

Source: https://sipi.usc.edu/database/database.php (the "Miscellaneous"
volume). The images are converted to PNG so every tool in the pipeline can
read them. Run manually; the pipeline also works on the synthetic set from
generate_images.py if you are offline.

Usage:
    python scripts/fetch_sipi.py --output data/input
"""

import argparse
import io
import os
import urllib.request

from PIL import Image

BASE_URL = "https://sipi.usc.edu/database/download.php?vol=misc&img="
# SIPI ids from the Miscellaneous volume (colour and grayscale classics).
IMAGE_IDS = {
    "4.1.05": "house",
    "4.2.03": "mandrill",
    "4.2.06": "sailboat",
    "4.2.07": "peppers",
    "5.3.01": "man_1024",
    "5.3.02": "airport_1024",
    "7.1.01": "truck",
    "boat.512": "boat",
    "house": "house_512",
    "gray21.512": "gray21",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="data/input")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    for image_id, name in IMAGE_IDS.items():
        url = BASE_URL + image_id
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                image = Image.open(io.BytesIO(response.read()))
                image.load()
        except (OSError, ValueError) as exc:
            print(f"skipped {image_id}: {exc}")
            continue
        path = os.path.join(args.output, f"sipi_{name}.png")
        image.convert("RGB").save(path)
        print(f"wrote {path} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
