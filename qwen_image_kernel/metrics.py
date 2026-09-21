# PSNR between two images (float RGB in [0, 1]).
import numpy as np


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)
    b = b.astype(np.float64) / 255.0 if b.dtype == np.uint8 else b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def image_psnr(path_a: str, path_b: str) -> float:
    from PIL import Image

    a = np.asarray(Image.open(path_a).convert("RGB"))
    b = np.asarray(Image.open(path_b).convert("RGB"))
    if a.shape != b.shape:
        from PIL import Image as I

        b = np.asarray(Image.open(path_b).convert("RGB").resize((a.shape[1], a.shape[0])))
    return psnr(a, b)