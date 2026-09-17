"""Prepare smaller texture copies for the iOS resource bundle."""
from pathlib import Path
import shutil

from PIL import Image


def copy_asset(source: Path, target: Path, max_dimension: int):
    """Copy an asset, limiting PNG/JPEG dimensions without changing its path."""
    if source.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
        shutil.copy2(source, target)
        return None

    with Image.open(source) as image:
        original = image.size
        if max_dimension == 0 or max(original) <= max_dimension:
            shutil.copy2(source, target)
            packaged = original
        else:
            scale = max_dimension / max(original)
            packaged = tuple(max(1, round(dimension * scale)) for dimension in original)
            # Resample stored channels directly; normal/roughness maps must not
            # undergo a color-space conversion. Expand palette transparency.
            pixels = image
            if 'transparency' in image.info:
                pixels = image.convert('RGBA')
            elif image.mode == 'P':
                pixels = image.convert('RGB')
            pixels = pixels.resize(packaged, Image.Resampling.LANCZOS)
            options = {'icc_profile': image.info['icc_profile']} if 'icc_profile' in image.info else {}
            if image.format == 'JPEG':
                options.update(quality=95, subsampling=0)
            elif image.format == 'PNG':
                options['optimize'] = True
            pixels.save(target, format=image.format, **options)

    return {'original_size': list(original), 'packaged_size': list(packaged),
            'original_rgba8_bytes': original[0] * original[1] * 4,
            'packaged_rgba8_bytes': packaged[0] * packaged[1] * 4}
