import math

def scale_nutrients(nutrients, from_grams, to_grams):
    """
    Rescales an NLP nutrients block (computed against the resolved grams from DB) to the photo-estimated mass (g),
    so the app shows nutrition for what the geometry actually measured rather than what the text implied
    """
    if not nutrients or not from_grams:
        return None

    factor = to_grams / from_grams
    return {
        nutrient: round(value * factor, 2) if value is not None else None
        for nutrient, value in nutrients.items()
    }


def volume_to_mass(volume_ml, density_block):
    """Converts a volume estimate to a mass estimate using a matched food's density block"""
    if volume_ml is None or volume_ml <= 0 or not density_block:
        return None

    density = density_block["density_g_per_ml"]
    log_sigma = density_block["density_log_sigma"]
    mass_g = volume_ml * density

    return {
        "mass_g": round(mass_g, 1),
        "mass_low_g": round(mass_g * math.exp(-1.96 * log_sigma), 1),
        "mass_high_g": round(mass_g * math.exp(1.96 * log_sigma), 1),
        "density_g_per_ml": density,
        "density_source": density_block.get("density_source"),
        "presentation": density_block.get("presentation")
    }