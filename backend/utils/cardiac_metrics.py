import numpy as np

# ==========================================
# User Provided Pipeline Logic
# ==========================================
def compute_area(mask):
    """
    mask: binary numpy array (1 = LV, 0 = background)
    returns: area in pixels
    """
    return np.sum(mask)

def area_to_volume(area, length):
    """
    Area-Length method:
    Volume = (8 * A^2) / (3 * pi * L)
    
    area: LV area
    length: LV long-axis length
    """
    return (8 * (area ** 2)) / (3 * np.pi * length)

def compute_stroke_volume(segmentation_masks, lengths):
    """
    segmentation_masks: list of 2D binary masks (frames)
    lengths: list of LV lengths per frame
    
    returns: stroke volume
    """
    areas = [compute_area(mask) for mask in segmentation_masks]

    # Find EDV (max area) and ESV (min area)
    ed_idx = np.argmax(areas)
    es_idx = np.argmin(areas)

    ed_area = areas[ed_idx]
    es_area = areas[es_idx]

    ed_length = lengths[ed_idx]
    es_length = lengths[es_idx]

    # Convert to volumes
    EDV = area_to_volume(ed_area, ed_length)
    ESV = area_to_volume(es_area, es_length)

    # Stroke Volume
    SV = EDV - ESV

    return SV, EDV, ESV

# ==========================================
# Integration & Physical Calibration
# ==========================================
def compute_lv_length(mask: np.ndarray) -> float:
    """
    Compute LV long-axis length in pixels by finding the vertical span.
    """
    rows_occupied = np.any(mask, axis=1)
    row_indices = np.where(rows_occupied)[0]
    if len(row_indices) < 2:
        return max(float(np.sqrt(np.sum(mask))), 1.0)
    return float(row_indices[-1] - row_indices[0])

def analyze_volumes_ml(segmentation_masks: np.ndarray) -> tuple[float, float, float]:
    """
    Uses the user-provided compute_stroke_volume on binarized masks
    and translates the resulting pixel^3 volumes to milliliters (cm^3).
    """
    # EchoNet standard view covers ~16cm across 112px
    FOV_CM = 16.0
    IMAGE_SIZE_PX = 112
    PIXEL_CM = FOV_CM / IMAGE_SIZE_PX
    PIXELS_TO_ML = PIXEL_CM ** 3

    # Binarize masks (threshold 0.5)
    binary_masks = [m > 0.5 for m in segmentation_masks]
    
    # Calculate lengths in pixels for each frame
    lengths_px = [compute_lv_length(m) for m in binary_masks]
    
    # Compute using the user's logic
    sv_px3, edv_px3, esv_px3 = compute_stroke_volume(binary_masks, lengths_px)
    
    # Convert pixels^3 to mL (cm^3)
    sv_ml = max(sv_px3 * PIXELS_TO_ML, 0.0)
    edv_ml = edv_px3 * PIXELS_TO_ML
    esv_ml = esv_px3 * PIXELS_TO_ML
    
    return round(sv_ml, 1), round(edv_ml, 1), round(esv_ml, 1)
