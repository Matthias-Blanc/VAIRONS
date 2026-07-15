import os
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from skimage import measure, morphology, segmentation
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, uniform_filter1d
from scipy.signal import find_peaks as scipy_find_peaks
from skimage.feature import peak_local_max, graycomatrix, graycoprops
from cellpose import models as cp_models, version as cellpose_version


# =============================================================================
# CPU PARALLELISM
# =============================================================================
#
# The per-region feature-extraction and stain-quantification loops are the CPU-bound
# part of the pipeline, and their heavy work (regionprops, GLCM, marching cubes,
# distance transforms, connected components) runs in skimage/scipy C code that
# RELEASES the GIL -- so a thread pool gives real speedup without pickling or copying
# the shared read-only image arrays. Cellpose/torch is deliberately left out (it runs
# on the GPU, or manages its own threads). The worker count is a process-wide setting
# so the many helper functions don't each need a threads argument.

_NUM_WORKERS = 1


def resolve_worker_count(n=None):
    """Effective CPU worker count: an explicit ``n`` (>=1), else all detected logical
    cores **minus one** (leaving one free for the OS/UI), floored at 1."""
    if n:
        return max(1, int(n))
    return max(1, (os.cpu_count() or 1) - 1)


def _configure_native_threads(n):
    """Point the native compute libraries at ``n`` threads as well, so the CPU work
    that does NOT go through our thread pool -- chiefly Cellpose/torch tensor ops and
    BLAS-backed numpy -- uses the cores too instead of a default subset. Best-effort:
    ``torch.set_num_threads`` takes effect at runtime; the env vars help any pool that
    initialises later; ``threadpoolctl`` (if installed) caps already-loaded BLAS."""
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = str(n)
    try:
        import torch
        torch.set_num_threads(int(n))
    except Exception:
        pass
    global _THREADPOOL_LIMIT
    try:
        from threadpoolctl import threadpool_limits
        _THREADPOOL_LIMIT = threadpool_limits(limits=int(n))   # held so it stays applied
    except Exception:
        _THREADPOOL_LIMIT = None


_THREADPOOL_LIMIT = None


def set_worker_count(n):
    """Set the process-wide worker count for the parallel CPU loops -- and the native
    (torch / BLAS) thread count to match -- and return the resolved value. Call once at
    the start of an analysis (``n=None`` -> cores-1)."""
    global _NUM_WORKERS
    _NUM_WORKERS = resolve_worker_count(n)
    _configure_native_threads(_NUM_WORKERS)
    return _NUM_WORKERS


def thread_report():
    """One-line summary of the effective thread settings, for run-start logging."""
    parts = [f"pool={_NUM_WORKERS}"]
    try:
        import torch
        parts.append(f"torch={torch.get_num_threads()}")
        dev = detect_device()
        parts.append(f"device={dev[0]}")
    except Exception:
        pass
    return "CPU threads -> " + ", ".join(parts) + f" (of {os.cpu_count() or '?'} logical cores)"


def _parallel_map(fn, items, workers=None):
    """``[fn(x) for x in items]``, spread over a thread pool when it pays off.

    Threads (not processes) because the loop bodies spend their time in GIL-releasing
    C extensions, so the shared image arrays stay shared and nothing is pickled.
    Order is preserved. Falls back to a plain loop for <=1 worker or trivially small
    inputs (thread setup would cost more than it saves).
    """
    items = list(items)
    w = workers or _NUM_WORKERS
    if w <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(fn, items))


# =============================================================================
# NUCLEAR HELPERS
# =============================================================================

def _major_axis_director(region, spacing=None):
    """Unit vector along a 3D region's longest principal axis.

    With ``spacing`` = (z_um, y_um, x_um) the director is taken from physically
    scaled coordinates (anisotropy-correct, ``_principal_axes_3d``); without it,
    falls back to the raw inertia tensor in voxel units. Returns None if unavailable.
    """
    try:
        if spacing is not None:
            _, evecs = _principal_axes_3d(region, spacing)
            return evecs[:, -1]                           # largest eigenvalue -> long axis
        _, vecs = np.linalg.eigh(region.inertia_tensor)   # ascending eigenvalues
        return vecs[:, 0]
    except Exception:
        return None


def _nematic_order_3d(directors):
    """Scalar nematic order parameter S in [0,1] for a set of 3D unit directors.

    S = largest eigenvalue of Q = <(3/2) n(x)n - (1/2) I>; S=1 perfectly aligned,
    S->0 isotropic. Axial (sign-invariant) -- the 3D analog of the 2D axial order
    parameter used in the 2D path.
    """
    dirs = [np.asarray(n, dtype=float) for n in directors if n is not None]
    if len(dirs) < 2:
        return np.nan
    Q = np.zeros((3, 3))
    for n in dirs:
        nn = n / (np.linalg.norm(n) or 1.0)
        Q += 1.5 * np.outer(nn, nn) - 0.5 * np.eye(3)
    Q /= len(dirs)
    return float(np.linalg.eigvalsh(Q).max())


def compute_nuclear_alignment(labeled_image, spacing=None):
    regions = measure.regionprops(labeled_image)
    if len(regions) < 2:
        return np.nan
    if labeled_image.ndim != 2:
        # 3D nematic order parameter from each nucleus's principal-axis director;
        # ``spacing`` = (z_um, y_um, x_um) makes the directors anisotropy-correct.
        return _nematic_order_3d([_major_axis_director(r, spacing) for r in regions])
    orientations = np.array([r.orientation for r in regions])
    doubled = 2 * orientations
    return np.sqrt(np.mean(np.cos(doubled))**2 + np.mean(np.sin(doubled))**2)


def compute_convex_hull_features(region):
    """Solidity and convexity-defect count, guarded against degenerate hulls.

    Both quantities need the object's convex hull. On thin / tiny / effectively
    lower-dimensional objects (common for stain particles) qhull cannot build a
    hull: skimage emits a repeated ``QH6013 ... input is less than N-dimensional``
    UserWarning and returns an *empty* convex image, which would make ``solidity``
    infinite/garbage. Shape is meaningless for such objects (see README), so we
    silence that noise and return ``solidity = NaN`` / ``convexity_defects = 0``
    instead of polluting the per-cell sheets.
    """
    image_filled = region.image_filled
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # swallow qhull's QH6013 storm
        image_convex = region.image_convex
    area_convex = int(image_convex.sum())
    # A valid hull always encloses the filled object; a smaller/empty convex image
    # means qhull failed -> degenerate object, shape undefined.
    if area_convex < int(image_filled.sum()):
        return np.nan, 0
    solidity = region.area / area_convex if area_convex > 0 else np.nan
    defects = image_convex.astype(int) - image_filled.astype(int)
    defects[defects < 0] = 0
    return solidity, int(measure.label(defects).max())


def _texture_features_3d(region, intensity_image):
    """3D texture = mean of the per-z-slice 2D GLCM descriptors over the object.

    skimage's graycomatrix is 2D-only, so a true 3D GLCM is unavailable; averaging
    the four Haralick descriptors across the object's z-slices is the closest
    counterpart (plan D4). Slices with <2 gray levels or too few pixels are skipped.
    """
    zmin, ymin, xmin, zmax, ymax, xmax = region.bbox
    sub = intensity_image[zmin:zmax, ymin:ymax, xmin:xmax]
    mask = region.image_filled                       # (Z,Y,X) bool, matches bbox
    keys = ('texture_contrast', 'texture_homogeneity',
            'texture_energy', 'texture_correlation')
    acc = {k: [] for k in keys}
    for z in range(sub.shape[0]):
        sl, m = sub[z], mask[z]
        if sl.size < 4 or min(sl.shape) < 2:
            continue
        vals = sl[m]
        if vals.size == 0:
            continue
        vr = vals.max() - vals.min()
        if vr == 0:
            continue
        plane = sl.astype(np.float64).copy()
        plane[~m] = vals.mean()
        plane8 = ((plane - vals.min()) / vr * 255).astype(np.uint8)
        glcm = graycomatrix(plane8, distances=[1],
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        acc['texture_contrast'].append(graycoprops(glcm, 'contrast').mean())
        acc['texture_homogeneity'].append(graycoprops(glcm, 'homogeneity').mean())
        acc['texture_energy'].append(graycoprops(glcm, 'energy').mean())
        acc['texture_correlation'].append(graycoprops(glcm, 'correlation').mean())
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in acc.items()}


def compute_texture_features(region, intensity_image):
    nan_result = {
        'texture_contrast': np.nan, 'texture_homogeneity': np.nan,
        'texture_energy': np.nan, 'texture_correlation': np.nan,
    }
    if intensity_image.ndim == 3:
        return _texture_features_3d(region, intensity_image)
    min_row, min_col, max_row, max_col = region.bbox
    bbox_intensity = intensity_image[min_row:max_row, min_col:max_col].copy()
    mask = region.image_filled
    if bbox_intensity.size < 4 or min(bbox_intensity.shape) < 2:
        return nan_result
    masked_values = bbox_intensity[mask]
    if len(masked_values) == 0:
        return nan_result
    val_range = masked_values.max() - masked_values.min()
    if val_range == 0:
        return nan_result
    bbox_intensity[~mask] = masked_values.mean()
    bbox_8bit = ((bbox_intensity - masked_values.min()) / val_range * 255).astype(np.uint8)
    glcm = graycomatrix(bbox_8bit, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    return {
        'texture_contrast': graycoprops(glcm, 'contrast').mean(),
        'texture_homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'texture_energy': graycoprops(glcm, 'energy').mean(),
        'texture_correlation': graycoprops(glcm, 'correlation').mean(),
    }


def compute_intensity_statistics(intensity_values):
    return {
        'intensity_mean': np.mean(intensity_values),
        'intensity_sd': np.std(intensity_values),
        'intensity_median': np.median(intensity_values),
        'intensity_p10': np.percentile(intensity_values, 10),
        'intensity_p90': np.percentile(intensity_values, 90),
        'integrated_intensity': np.sum(intensity_values),
    }


def compute_doublet_score(region, median_area):
    circularity = 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
    return min(1.0,
        0.25 * (1 - circularity) +
        0.25 * (1 - region.solidity) +
        0.25 * region.eccentricity +
        0.25 * min(1.0, max(0, region.area / median_area - 1))
    )


def split_doublet(object_mask, min_distance=80):
    distance = ndimage.distance_transform_edt(object_mask)
    local_maxima = peak_local_max(distance, min_distance=min_distance,
                                   labels=object_mask, footprint=np.ones((3, 3)))
    if len(local_maxima) < 2:
        return None
    markers = np.zeros_like(object_mask, dtype=int)
    for idx, coord in enumerate(local_maxima, start=1):
        markers[coord[0], coord[1]] = idx
    markers = morphology.dilation(markers, morphology.disk(2))
    return segmentation.watershed(-distance, markers, mask=object_mask)


# =============================================================================
# NUCLEAR SEGMENTATION
# =============================================================================

def _touches_border(coords, shape, margin):
    """True if an object reaches within `margin` of the lateral (Y,X) frame.

    Only the two trailing axes are tested: in 2D those are the whole image (so
    this is exactly the original row/col edge test); in 3D `(Z,Y,X)` the Z axis is
    intentionally excluded, because nuclei routinely span the full stack depth and
    legitimately touch the top/bottom planes. No-op when margin <= 0.
    """
    if margin <= 0:
        return False
    for axis in range(coords.shape[1] - 2, coords.shape[1]):   # lateral Y,X only
        c = coords[:, axis]
        if np.any((c < margin) | (c >= shape[axis] - margin)):
            return True
    return False


def segment_nuclei(detection_image, config):
    if config.get('nuclear_method', 'threshold') == 'cellpose':
        return segment_nuclei_cellpose(detection_image, config)
    threshold_min = config['threshold_min']
    threshold_max = config['threshold_max']
    min_object_size = config['min_object_size']
    connectivity = config['connectivity']
    edge_margin = config['edge_exclusion_margin']
    doublet_threshold = config['doublet_threshold']
    ws_min_distance = config['watershed_min_distance']

    binary_mask = (detection_image >= threshold_min) & (detection_image <= threshold_max)
    binary_mask = binary_fill_holes(binary_mask)
    labeled_image = measure.label(binary_mask, connectivity=connectivity)
    regions = measure.regionprops(labeled_image, intensity_image=detection_image)
    shape = detection_image.shape

    initial_valid = []
    for region in regions:
        if region.area < min_object_size:
            continue
        if _touches_border(region.coords, shape, edge_margin):
            continue
        initial_valid.append(region)

    if not initial_valid:
        return np.zeros_like(labeled_image, dtype=np.uint16), [], {
            'initial_count': 0, 'doublet_count': 0, 'split_count': 0, 'final_count': 0,
        }

    # Doublet splitting uses a 2D-only shape heuristic (circularity/eccentricity);
    # in 3D rely on Cellpose to separate touching nuclei (plan D2).
    if detection_image.ndim == 2:
        median_area = np.median([r.area for r in initial_valid])
        doublets, singles = [], []
        for region in initial_valid:
            (doublets if compute_doublet_score(region, median_area) > doublet_threshold
             else singles).append(region.label)
    else:
        doublets, singles = [], [r.label for r in initial_valid]

    final = np.zeros_like(labeled_image, dtype=np.uint16)
    current = 1
    for lbl in singles:
        final[labeled_image == lbl] = current
        current += 1

    split_count = 0
    for lbl in doublets:
        mask = labeled_image == lbl
        result = split_doublet(mask, min_distance=ws_min_distance)
        if result is not None:
            for sl in np.unique(result[result > 0]):
                if np.sum(result == sl) >= min_object_size:
                    final[result == sl] = current
                    current += 1
                    split_count += 1
        else:
            final[mask] = current
            current += 1

    return final, list(range(1, current)), {
        'initial_count': len(initial_valid), 'doublet_count': len(doublets),
        'split_count': split_count, 'final_count': current - 1,
    }


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def _surface_area_3d(region, spacing):
    """Calibrated surface area (um^2) of a 3D region via marching cubes.

    The mask is padded by 1 so faces on the bbox border are closed; `spacing`
    is (z_um, y_um, x_um) so the area is anisotropy-correct.
    """
    try:
        mask = np.pad(region.image, 1).astype(np.float32)
        verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, spacing=spacing)
        return float(measure.mesh_surface_area(verts, faces))
    except Exception:
        return np.nan


def _principal_axes_3d(region, spacing):
    """Anisotropy-corrected principal-axis analysis of a 3D region.

    Builds the mass-normalized covariance of the object's voxel coordinates *scaled
    to physical units* by ``spacing`` = (z_um, y_um, x_um), then eigen-decomposes
    it. Returns ``(evals, evecs)`` with eigenvalues ascending (µm²) and eigenvectors
    as columns, so ``evecs[:, -1]`` is the long-axis director. Unlike
    ``region.inertia_tensor_eigvals`` (voxel units, isotropic assumption), scaling
    the coordinates first makes every derived length/ratio correct when the z
    sampling differs from xy.
    """
    coords = region.coords.astype(np.float64) * np.asarray(spacing, dtype=np.float64)
    coords -= coords.mean(axis=0)
    cov = (coords.T @ coords) / coords.shape[0]
    evals, evecs = np.linalg.eigh(cov)               # ascending eigenvalues
    return np.clip(evals, 0.0, None), evecs


def _ellipsoid_axes_um(evals):
    """(major_len, minor_len, elongation, flatness) in physical units.

    From the ascending covariance eigenvalues of a 3D region (see
    ``_principal_axes_3d``): the equivalent uniform solid ellipsoid has semi-axis_i
    = sqrt(5·λ_i), so full axis lengths are calibrated (µm) and the ratios are
    anisotropy-correct. ``elongation`` = 1 − mid/major, ``flatness`` = 1 − minor/mid.
    """
    lam = np.sort(np.asarray(evals, dtype=float))[::-1]   # descending: major, mid, minor
    if lam.size < 3:
        return np.nan, np.nan, np.nan, np.nan
    semi = np.sqrt(5.0 * lam)
    major, mid, minor = 2.0 * semi[0], 2.0 * semi[1], 2.0 * semi[2]
    elongation = (1.0 - semi[1] / semi[0]) if semi[0] > 0 else np.nan
    flatness = (1.0 - semi[2] / semi[1]) if semi[1] > 0 else np.nan
    return major, minor, elongation, flatness


def extract_region_features(region, intensity_image, channels, config, compute_texture=True):
    px = config['pixel_size_um']
    pz = config.get('z_size_um')
    ndim = intensity_image.ndim
    coords_idx = tuple(region.coords.T)            # nD-safe gather (2D == coords[:,0],[:,1])
    intensity_values = intensity_image[coords_idx]

    solidity, convexity_defects = compute_convex_hull_features(region)
    features = {
        'solidity': solidity,
        'convexity_defects': convexity_defects,
    }
    if ndim == 2:
        circularity = (4 * np.pi * region.area / (region.perimeter ** 2)
                       if region.perimeter > 0 else 0)
        features['area_um2'] = region.area * px**2
        features['perimeter_um'] = region.perimeter * px
        features['circularity'] = circularity
        features['eccentricity'] = region.eccentricity
        features['major_axis_length_um'] = region.axis_major_length * px
        features['minor_axis_length_um'] = region.axis_minor_length * px
        features['centroid_x'] = region.centroid[1]
        features['centroid_y'] = region.centroid[0]
    else:
        # 3D counterparts (plan D4): area->volume, perimeter->surface area,
        # circularity->sphericity, eccentricity->elongation/flatness.
        spacing = ((pz or px), px, px)
        voxel_vol = px * px * (pz or px)
        volume = region.area * voxel_vol
        surface = _surface_area_3d(region, spacing)
        evals, _ = _principal_axes_3d(region, spacing)   # anisotropy-corrected axes
        major, minor, elongation, flatness = _ellipsoid_axes_um(evals)
        features['volume_um3'] = volume
        features['surface_area_um2'] = surface
        features['sphericity'] = (
            np.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / surface
            if surface and surface > 0 else np.nan)
        features['elongation'] = elongation
        features['flatness'] = flatness
        features['major_axis_length_um'] = major
        features['minor_axis_length_um'] = minor
        features['centroid_z'] = region.centroid[0]
        features['centroid_y'] = region.centroid[1]
        features['centroid_x'] = region.centroid[2]
    features.update(compute_intensity_statistics(intensity_values))
    # Texture is meaningless on a membrane channel (signal is at boundaries, not a
    # cytoplasmic texture), so callers can skip it for membrane-derived cytoplasm.
    if compute_texture:
        features.update(compute_texture_features(region, intensity_image))

    for stem, img in channels.items():
        ch_vals = img[coords_idx]
        features[f"{stem}_mean"] = np.mean(ch_vals)
        features[f"{stem}_integrated"] = np.sum(ch_vals)

    return features


def extract_all_nuclear_features(labeled_image, intensity_image, channels, config,
                                 compute_texture=True):
    regions = measure.regionprops(labeled_image, intensity_image=intensity_image)

    def _one(region):
        f = extract_region_features(region, intensity_image, channels, config,
                                    compute_texture=compute_texture)
        f['cell_id'] = region.label
        return f
    return _parallel_map(_one, regions)


def extract_all_cytoplasm_features(cytoplasm_labels, intensity_image, channels, config,
                                   compute_texture=True):
    regions = measure.regionprops(cytoplasm_labels, intensity_image=intensity_image)

    def _one(region):
        f = extract_region_features(region, intensity_image, channels, config,
                                    compute_texture=compute_texture)
        f['cell_id'] = region.label
        return f
    return _parallel_map(_one, regions)


# =============================================================================
# CELLPOSE CYTOPLASM SEGMENTATION
# =============================================================================

_CELLPOSE_MODEL_CACHE = {}
_DEVICE_CACHE = None


def detect_device():
    """Best-effort (device_str, gpu_available), cached.

    Returns ('cuda'|'mps'|'cpu', bool). Lets the app default Cellpose to the GPU
    when one is actually present and fall back to CPU otherwise, instead of
    blindly requesting a GPU that may not exist.
    """
    global _DEVICE_CACHE
    if _DEVICE_CACHE is None:
        device, gpu = 'cpu', False
        try:
            import torch
            if torch.cuda.is_available():
                device, gpu = 'cuda', True
            else:
                mps = getattr(getattr(torch, 'backends', None), 'mps', None)
                if mps is not None and mps.is_available():
                    device, gpu = 'mps', True
        except Exception:
            pass
        _DEVICE_CACHE = (device, gpu)
    return _DEVICE_CACHE


def get_cellpose_model_by_name(model_name='cpsam', use_gpu=True):
    """Return a (process-cached) Cellpose model; validates the name for Cellpose 4.x.

    Caching by (name, gpu) lets nuclei/stain/cytoplasm Cellpose all reuse one
    loaded model when they share a model_type, across every image in a batch.
    """
    # Never request a GPU that isn't present -- clamp to the detected device.
    use_gpu = bool(use_gpu) and detect_device()[1]
    key = (model_name, bool(use_gpu))
    model = _CELLPOSE_MODEL_CACHE.get(key)
    if model is None:
        valid_models = set(cp_models.MODEL_NAMES)
        if model_name not in valid_models and not os.path.exists(model_name):
            raise ValueError(
                f"Cellpose model '{model_name}' is not a valid built-in model for the "
                f"installed Cellpose ({cellpose_version}) and is not an existing file path. "
                f"Valid built-in models: {sorted(valid_models)}. "
                f"Cellpose 4.x removed the legacy cyto/cyto2/cyto3/nuclei models; use 'cpsam'."
            )
        model = cp_models.CellposeModel(pretrained_model=model_name, gpu=use_gpu)
        _CELLPOSE_MODEL_CACHE[key] = model
    return model


def is_model_cached(model_name='cpsam', use_gpu=True):
    """True if this model is already loaded, i.e. the next segmentation will NOT pay
    the weight-load + CUDA-context cost. Lets a caller show a 'loading model' status
    only when it is actually about to happen. Mirrors get_cellpose_model_by_name's key.
    """
    return (model_name, bool(use_gpu) and detect_device()[1]) in _CELLPOSE_MODEL_CACHE


def get_cellpose_model(config):
    """Backward-compatible model factory for the cytoplasm cellpose_* config keys."""
    return get_cellpose_model_by_name(
        config.get('cellpose_model_type', 'cpsam'), config.get('cellpose_gpu', True))


def label_dtype(n_labels):
    """Narrowest unsigned dtype that can hold labels 1..n_labels.

    Cellpose returns uint32 once an image holds more than 65535 instances (it only
    logs a warning). Casting that to uint16 wraps modulo 2**16 -- labels collide and
    object 65536 silently becomes 0, i.e. background -- so widen instead.
    """
    return np.uint16 if n_labels <= np.iinfo(np.uint16).max else np.uint32


def run_cellpose_labels(image, cp_params, model=None):
    """Run Cellpose on `image`; return instance labels at original resolution.

    `cp_params` is a flat dict (model_type, diameter, cellprob_threshold,
    flow_threshold, niter, min_size, max_size_fraction, gpu, invert, and -- for 3D
    z-stacks -- anisotropy). Role-agnostic: the same routine segments nuclei,
    stains/particles, or whole cells -- the caller decides. A 3D `(Z,Y,X)` input
    triggers do_3D.
    """
    if model is None:
        model = get_cellpose_model_by_name(
            cp_params.get('model_type', 'cpsam'), cp_params.get('gpu', True))
    if cp_params.get('invert', False):
        image = image.max() - image  # Cellpose 4.x eval() has no 'invert' arg
    diameter = cp_params.get('diameter', 0) or None
    eval_kwargs = dict(
        diameter=diameter,
        flow_threshold=cp_params.get('flow_threshold', 0.4),
        cellprob_threshold=cp_params.get('cellprob_threshold', 0.0),
        # 0 -> None lets Cellpose scale its 200 iterations by 30/diameter; pinning a
        # value here would under-integrate the dynamics for objects larger than 30 px.
        niter=cp_params.get('niter', 0) or None,
        min_size=cp_params.get('min_size', 15),
        # Cellpose deletes any instance covering more than this fraction of the frame.
        # Left at its 0.4 default it silently drops whole cells in cropped fields.
        max_size_fraction=cp_params.get('max_size_fraction', 0.4),
    )
    if image.ndim == 3:
        # Volumetric segmentation: one instance spans z (plan D1, do_3D only,
        # auto on a (Z,Y,X) stack). anisotropy = z/xy is injected by the caller
        # from the per-sample voxel calibration; None lets Cellpose assume isotropic.
        eval_kwargs.update(
            do_3D=True, z_axis=0, channel_axis=None,
            anisotropy=cp_params.get('anisotropy') or None,
        )
    result = model.eval(image, **eval_kwargs)
    masks = np.asarray(result[0])                    # result == (masks, flows, styles)
    n = int(masks.max()) if masks.size else 0
    return masks.astype(label_dtype(n))


def _filter_and_relabel(labels, min_object_size, edge_margin, shape):
    """Keep instances >= min_object_size and clear of the edge margin; relabel 1..N.

    Returns ``(final, kept, initial)`` where ``initial`` is the instance count Cellpose
    produced *before* filtering, so callers can report how many objects were removed.
    """
    initial = int(labels.max()) if labels.size else 0
    # One old_id -> new_id lookup table, applied in a single pass. A per-object
    # `labels == region.label` scan would cost O(n_objects * n_pixels).
    lut = np.zeros(initial + 1, dtype=label_dtype(initial))
    kept = 0
    for region in measure.regionprops(labels):
        if region.area < min_object_size:
            continue
        if _touches_border(region.coords, shape, edge_margin):
            continue
        kept += 1
        lut[region.label] = kept
    return lut[labels], kept, initial


def segment_nuclei_cellpose(detection_image, config, model=None):
    """Nuclei via Cellpose: instance labels, then min-size + edge-margin filtering.

    Cellpose separates touching instances itself, so the threshold pipeline's
    doublet/watershed step is not needed; min_object_size and edge_exclusion_margin
    still apply for parity with the threshold path.
    """
    labels = run_cellpose_labels(detection_image, config.get('nuclear_cellpose', {}), model=model)
    final, n, initial = _filter_and_relabel(
        labels, config['min_object_size'], config['edge_exclusion_margin'],
        detection_image.shape)
    return final, list(range(1, n + 1)), {
        'initial_count': initial, 'doublet_count': 0, 'split_count': 0, 'final_count': n,
    }


def segment_stain_cellpose(stain_image, stain_config, edge_margin=0):
    """Stain/particle detection via Cellpose: labelled instances, size + edge-filtered."""
    labels = run_cellpose_labels(stain_image, stain_config.get('cellpose', {}))
    final, n, initial = _filter_and_relabel(labels, stain_config['min_object_size'],
                                            edge_margin, stain_image.shape)
    return final, {
        'initial_count': initial, 'doublet_count': 0, 'split_count': 0,
        'final_count': n, 'output_type': 'labeled',
    }


def segment_cytoplasm_cellpose(image, nuclear_labels, config, model=None):
    """Whole-cell segmentation via Cellpose, then subtract nuclei -> cytoplasm."""
    cp_params = {
        'model_type': config.get('cellpose_model_type', 'cpsam'),
        'diameter': config.get('cellpose_diameter', 0),
        'cellprob_threshold': config.get('cellpose_cellprob_threshold', 0.0),
        'flow_threshold': config.get('cellpose_flow_threshold', 0.4),
        'niter': config.get('cellpose_niter', 0),
        'min_size': config.get('cellpose_min_size', 15),
        'max_size_fraction': config.get('cellpose_max_size_fraction', 0.4),
        'gpu': config.get('cellpose_gpu', True),
        'invert': config.get('cellpose_invert', False),
        'anisotropy': config.get('_anisotropy'),     # 3D only; ignored in 2D
    }
    masks = run_cellpose_labels(image, cp_params, model=model)
    if nuclear_labels is not None:
        masks[nuclear_labels > 0] = 0
    n_cyto = len(np.unique(masks[masks > 0]))
    return masks, {'n_cytoplasm': n_cyto}


def segment_cytoplasm_threshold(image, nuclear_labels, config, source='channel'):
    """Threshold-based cytoplasm: a nuclei-seeded watershed over a thresholded
    foreground, then subtract nuclei. Cytoplasm ids match their nucleus id.

    The watershed elevation adapts to the channel type (``source``):
      - 'membrane': the bright membrane acts as ridges, so basins flood out from
        each nucleus until they meet the membrane (elevation = channel intensity).
      - 'channel' (cytoplasm fill): partition the thresholded area by proximity to
        the nearest nucleus (elevation = distance from nuclei).
    """
    if nuclear_labels is None:
        return np.zeros(image.shape, dtype=np.uint16), {'n_cytoplasm': 0}

    cc = config.get('cytoplasm_threshold', {})
    thr_min = cc.get('threshold_min', 18)
    thr_max = cc.get('threshold_max', 255)
    min_object_size = cc.get('min_object_size', 0)

    foreground = (image >= thr_min) & (image <= thr_max)
    foreground = binary_fill_holes(foreground)
    foreground = foreground | (nuclear_labels > 0)  # nuclei must sit inside the basins

    if source == 'membrane':
        elevation = image.astype(np.float64)               # bright membrane = ridges
    else:
        # proximity to nearest nucleus; sampling makes z-distance anisotropy-aware
        sampling = None
        if image.ndim == 3:
            sampling = ((config.get('_anisotropy') or 1.0), 1.0, 1.0)
        elevation = ndimage.distance_transform_edt(nuclear_labels == 0, sampling=sampling)

    labels = segmentation.watershed(elevation, markers=nuclear_labels, mask=foreground)
    labels = np.asarray(labels).astype(np.uint16)
    labels[nuclear_labels > 0] = 0                         # cytoplasm = territory - nucleus

    if min_object_size > 0:
        for region in measure.regionprops(labels):
            if region.area < min_object_size:
                labels[labels == region.label] = 0

    n_cyto = len(np.unique(labels[labels > 0]))
    return labels, {'n_cytoplasm': n_cyto}

# =============================================================================
# STAIN SEGMENTATION
# =============================================================================

def segment_stain(stain_image, stain_config, edge_margin=0):
    """Segment a stain/particle channel. ``edge_margin`` (px) drops objects within
    that distance of the lateral image border (0 = keep all)."""
    method = stain_config.get('method', 'threshold')
    if method == 'none':
        # No segmentation for this detection; report an empty binary mask.
        return np.zeros(stain_image.shape, dtype=bool), {'object_count': 0, 'output_type': 'binary'}
    if method == 'cellpose':
        return segment_stain_cellpose(stain_image, stain_config, edge_margin)
    threshold_min = stain_config['threshold_min']
    threshold_max = stain_config['threshold_max']
    min_object_size = stain_config['min_object_size']
    output_type = stain_config.get('output_type', 'binary')

    binary_mask = (stain_image >= threshold_min) & (stain_image <= threshold_max)
    binary_mask = binary_fill_holes(binary_mask)

    shape = binary_mask.shape
    if output_type == 'binary':
        labeled_temp = measure.label(binary_mask, connectivity=2)
        filtered = np.zeros_like(binary_mask)
        kept = 0
        for region in measure.regionprops(labeled_temp):
            if region.area >= min_object_size and not _touches_border(region.coords, shape, edge_margin):
                filtered[labeled_temp == region.label] = True
                kept += 1
        return filtered.astype(bool), {'object_count': kept, 'output_type': 'binary'}

    doublet_threshold = stain_config.get('doublet_threshold', 0.7)
    ws_distance = stain_config.get('watershed_min_distance', 80)

    labeled = measure.label(binary_mask, connectivity=2)
    valid = [r for r in measure.regionprops(labeled)
             if r.area >= min_object_size and not _touches_border(r.coords, shape, edge_margin)]
    if not valid:
        return np.zeros_like(labeled, dtype=np.uint16), {
            'initial_count': 0, 'doublet_count': 0, 'split_count': 0,
            'final_count': 0, 'output_type': 'labeled',
        }

    # 2D-only doublet heuristic (plan D2); in 3D keep connected components as-is.
    if stain_image.ndim == 2:
        median_area = np.median([r.area for r in valid])
        doublet_labels, single_labels = [], []
        for r in valid:
            (doublet_labels if compute_doublet_score(r, median_area) > doublet_threshold
             else single_labels).append(r.label)
    else:
        doublet_labels, single_labels = [], [r.label for r in valid]

    final = np.zeros_like(labeled, dtype=np.uint16)
    current = 1
    for lbl in single_labels:
        final[labeled == lbl] = current
        current += 1
    split_count = 0
    for lbl in doublet_labels:
        mask = labeled == lbl
        result = split_doublet(mask, min_distance=ws_distance)
        if result is not None:
            for sl in np.unique(result[result > 0]):
                if np.sum(result == sl) >= min_object_size:
                    final[result == sl] = current
                    current += 1
                    split_count += 1
        else:
            final[mask] = current
            current += 1

    return final, {
        'initial_count': len(valid), 'doublet_count': len(doublet_labels),
        'split_count': split_count, 'final_count': current - 1, 'output_type': 'labeled',
    }


# =============================================================================
# STAIN QUANTIFICATION
# =============================================================================

def quantify_stain_per_labels(stain_binary, label_image, pixel_size_um, z_size_um=None):
    """Per-region stain quantities. `*_measure_cal` is calibrated area (2D, um^2)
    or volume (3D, um^3); the dataframe layer names the output column accordingly.
    """
    factor = (pixel_size_um ** 2 if z_size_um is None
              else pixel_size_um * pixel_size_um * z_size_um)

    def _one(lbl):
        region_mask = label_image == lbl
        region_area = np.sum(region_mask)
        stain_in = stain_binary & region_mask
        stain_px = np.sum(stain_in)
        particle_count = measure.label(stain_in, connectivity=2).max()
        return {
            'label_id': int(lbl),
            'region_measure_cal': region_area * factor,
            'stain_measure_cal': stain_px * factor,
            'stain_area_px': int(stain_px),
            'stain_particle_count': int(particle_count),
            'stain_coverage_fraction': stain_px / region_area if region_area > 0 else 0.0,
        }
    return _parallel_map(_one, np.unique(label_image[label_image > 0]))


def particle_size_key(ndim):
    """Name of the size metric the outlier screen uses for this dimensionality."""
    return 'area_um2' if ndim == 2 else 'volume_um3'


def quantify_stain_particles(stain_binary, cell_territory, pixel_size_um, z_size_um=None):
    """Per *physical* stain particle: its size and the cell that owns most of it.

    The stain is labelled ONCE over the whole image (connectivity=2, matching
    ``mean_particle_area_px``), then each particle is assigned to the cell holding the
    majority of its pixels. Clipping the stain per region instead -- once to the nuclei,
    once to the disjoint cytoplasms -- would split a punctum straddling the nuclear
    border into two fragments and measure both, inflating the small-size tail.

    ``cell_territory`` is a label image carrying each cell's id over its whole territory
    (nucleus + its cytoplasm). Particles overlapping no cell contribute nothing. Returns
    a flat list of ``{size_key: float, 'parent_label': cell_id}``; only size is emitted,
    because shape is unresolvable at particle scale (see the outlier screen).
    """
    if cell_territory is None or not np.any(stain_binary):
        return []
    labels = measure.label(stain_binary, connectivity=2)
    ndim = labels.ndim
    px = pixel_size_um
    voxel = px ** 2 if ndim == 2 else px * px * (z_size_um or px)
    size_key = particle_size_key(ndim)
    minlength = int(cell_territory.max()) + 1

    def _one(region):
        owners = cell_territory[tuple(region.coords.T)]
        owners = owners[owners > 0]
        if owners.size == 0:
            return None
        parent = int(np.bincount(owners, minlength=minlength).argmax())
        return {size_key: region.area * voxel, 'parent_label': parent}

    return [p for p in _parallel_map(_one, measure.regionprops(labels)) if p is not None]


def mean_particle_area_px(stain_binary):
    """Mean area (in pixels/voxels) of an individual stain particle over ALL
    particles in the image: total stain area / number of connected components
    (connectivity=2, matching the per-region particle counts). Returns 0.0 for an
    empty mask. This is the per-image analysis-filter threshold -- a cell "contains"
    the stain when the stain area inside it reaches one average particle.
    """
    total = int(np.sum(stain_binary))
    if total == 0:
        return 0.0
    n = int(measure.label(stain_binary, connectivity=2).max())
    return total / n if n > 0 else 0.0


# =============================================================================
# STAIN SPATIAL ANALYSIS
# =============================================================================

def _project_coords_onto_axes(coords_yx, centroid_yx, orientation):
    dy = coords_yx[:, 0] - centroid_yx[0]
    dx = coords_yx[:, 1] - centroid_yx[1]
    proj_major = dy * (-np.sin(orientation)) + dx * np.cos(orientation)
    proj_minor = dy * np.cos(orientation) + dx * np.sin(orientation)
    return proj_major, proj_minor


def _count_peaks_in_profile(projections, pixel_size_um):
    if len(projections) < 3:
        return 0, np.nan
    lo, hi = projections.min(), projections.max()
    span_px = hi - lo
    if span_px < 2:
        return 0, np.nan
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
    hist, _ = np.histogram(projections, bins=bins)
    smoothed = uniform_filter1d(hist.astype(np.float64), size=max(3, int(len(hist) * 0.05)))
    peaks, _ = scipy_find_peaks(smoothed, height=0.5)
    span_um = span_px * pixel_size_um
    return int(len(peaks)), len(peaks) / span_um if span_um > 0 else np.nan


def _stain_spatial_3d(stain_binary, label_image, regions, pixel_size_um):
    """3D analog of the stain spatial descriptors.

    Particle alignment = 3D nematic order of the stain particles' principal-axis
    directors; periodicity = peak counts of the stain voxels projected onto the
    nucleus's longest ('major') and shortest ('minor') principal axes. Projection
    distances are in voxel units (anisotropy not corrected -- exploratory metric).
    """
    results = []
    for region in regions:
        region_mask = label_image == region.label
        stain_in = stain_binary & region_mask
        if not np.any(stain_in):
            results.append({
                'label_id': region.label, 'n_stain_particles': 0,
                'particle_alignment': np.nan,
                'peak_count_major': 0, 'freq_major_axis': np.nan,
                'peak_count_minor': 0, 'freq_minor_axis': np.nan,
            })
            continue
        particles = measure.regionprops(measure.label(stain_in))
        n_particles = len(particles)
        alignment = (_nematic_order_3d([_major_axis_director(p) for p in particles])
                     if n_particles >= 2 else np.nan)
        try:
            _, vecs = np.linalg.eigh(region.inertia_tensor)
            major_vec, minor_vec = vecs[:, 0], vecs[:, 2]   # long / short axis
        except Exception:
            major_vec = minor_vec = None
        if major_vec is not None:
            rel = np.argwhere(stain_in).astype(float) - np.asarray(region.centroid, float)
            pk_maj, freq_maj = _count_peaks_in_profile(rel @ major_vec, pixel_size_um)
            pk_min, freq_min = _count_peaks_in_profile(rel @ minor_vec, pixel_size_um)
        else:
            pk_maj = pk_min = 0
            freq_maj = freq_min = np.nan
        results.append({
            'label_id': region.label, 'n_stain_particles': n_particles,
            'particle_alignment': alignment,
            'peak_count_major': pk_maj, 'freq_major_axis': freq_maj,
            'peak_count_minor': pk_min, 'freq_minor_axis': freq_min,
        })
    return results


def compute_stain_spatial_analysis(stain_binary, label_image, pixel_size_um):
    regions = measure.regionprops(label_image)
    if label_image.ndim != 2:
        return _stain_spatial_3d(stain_binary, label_image, regions, pixel_size_um)
    results = []
    for region in regions:
        region_mask = label_image == region.label
        stain_in = stain_binary & region_mask
        if not np.any(stain_in):
            results.append({
                'label_id': region.label, 'n_stain_particles': 0,
                'particle_alignment': np.nan,
                'peak_count_major': 0, 'freq_major_axis': np.nan,
                'peak_count_minor': 0, 'freq_minor_axis': np.nan,
            })
            continue

        labeled_stain = measure.label(stain_in, connectivity=2)
        particles = measure.regionprops(labeled_stain)
        n_particles = len(particles)

        if n_particles >= 2:
            p_orient = np.array([p.orientation for p in particles])
            rel = 2.0 * (p_orient - region.orientation)
            alignment = np.sqrt(np.mean(np.cos(rel))**2 + np.mean(np.sin(rel))**2)
        else:
            alignment = np.nan

        stain_coords = np.argwhere(stain_in)
        centroid_yx = (region.centroid[0], region.centroid[1])
        proj_maj, proj_min = _project_coords_onto_axes(stain_coords, centroid_yx, region.orientation)
        pk_maj, freq_maj = _count_peaks_in_profile(proj_maj, pixel_size_um)
        pk_min, freq_min = _count_peaks_in_profile(proj_min, pixel_size_um)

        results.append({
            'label_id': region.label, 'n_stain_particles': n_particles,
            'particle_alignment': alignment,
            'peak_count_major': pk_maj, 'freq_major_axis': freq_maj,
            'peak_count_minor': pk_min, 'freq_minor_axis': freq_min,
        })
    return results


# =============================================================================
# PER-IMAGE QC (acquisition portability diagnostics)
# =============================================================================

def compute_channel_qc(image, threshold_min=None, threshold_max=None):
    """Per-image QC stats on RAW intensities for one channel.

    These do NOT change segmentation; they let you detect when a sample is not
    comparable to the rest of the batch under the fixed global threshold. Watch
    ``pct_in_threshold_band`` across samples: large swings at the same threshold
    indicate acquisition (exposure/illumination/bleaching) differences, not
    biology. ``pct_saturated`` flags clipped images whose intensities are
    unreliable.
    """
    flat = np.asarray(image).reshape(-1)
    n = flat.size
    stats = {
        'intensity_mean': float(np.mean(flat)),
    }
    if np.issubdtype(flat.dtype, np.integer):
        dtype_max = np.iinfo(flat.dtype).max
        stats['pct_saturated'] = float(np.count_nonzero(flat >= dtype_max) / n * 100.0)
    else:
        stats['pct_saturated'] = np.nan
    if threshold_min is not None:
        lo = threshold_min
        hi = threshold_max if threshold_max is not None else np.inf
        in_band = (flat >= lo) & (flat <= hi)
        stats['threshold_min_used'] = float(lo)
        if threshold_max is not None:
            stats['threshold_max_used'] = float(threshold_max)
        stats['pct_in_threshold_band'] = float(np.count_nonzero(in_band) / n * 100.0)
    return stats


# =============================================================================
# NUCLEUS → CYTOPLASM MAPPING
# =============================================================================

def map_nuclei_to_cytoplasm(nuclear_labels, cytoplasm_labels, adjacency_radius=2):
    selem = (morphology.ball(adjacency_radius) if nuclear_labels.ndim == 3
             else morphology.disk(adjacency_radius))
    mapping = {}
    for nuc_id in np.unique(nuclear_labels[nuclear_labels > 0]):
        nuc_mask = nuclear_labels == nuc_id
        # morphology.dilation on a boolean image == binary dilation; binary_dilation
        # is deprecated in scikit-image 0.26 and removed in 0.28 (matches the label
        # dilation already used in split_doublet).
        ring = morphology.dilation(nuc_mask, selem) & ~nuc_mask
        cyto_vals = cytoplasm_labels[ring]
        cyto_vals = cyto_vals[cyto_vals > 0]
        if len(cyto_vals) > 0:
            unique, counts = np.unique(cyto_vals, return_counts=True)
            mapping[nuc_id] = int(unique[np.argmax(counts)])
    return mapping