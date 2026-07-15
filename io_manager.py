import numpy as np
import pandas as pd
from skimage import io, measure
from pathlib import Path
import os
import re
import itertools
import tifffile
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from processing import (
    detect_device,
    segment_nuclei,
    extract_all_nuclear_features,
    compute_nuclear_alignment,
    segment_cytoplasm_cellpose,
    segment_cytoplasm_threshold,
    get_cellpose_model,
    label_dtype,
    extract_all_cytoplasm_features,
    segment_stain,
    quantify_stain_per_labels,
    quantify_stain_particles,
    particle_size_key,
    mean_particle_area_px,
    compute_stain_spatial_analysis,
    compute_channel_qc,
    map_nuclei_to_cytoplasm,
    set_worker_count,
    thread_report,
)


# Default to the GPU only when one is actually detected (CUDA/MPS), else CPU.
GPU_AVAILABLE = detect_device()[1]

# Reusable Cellpose parameter block (one independent copy per detection, so each
# nucleus/stain/cytoplasm step can be tuned separately).
DEFAULT_CELLPOSE_PARAMS = {
    'model_type': 'cpsam',
    'diameter': 0,
    'cellprob_threshold': 0.0,
    'flow_threshold': 0.4,
    'niter': 0,                  # 0 = auto: Cellpose scales 200 iterations by 30/diameter
    'min_size': 15,
    'max_size_fraction': 0.4,    # Cellpose drops instances larger than this fraction
    'gpu': GPU_AVAILABLE,
    'invert': False,
}

DEFAULT_CONFIG = {
    # CPU worker threads for the parallel feature-extraction / stain-quantification
    # loops. None -> auto = (logical cores - 1). 1 = sequential. Set once per run.
    'num_threads': None,
    'pixel_size_um': 0.0613414,
    # Voxel depth for 3D z-stacks (um/plane). None -> resolved per-sample from the
    # TIFF metadata at load time, falling back to pixel_size_um (isotropic) when the
    # file carries no z-spacing. Ignored for 2D inputs.
    'z_size_um': None,
    # Nuclear detection
    'nuclear_method': 'threshold',  # 'threshold' or 'cellpose' (independent of role)
    'nuclear_cellpose': dict(DEFAULT_CELLPOSE_PARAMS),
    'threshold_min': 18,
    'threshold_max': 255,
    'min_object_size': 2500,
    'connectivity': 2,
    'edge_exclusion_margin': 5,
    'doublet_threshold': 0.7,
    'watershed_min_distance': 80,
    # Cytoplasm source
    'cytoplasm_source': 'none',  # 'none', 'membrane', 'channel'
    'cytoplasm_method': 'cellpose',  # 'cellpose' or 'threshold' (nuclei-seeded watershed)
    'cytoplasm_threshold': {'threshold_min': 18, 'threshold_max': 255, 'min_object_size': 0},
    # Cellpose params
    'cellpose_model_type': 'cpsam',
    'cellpose_diameter': 0,
    'cellpose_cellprob_threshold': 0.0,
    'cellpose_flow_threshold': 0.4,
    'cellpose_niter': 0,
    'cellpose_min_size': 15,
    'cellpose_max_size_fraction': 0.4,
    'cellpose_gpu': GPU_AVAILABLE,
    'cellpose_invert': False,
    # Channel roles
    'channel_roles': {},
    # Per-stain configs
    'stain_configs': {},
    'input_folder': None,
    # Input organization: 'folder_per_sample' (each folder = a sample, files/splits =
    # channels), 'file_per_sample' (each multi-channel file = a sample, bare c0/c1
    # channels), or 'auto' (prefer folder; fall back to file). See IO_PLAN.md.
    'input_mode': 'auto',
    # Analysis filter
    'filter_stain': None,
    'filter_condition': 'contains',
    # Outlier removal (two-sided robust z on log-size, per channel -> extra output tier).
    # 4.0, not 5.0: the statistic screens log10(size), on which 5.0 flags nothing at all.
    'remove_outliers': True,
    'outlier_mad_threshold': 4.0,
    # Pre-segmented
    'use_presegmented_nuclear': False,
    'presegmented_nuclear_stem': '',
    'use_presegmented_cytoplasm': False,
    'presegmented_cytoplasm_stem': '',
}

DEFAULT_STAIN_CONFIG = {
    'method': 'threshold',  # 'threshold' or 'cellpose' (independent of role)
    'cellpose': dict(DEFAULT_CELLPOSE_PARAMS),
    'threshold_min': 18,
    'threshold_max': 255,
    'min_object_size': 100,
    'output_type': 'binary',
    'doublet_threshold': 0.7,
    'watershed_min_distance': 80,
}


# =============================================================================
# COLUMN ORDERING
# =============================================================================

# Union of 2D and 3D morphology keys; _prefix_and_order emits whichever are
# present in a given feature dict, so 2D rows keep their original column order and
# 3D rows get the volumetric counterparts (plan D3/D4).
MORPHOLOGY_ORDER = [
    'area_um2', 'volume_um3',
    'circularity', 'sphericity', 'solidity', 'eccentricity',
    'elongation', 'flatness',
    'major_axis_length_um', 'minor_axis_length_um',
    'convexity_defects',
    'perimeter_um', 'surface_area_um2',
]
INTENSITY_ORDER = ['intensity_mean', 'intensity_sd']
TEXTURE_ORDER = [
    'texture_contrast', 'texture_homogeneity',
    'texture_energy', 'texture_correlation',
]

EXCLUDED_KEYS = {
    'centroid_x', 'centroid_y', 'centroid_z',
    'intensity_median', 'intensity_p10', 'intensity_p90',
    'integrated_intensity',
}


def _is_excluded(col):
    return col in EXCLUDED_KEYS or col.endswith('_integrated')


def _prefix_and_order(features, prefix, channel_stems, skip_stem=None):
    out = {}
    for group in (MORPHOLOGY_ORDER, INTENSITY_ORDER, TEXTURE_ORDER):
        for k in group:
            if k in features and not _is_excluded(k):
                out[f'{prefix}_{k}'] = features[k]
    # The compartment's own detection channel mean equals intensity_mean, so skip
    # it here to avoid a duplicate column (intensity_sd has no per-channel twin).
    for stem in sorted(channel_stems):
        if stem == skip_stem:
            continue
        k = f'{stem}_mean'
        if k in features:
            out[f'{prefix}_{k}'] = features[k]
    return out


def _particle_feature_columns(features, self_stem, channel_stems):
    """Bare (un-prefixed) feature columns for one stain particle, in the same order
    as the cell tables: morphology, intensity, texture, then each *other* channel's
    mean (the particle's own channel mean duplicates intensity_mean, so it is
    skipped). Used by particle mode, where there is no N_/C_ compartment prefix."""
    out = {}
    for group in (MORPHOLOGY_ORDER, INTENSITY_ORDER, TEXTURE_ORDER):
        for k in group:
            if k in features and not _is_excluded(k):
                out[k] = features[k]
    for stem in sorted(channel_stems):
        if stem == self_stem:
            continue
        k = f'{stem}_mean'
        if k in features:
            out[k] = features[k]
    return out


# =============================================================================
# CHANNEL ROLE HELPERS
# =============================================================================

def get_role_stem(channel_roles, role):
    for stem, r in channel_roles.items():
        if r == role:
            return stem
    return None


def get_stain_stems(channel_roles):
    return [stem for stem, r in channel_roles.items() if r == 'stain']


def get_primary_stem(channel_roles):
    """The channel that defines a processable sample: the nuclear channel when one
    is assigned, else the first stain (particle mode -- no nuclei), else None.

    Lets discovery/orchestration accept nucleus-free samples: when there is no
    nuclear role the stains are analysed as standalone particles.
    """
    nuclear_stem = get_role_stem(channel_roles, 'nuclear')
    if nuclear_stem is not None:
        return nuclear_stem
    stains = get_stain_stems(channel_roles)
    return stains[0] if stains else None


def active_stain_stems(channel_roles, stain_configs):
    """Stain channels that are actually segmented (method != 'none').

    A stain whose method is 'none' stays a measured channel (its per-cell mean is
    still reported) but produces no stain mask or stain_<s>_* columns.
    """
    stain_configs = stain_configs or {}
    return [s for s in get_stain_stems(channel_roles)
            if stain_configs.get(s, {}).get('method', 'threshold') != 'none']


# Accepted channel-image extensions, matched case-insensitively (.tif/.tiff,
# .TIF/.TIFF, .Tif, ...) so detection does not depend on the filesystem's case rules.
TIFF_SUFFIXES = ('.tif', '.tiff')


def _is_tiff(path):
    return path.suffix.lower() in TIFF_SUFFIXES


def _is_supported_image(path):
    return path.suffix.lower() in supported_image_extensions()


def list_tiffs(folder):
    """Top-level supported image files in `folder`, sorted by name.

    TIFF always, plus ND2/CZI/LIF when aicsimageio is installed. `.tif` sorts before
    the longer `.tiff` for the same stem, so it wins in the stem de-duplication used
    by discovery/loading (preserving prior precedence).
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.glob('*') if p.is_file() and _is_supported_image(p)),
                  key=lambda p: p.name)


def resolve_stem(folder, stem):
    """Actual path of the TIFF named `stem` (.tif/.tiff, any case), or None.

    Returns the real on-disk path (correct case, so it also resolves on
    case-sensitive filesystems); prefers `.tif` over `.tiff` when both exist.
    """
    matches = [p for p in list_tiffs(folder) if p.stem == stem]
    if not matches:
        return None
    matches.sort(key=lambda p: (p.suffix.lower() != '.tif', p.name))
    return matches[0]


# =============================================================================
# TIFF AXES / METADATA  (3D-aware I/O, inspired by github.com/Matthias-Blanc/RAD)
# =============================================================================
#
# A channel file may be 2D (Y,X) or a 3D z-stack (Z,Y,X), and may bundle several
# channels/timepoints (C/T axes). These helpers read the tifffile axis string,
# canonicalize arrays to (Z,Y,X)/(Y,X), auto-split any C/T axes into separate
# virtual channels (so multi-channel acquisitions need no manual pre-split), and
# read voxel calibration (xy pixel size, z spacing, unit) from the metadata.
# Pixel values are returned RAW -- Vairons thresholds on raw intensities.

_OME_PSX = re.compile(r'PhysicalSizeX="([0-9.eE+-]+)"')
_OME_PSZ = re.compile(r'PhysicalSizeZ="([0-9.eE+-]+)"')

# Length units -> micrometres. Used to make file-embedded pixel sizes unit-correct.
_UNIT_UM = {
    'um': 1.0, 'µm': 1.0, 'μm': 1.0, 'micron': 1.0, 'microns': 1.0,
    'micrometer': 1.0, 'micrometre': 1.0,
    'nm': 1e-3, 'nanometer': 1e-3, 'nanometre': 1e-3,
    'mm': 1e3, 'millimeter': 1e3, 'millimetre': 1e3,
    'cm': 1e4, 'centimeter': 1e4, 'centimetre': 1e4,
    'm': 1e6, 'meter': 1e6, 'metre': 1e6,
    'in': 25400.0, 'inch': 25400.0,
}


def _unit_to_um(unit):
    """Multiplicative factor from ``unit`` to micrometres, or None if ``unit`` is not
    a physical length (blank / 'pixel' / unknown -> uncalibrated)."""
    if not unit:
        return None
    return _UNIT_UM.get(str(unit).strip().lower())


def _resolve_axes(axes, ndim):
    """Return an axis-letter string of length ndim, repairing missing/odd labels.

    tifffile usually provides a per-series axes string (e.g. 'YX', 'ZYX',
    'CZYX'). When it is absent or inconsistent, fall back to positional guesses
    and promote an unlabeled leading axis of a >=3D stack to Z.
    """
    axes = (axes or '').upper()
    if len(axes) != ndim:
        axes = {1: 'X', 2: 'YX', 3: 'ZYX', 4: 'CZYX', 5: 'TCZYX'}.get(ndim, 'Q' * ndim)
    # Promote an *unlabeled* leading axis (Q/I, as on a plain multipage stack with
    # no metadata) to Z. A known C/T/S axis is left alone -- a CYX file is genuine
    # 2D-multichannel, not a z-stack.
    if 'Z' not in axes and ndim >= 3:
        for i, a in enumerate(axes):
            if a not in 'YXCTS':
                axes = axes[:i] + 'Z' + axes[i + 1:]
                break
    return axes


def _split_plan(axes, shape):
    """Plan how to canonicalize + split an array of the given axes/shape.

    Returns (is_3d, spatial_pos, entries): ``spatial_pos`` are the current axis
    indices of the spatial axes in canonical order ((Z,)Y,X); ``entries`` is a
    list of (suffix, combo) -- one per virtual channel -- where ``combo`` indexes
    the non-spatial (C/T/...) axes and ``suffix`` names it ('' when nothing
    splits, '_c0'/'_t1'/... otherwise).
    """
    axes = _resolve_axes(axes, len(shape))
    keep = (['Z'] if ('Z' in axes and shape[axes.index('Z')] > 1) else []) + ['Y', 'X']
    spatial_pos = [axes.index(a) for a in keep]
    split_pos = [i for i in range(len(shape)) if i not in spatial_pos]
    is_3d = 'Z' in keep
    if not split_pos:
        return is_3d, spatial_pos, [('', ())]
    letters = [axes[i] for i in split_pos]
    sizes = [shape[i] for i in split_pos]
    entries = []
    for combo in itertools.product(*[range(s) for s in sizes]):
        suffix = ''.join(f'_{lett.lower()}{idx}'
                         for lett, idx, s in zip(letters, combo, sizes) if s > 1)
        entries.append((suffix, combo))
    return is_3d, spatial_pos, entries


def _virtual_stems(path):
    """(cheap, header-only for TIFF) -> list of (suffix, is_3d) for a channel file."""
    if Path(path).suffix.lower() in TIFF_SUFFIXES:
        try:
            with tifffile.TiffFile(path) as tif:
                s = tif.series[0]
                is_3d, _, entries = _split_plan(s.axes, tuple(s.shape))
            return [(suf, is_3d) for suf, _ in entries]
        except Exception:
            return [('', False)]
    # Non-TIFF: no cheap header API here -> best-effort via a full read.
    try:
        vols, meta = _read_via_aicsimageio(path)
        return [(suf, bool(meta.get('is_3d'))) for suf, _ in vols]
    except Exception:
        return [('', False)]


def _extract_calibration(tif):
    """Best-effort, *unit-aware* voxel calibration: (pixel_size_um, z_size_um, unit, source).

    Trusts only genuine microscopy calibration: ImageJ metadata whose ``unit`` is a
    length (micron / nm / mm / ...) — converted to µm — or OME ``PhysicalSize*`` (µm).
    A bare TIFF resolution in inches or with no unit (i.e. a print-DPI default) is
    treated as UNCALIBRATED, so it can never be mistaken for a pixel size. Returns
    px/pz in µm (or None when unknown) and a ``source`` tag ('imagej' | 'ome' | None)
    that the caller records for reproducibility.
    """
    px = pz = source = None
    try:
        ij = tif.imagej_metadata or {}
    except Exception:
        ij = {}
    unit = ij.get('unit') if ij else None
    factor = _unit_to_um(unit)
    if factor is not None:                             # trustworthy length unit
        try:
            xr = tif.pages[0].tags['XResolution'].value
            if xr and xr[0]:
                px = (xr[1] / xr[0]) * factor          # units/pixel -> µm/pixel
                source = 'imagej'
        except Exception:
            px = None
        sp = ij.get('spacing')
        if sp:
            pz = float(sp) * factor
            source = source or 'imagej'
    if px is None or pz is None:                       # OME PhysicalSize* are in µm
        try:
            ome = tif.ome_metadata
        except Exception:
            ome = None
        if ome:
            if px is None:
                m = _OME_PSX.search(ome)
                if m:
                    px = float(m.group(1)); source = source or 'ome'; unit = unit or 'um'
            if pz is None:
                m = _OME_PSZ.search(ome)
                if m:
                    pz = float(m.group(1)); source = source or 'ome'
    return px, pz, unit, source


def read_tiff_with_meta(path):
    """Read a channel TIFF into one or more canonical volumes + calibration.

    Returns (volumes, meta):
      volumes : list of (suffix, ndarray); ndarray is RAW pixels in (Z,Y,X) (3D)
                or (Y,X) (2D). More than one entry only when the file bundles C/T.
      meta    : {axes, is_3d, pixel_size_um, z_size_um, unit}
    """
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
        axes = _resolve_axes(tif.series[0].axes, arr.ndim)
        px, pz, unit, source = _extract_calibration(tif)
    is_3d, spatial_pos, entries = _split_plan(axes, arr.shape)
    dst = list(range(arr.ndim - len(spatial_pos), arr.ndim))
    moved = np.moveaxis(arr, spatial_pos, dst)        # spatial axes -> trailing
    volumes = [(suffix, np.ascontiguousarray(moved[combo] if combo else moved))
               for suffix, combo in entries]
    meta = {'axes': axes, 'is_3d': is_3d,
            'pixel_size_um': px, 'z_size_um': pz, 'unit': unit,
            'calibration_source': source}
    return volumes, meta


# =============================================================================
# READER REGISTRY  (TIFF always; ND2/CZI/LIF via aicsimageio when installed)
# =============================================================================

_AICS_EXTS = {'.nd2', '.czi', '.lif'}


def _aicsimageio_available():
    try:
        import aicsimageio  # noqa: F401
        return True
    except Exception:
        return False


def supported_image_extensions():
    """Image extensions the pipeline can read. TIFF is always available; ND2/CZI/LIF
    are added only when the optional ``aicsimageio`` dependency is installed, so with
    a plain TIFF-only environment behaviour is unchanged."""
    exts = set(TIFF_SUFFIXES)
    if _aicsimageio_available():
        exts |= _AICS_EXTS
    return exts


def _read_via_aicsimageio(path):
    """Read ND2/CZI/LIF (any container aicsimageio supports) into the canonical
    ``(suffix, volume)`` list + calibration meta, splitting channels/timepoints into
    virtual channels ('_c0','_t1', ...) exactly like the TIFF path. Physical pixel
    sizes come from the file (µm). Requires ``aicsimageio``."""
    from aicsimageio import AICSImage
    img = AICSImage(path)
    d = img.dims
    nC, nT, nZ = int(getattr(d, 'C', 1)), int(getattr(d, 'T', 1)), int(getattr(d, 'Z', 1))
    is_3d = nZ > 1
    pps = img.physical_pixel_sizes                    # (Z, Y, X) µm; any may be None
    order = 'ZYX' if is_3d else 'YX'
    volumes = []
    for ti in range(nT):
        for ci in range(nC):
            data = np.ascontiguousarray(img.get_image_data(order, C=ci, T=ti))
            suffix = (f'_c{ci}' if nC > 1 else '') + (f'_t{ti}' if nT > 1 else '')
            volumes.append((suffix, data))
    meta = {'axes': order, 'is_3d': is_3d,
            'pixel_size_um': (pps.X or pps.Y), 'z_size_um': (pps.Z if is_3d else None),
            'unit': 'um', 'calibration_source': 'aicsimageio'}
    return volumes, meta


def read_image_with_meta(path):
    """Read any supported image into ``(volumes, meta)``: TIFF via tifffile, else
    ND2/CZI/LIF via aicsimageio. Raises a clear error for anything unsupported."""
    ext = Path(path).suffix.lower()
    if ext in TIFF_SUFFIXES:
        return read_tiff_with_meta(path)
    if ext in _AICS_EXTS:
        return _read_via_aicsimageio(path)
    raise ValueError(f"Unsupported image format '{ext}' for {path}. "
                     f"Supported: {sorted(supported_image_extensions())}.")


def resolve_calibration(config, sample_meta):
    """Effective (pixel_size_um, z_size_um, anisotropy) for a sample.

    xy stays user-authoritative (config), preserving 2D behavior; z is taken from
    config when set, else the file metadata, else the isotropic fallback (z = xy).
    anisotropy = z/xy (1.0 in 2D).
    """
    px = config.get('pixel_size_um') or sample_meta.get('pixel_size_um') or 1.0
    if not sample_meta.get('is_3d'):
        return px, None, 1.0
    pz = config.get('z_size_um') or sample_meta.get('z_size_um') or px
    aniso = (pz / px) if px else 1.0
    return px, pz, aniso


def describe_calibration(config, sample_meta):
    """Provenance record of a sample's resolved calibration, for the QC sidecar and a
    console warning. Reports where the pixel/voxel size came from — the user's config,
    the file metadata (with the reader that supplied it), or the uncalibrated 1 µm/px
    fallback — and whether the result is physically calibrated at all."""
    cfg_px = config.get('pixel_size_um')
    file_px = sample_meta.get('pixel_size_um')
    if cfg_px:
        px, src = float(cfg_px), 'config'
    elif file_px:
        px, src = float(file_px), (sample_meta.get('calibration_source') or 'file')
    else:
        px, src = 1.0, 'fallback'
    rec = {'pixel_size_um': px, 'source': src,
           'unit': sample_meta.get('unit'), 'calibrated': src != 'fallback'}
    if sample_meta.get('is_3d'):
        cfg_pz, file_pz = config.get('z_size_um'), sample_meta.get('z_size_um')
        rec['z_size_um'] = float(cfg_pz) if cfg_pz else (float(file_pz) if file_pz else px)
        rec['z_source'] = 'config' if cfg_pz else ('file' if file_pz else 'fallback')
    return rec


def _apply_sample_calibration(config, px, pz, aniso):
    """Return a per-sample copy of config with resolved calibration injected.

    Sets pixel_size_um / z_size_um and threads anisotropy into every per-detection
    Cellpose block (+ a top-level `_anisotropy` for the cytoplasm path). The shared
    batch config is never mutated. In 2D (aniso == 1, pz is None) nothing downstream
    reads these, so the 2D path is unaffected.
    """
    cfg = dict(config)
    cfg['pixel_size_um'] = px
    cfg['z_size_um'] = pz
    cfg['_anisotropy'] = aniso
    nc = dict(cfg.get('nuclear_cellpose') or {})
    nc['anisotropy'] = aniso
    cfg['nuclear_cellpose'] = nc
    stain_cfgs = {}
    for stem, sc in (cfg.get('stain_configs') or {}).items():
        sc2 = dict(sc)
        cp = dict(sc2.get('cellpose') or {})
        cp['anisotropy'] = aniso
        sc2['cellpose'] = cp
        stain_cfgs[stem] = sc2
    cfg['stain_configs'] = stain_cfgs
    return cfg


# =============================================================================
# FILE DISCOVERY
# =============================================================================

def discover_channels(folder):
    found = {}
    for f in list_tiffs(folder):                       # .tif/.tiff, any case
        for suffix, _is3d in _virtual_stems(f):        # expands bundled C/T channels
            stem = f.stem + suffix
            if stem not in found:
                found[stem] = f.name
    return sorted(found.items())


def available_channel_stems(folder):
    """Set of channel stems a folder offers, including C/T-split virtual channels.

    Matches what discover_channels/load expose, so a sample folder holding a single
    multi-channel TIFF (channels '<stem>_c0', '<stem>_c1', ...) is recognized just
    like one with one file per channel.
    """
    return {stem for stem, _ in discover_channels(folder)}


def find_processable_folders(root_path, channel_roles):
    # Primary channel = nuclear when assigned, else the first stain (particle mode).
    primary_stem = get_primary_stem(channel_roles)
    if primary_stem is None:
        return []
    folders = []
    for dirpath, _, _ in os.walk(root_path):
        if Path(dirpath).name == 'Analysis':          # skip our own output folders
            continue
        # Virtual-channel aware: works for one-file-per-channel AND multi-channel files.
        if primary_stem in available_channel_stems(dirpath):
            folders.append(Path(dirpath))
    return folders


def collect_downstream_summaries(folder_path):
    summary_files = []
    for root, _, files in os.walk(folder_path):
        if Path(root).name == "Analysis":
            for f in files:
                if f == "Feature_summary.xlsx":
                    summary_files.append(Path(root) / f)
    return summary_files


# =============================================================================
# LOADING
# =============================================================================

def load_image(path):
    return io.imread(path)


def load_channels(folder):
    channels, _meta = load_channels_with_meta(folder)
    return channels


def load_channels_with_meta(folder):
    """Load every channel TIFF in a folder as canonical raw volumes + sample meta.

    Returns (channels, sample_meta):
      channels    : {stem(+C/T suffix): ndarray}  -- 2D or 3D, RAW pixels.
      sample_meta : {dimensionality, is_3d, pixel_size_um, z_size_um, unit}
                    aggregated across the folder's channel files.
    """
    folder = Path(folder)
    files = {}
    for f in list_tiffs(folder):                       # supported images, any case
        files.setdefault(f.stem, f)

    channels = {}
    is_3d_any = False
    px = pz = unit = cal_source = None
    for stem, path in sorted(files.items()):
        try:
            volumes, meta = read_image_with_meta(path)
        except Exception:
            # A metadata hiccup must never block a load: fall back to plain read.
            arr = io.imread(path)
            volumes = [('', arr)]
            meta = {'is_3d': arr.ndim >= 3, 'pixel_size_um': None,
                    'z_size_um': None, 'unit': None, 'calibration_source': None}
        is_3d_any = is_3d_any or meta['is_3d']
        if px is None:
            px = meta.get('pixel_size_um')
            cal_source = meta.get('calibration_source')
        if pz is None:
            pz = meta.get('z_size_um')
        if unit is None:
            unit = meta.get('unit')
        for suffix, vol in volumes:
            channels[stem + suffix] = vol

    sample_meta = {
        'dimensionality': '3d' if is_3d_any else '2d',
        'is_3d': is_3d_any,
        'pixel_size_um': px,
        'z_size_um': pz,
        'unit': unit,
        'calibration_source': cal_source,
    }
    return channels, sample_meta


def _bare_channel_name(suffix):
    """Bare positional channel name for a split suffix ('_c0' -> 'c0', '' -> 'c0')."""
    return suffix.lstrip('_') or 'c0'


def load_file_as_sample(path):
    """Load ONE multi-channel image as a sample: bare positional channels (c0, c1, ...).

    Used by the 'file_per_sample' input mode, where each file is a separate sample,
    so channel names must be file-independent (not '<stem>_c0') to stay consistent
    across samples for role assignment. TIFF plus any format the reader registry
    supports (ND2/CZI/LIF via aicsimageio).
    """
    volumes, meta = read_image_with_meta(path)
    channels = {_bare_channel_name(suffix): vol for suffix, vol in volumes}
    sample_meta = {
        'dimensionality': '3d' if meta['is_3d'] else '2d',
        'is_3d': meta['is_3d'],
        'pixel_size_um': meta.get('pixel_size_um'),
        'z_size_um': meta.get('z_size_um'),
        'unit': meta.get('unit'),
        'calibration_source': meta.get('calibration_source'),
    }
    return channels, sample_meta


def file_channel_names(path):
    """Bare positional channel names a single file offers (header-only)."""
    return [_bare_channel_name(suf) for suf, _ in _virtual_stems(path)]


def load_presegmented_masks(image_folder, config):
    folder = Path(image_folder)
    nuclear_labels = None
    cytoplasm_labels = None
    if config.get('use_presegmented_nuclear'):
        stem = config.get('presegmented_nuclear_stem', '')
        path = resolve_stem(folder, stem)
        if path is None:
            # Also check Analysis subfolder
            path = resolve_stem(folder / 'Analysis', stem)
        if path is not None:
            nuclear_labels = io.imread(path).astype(np.uint16)
            print(f"  Loaded pre-segmented nuclear labels from: {path}")
    if config.get('use_presegmented_cytoplasm'):
        stem = config.get('presegmented_cytoplasm_stem', '')
        path = resolve_stem(folder, stem)
        if path is None:
            path = resolve_stem(folder / 'Analysis', stem)
        if path is not None:
            cytoplasm_labels = io.imread(path).astype(np.uint16)
            print(f"  Loaded pre-segmented cytoplasm labels from: {path}")
    return nuclear_labels, cytoplasm_labels


# =============================================================================
# SAVING
# =============================================================================

def _imwrite_calibrated(path, arr, meta):
    """Write a (3D) array as an ImageJ TIFF carrying the voxel calibration."""
    px = meta.get('pixel_size_um') or 1.0
    pz = meta.get('z_size_um') or px
    unit = meta.get('unit') or 'micron'
    try:
        tifffile.imwrite(str(path), arr, imagej=True,
                         resolution=(1.0 / px, 1.0 / px),
                         metadata={'spacing': pz, 'unit': unit, 'axes': 'ZYX'})
    except Exception:
        io.imsave(path, arr, check_contrast=False)


def save_masks(analysis_path, name, mask, meta=None):
    """Save labeled mask and derived binary.  Binary-only if mask is boolean.

    The boolean dtype is the *only* signal that a mask is binary. Testing
    ``mask.max() <= 1`` instead would misread a label image holding exactly one
    object as binary and never write its ``_labeled.tif``.

    When ``meta`` carries 3D calibration, the TIFFs are written with ImageJ
    resolution/spacing tags so voxel size is preserved in the label stacks.
    With ``meta=None`` the plain writer is used and 2D output is unchanged.
    """
    def _imsave(filename, arr):
        path = analysis_path / filename
        if meta and meta.get('is_3d'):
            _imwrite_calibrated(path, arr, meta)
        else:
            io.imsave(path, arr, check_contrast=False)

    if mask.dtype == bool:
        _imsave(f"{name}_binary.tif", mask.astype(np.uint8) * 255)
    else:
        n = int(mask.max()) if mask.size else 0
        _imsave(f"{name}_labeled.tif", mask.astype(label_dtype(n)))
        _imsave(f"{name}_binary.tif", (mask > 0).astype(np.uint8) * 255)


# =============================================================================
# DATAFRAME CONSTRUCTION
# =============================================================================

def append_summary_statistics(df, id_column='cell_id', include_sem=True):
    """Append a MEAN row, and (only when ``include_sem``) a SEM row.

    SEM over individual cells is statistical pseudoreplication (cells within a
    sample are not independent replicates), so it is computed only for the
    sample-level sheets, where the unit of replication is the sample.
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != id_column]
    mean_row = {id_column: 'MEAN'}
    for col in numeric_cols:
        mean_row[col] = df[col].mean()                    # skips NaN
    extra_rows = [mean_row]
    if include_sem:
        sem_row = {id_column: 'SEM'}
        for col in numeric_cols:
            # SEM = SD / sqrt(k) over the column's OWN non-missing count (a metric
            # absent in some samples must not inflate the denominator).
            vals = df[col].dropna()
            k = len(vals)
            sem_row[col] = vals.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
        extra_rows.append(sem_row)
    return pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)


def _strip_summary_rows(df, id_column='cell_id'):
    if id_column not in df.columns:
        return df
    return df[~df[id_column].isin(['MEAN', 'SEM'])].reset_index(drop=True)


# =============================================================================
# MERGED CELLS DATAFRAME
# =============================================================================

def build_merged_cells_dataframe(
    nuclear_features, cytoplasm_features, nuc_to_cyto_map,
    stain_per_nucleus, stain_per_cytoplasm, channel_stems, stain_stems,
    stain_filter_areas=None, nuclear_stem=None, cyto_stem=None, is_3d=False,
):
    stain_filter_areas = stain_filter_areas or {}
    # Dimension-aware names: 2D measures areas (um2), 3D measures volumes (um3).
    area_key = 'volume_um3' if is_3d else 'area_um2'
    word, unit = ('volume', 'um3') if is_3d else ('area', 'um2')
    cyto_by_id = {}
    if cytoplasm_features:
        for f in cytoplasm_features:
            cyto_by_id[f['cell_id']] = f

    # Nuclei per cytoplasm territory: a cytoplasm shared by >=2 nuclei is a
    # polynucleated cell. Only meaningful when cytoplasm is enabled (in practice
    # only the Cellpose whole-cell method can put two nuclei in one territory).
    cyto_enabled = cytoplasm_features is not None
    cyto_nuc_counts = Counter(nuc_to_cyto_map.values())

    rows = []
    for nf in nuclear_features:
        cell_id = nf['cell_id']
        row = {'cell_id': cell_id}
        row.update(_prefix_and_order(nf, 'N', channel_stems, skip_stem=nuclear_stem))

        cyto_id = nuc_to_cyto_map.get(cell_id)
        cf = cyto_by_id.get(cyto_id) if cyto_id else None
        if cf:
            row.update(_prefix_and_order(cf, 'C', channel_stems, skip_stem=cyto_stem))

        # Number of nuclei sharing this cell's cytoplasm (NaN when the cell has no
        # cytoplasm -- nucleation cannot be assessed without a cell boundary).
        row['nuclei_in_cell'] = cyto_nuc_counts.get(cyto_id) if cyto_id else np.nan

        # A cell whose cytoplasm was expected (cytoplasm enabled) but not found is
        # incomplete: its whole-cell (_cell_*) measurements cannot represent a whole
        # cell, so they are blanked below rather than falling back to nucleus-only.
        cell_incomplete = cyto_enabled and cf is None

        nuc_area = nf.get(area_key, 0)
        cyto_area = cf.get(area_key, 0) if cf else 0

        for stem in stain_stems:
            nuc_stain = stain_per_nucleus.get(stem, {}).get(cell_id, {})
            cyto_stain = stain_per_cytoplasm.get(stem, {}).get(cyto_id, {}) if cyto_id else {}
            nuc_s_area = nuc_stain.get('stain_measure_cal', 0)
            nuc_s_count = nuc_stain.get('stain_particle_count', 0)
            cyto_s_area = cyto_stain.get('stain_measure_cal', 0)
            nuc_s_px = nuc_stain.get('stain_area_px', 0)
            cyto_s_px = cyto_stain.get('stain_area_px', 0)

            # particle_avg is the mean NUCLEAR stain-particle area/volume.
            cyto_s_count = cyto_stain.get('stain_particle_count', 0)
            # Whole cell = nucleus union cytoplasm (disjoint, so measures/counts add).
            total_stain = nuc_s_area + cyto_s_area            # stain area/volume per cell
            total_area = nuc_area + cyto_area
            particle_count_cell = nuc_s_count + cyto_s_count

            # Mean stain-particle area/volume: per nucleus and over the whole cell.
            particle_avg_nuc = nuc_s_area / nuc_s_count if nuc_s_count > 0 else 0.0
            particle_avg_cell = total_stain / particle_count_cell if particle_count_cell > 0 else 0.0
            # Coverage fractions: stain measure / region measure (dimensionless, 0..1).
            coverage_nuc = nuc_s_area / nuc_area if nuc_area > 0 else np.nan
            coverage_cyto = cyto_s_area / cyto_area if cyto_area > 0 else np.nan
            coverage_cell = total_stain / total_area if total_area > 0 else np.nan
            # Absolute cytoplasm stain measure is NaN when this cell has no cytoplasm.
            measure_cyto = cyto_s_area if cf else np.nan

            # Whole-cell stain presence over nucleus union cytoplasm. A cell
            # "contains" the stain when the stain area inside it reaches one average
            # particle for this image (mean particle area in px, computed over all
            # particles of the stain channel). Drives Filtered_cells. Falls back to
            # ">0" only when the image has no particles (threshold 0).
            cell_stain_px = nuc_s_px + cyto_s_px
            filter_area = stain_filter_areas.get(stem, 0)
            cell_has_stain = (cell_stain_px >= filter_area if filter_area > 0
                              else cell_stain_px > 0)

            # Whole-cell measurements are blanked for an incomplete cell (cytoplasm
            # enabled but none found); the presence flags below stay populated so the
            # analysis filter (which reads cell_has_stain) is unaffected.
            row[f'stain_{stem}_{word}_nuc_{unit}'] = nuc_s_area
            row[f'stain_{stem}_{word}_cyto_{unit}'] = measure_cyto
            row[f'stain_{stem}_{word}_cell_{unit}'] = np.nan if cell_incomplete else total_stain
            row[f'stain_{stem}_particle_count_cell'] = (
                np.nan if cell_incomplete else int(particle_count_cell))
            row[f'stain_{stem}_particle_avg_{word}_nuc_{unit}'] = particle_avg_nuc
            row[f'stain_{stem}_particle_avg_{word}_cell_{unit}'] = (
                np.nan if cell_incomplete else particle_avg_cell)
            row[f'stain_{stem}_coverage_fraction_nuc'] = coverage_nuc
            row[f'stain_{stem}_coverage_fraction_cyto'] = coverage_cyto
            row[f'stain_{stem}_coverage_fraction_cell'] = np.nan if cell_incomplete else coverage_cell
            row[f'stain_{stem}_cell_stain_area_px'] = (
                np.nan if cell_incomplete else int(cell_stain_px))
            row[f'stain_{stem}_nuc_has_stain'] = nuc_s_area > 0
            row[f'stain_{stem}_cell_has_stain'] = cell_has_stain

        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SAMPLE LEVEL DATAFRAME
# =============================================================================

def build_sample_level_dataframe(
    all_cells_df, nuclear_alignment, stain_spatial_data, stain_stems, is_3d=False,
    nuc_to_cyto_map=None,
):
    n = len(all_cells_df)
    word, unit = ('volume', 'um3') if is_3d else ('area', 'um2')
    rows = [
        {'metric': 'cell_count', 'value': n},
        {'metric': 'nuclear_alignment', 'value': nuclear_alignment},
    ]

    # Mono- vs poly-nucleated cell counts over unique cytoplasm territories (a cell
    # whose cytoplasm holds >=2 nuclei is polynucleated). Emitted only when cytoplasm
    # is enabled; skipped entirely otherwise so cytoplasm-less runs stay uncluttered.
    if nuc_to_cyto_map:
        counts = Counter(nuc_to_cyto_map.values())
        mono = sum(1 for v in counts.values() if v == 1)
        poly = sum(1 for v in counts.values() if v >= 2)
        rows.append({'metric': 'mononucleated_cell_count', 'value': mono})
        rows.append({'metric': 'polynucleated_cell_count', 'value': poly})
        rows.append({'metric': 'polynucleated_cell_fraction',
                     'value': poly / (mono + poly) if (mono + poly) else np.nan})

    for stem in stain_stems:
        # Fraction of cells scored stain-positive (the per-cell cell_has_stain flag,
        # the same criterion the analysis filter uses). Answers "% marker-positive
        # cells" directly; dimensionless, so the name is identical in 2D and 3D.
        pos_col = f'stain_{stem}_cell_has_stain'
        if pos_col in all_cells_df.columns:
            flags = all_cells_df[pos_col].dropna()
            rows.append({'metric': f'stain_{stem}_positive_fraction_cell',
                         'value': flags.mean() if len(flags) else np.nan})

        for col in (f'stain_{stem}_{word}_cell_{unit}',
                    f'stain_{stem}_particle_count_cell',
                    f'stain_{stem}_particle_avg_{word}_nuc_{unit}',
                    f'stain_{stem}_particle_avg_{word}_cell_{unit}',
                    f'stain_{stem}_coverage_fraction_cell'):
            if col in all_cells_df.columns:
                vals = all_cells_df[col].dropna()
                nv = len(vals)
                rows.append({'metric': f'{col}_mean',
                             'value': vals.mean() if nv else np.nan})
                # SEM = sample SD (ddof=1) / sqrt(n); undefined for n<2.
                rows.append({'metric': f'{col}_sem',
                             'value': vals.std(ddof=1) / np.sqrt(nv) if nv > 1 else np.nan})

        spatial = stain_spatial_data.get(stem, {})
        for key, metric_base in [('particle_alignment', 'alignment'),
                                  ('freq_major_axis', 'periodicity_major'),
                                  ('freq_minor_axis', 'periodicity_minor')]:
            vals = [d[key] for d in spatial.values() if not np.isnan(d.get(key, np.nan))]
            if vals:
                nv = len(vals)
                rows.append({'metric': f'stain_{stem}_{metric_base}_mean',
                             'value': np.mean(vals)})
                # Match the ddof=1 SEM convention used elsewhere in the output.
                rows.append({'metric': f'stain_{stem}_{metric_base}_sem',
                             'value': np.std(vals, ddof=1) / np.sqrt(nv) if nv > 1 else np.nan})

    return pd.DataFrame(rows)


# =============================================================================
# PER-IMAGE QC METRICS
# =============================================================================

def build_qc_metrics(channels, config):
    """Per-image acquisition-QC rows (metric/value), threshold-aware.

    Image-level and independent of cell filtering, so the same rows are appended
    to both the Sample_level and Filtered_sample_level sheets. Reports raw
    intensity percentiles/mean and % saturated for every channel, plus the
    fraction of pixels inside the segmentation threshold band for the nuclear
    and stain channels (the key cross-sample comparability indicator).
    """
    channel_roles = config.get('channel_roles', {})
    stain_configs = config.get('stain_configs', {})
    rows = []
    for stem in sorted(channels.keys()):
        role = channel_roles.get(stem, 'none')
        thr_min = thr_max = None
        # Threshold band only meaningful when that channel is segmented by threshold.
        if role == 'nuclear' and config.get('nuclear_method', 'threshold') == 'threshold':
            thr_min = config.get('threshold_min')
            thr_max = config.get('threshold_max')
        elif role == 'stain':
            sc = stain_configs.get(stem, DEFAULT_STAIN_CONFIG)
            if sc.get('method', 'threshold') == 'threshold':
                thr_min = sc.get('threshold_min')
                thr_max = sc.get('threshold_max')
        qc = compute_channel_qc(channels[stem], threshold_min=thr_min, threshold_max=thr_max)
        for k, v in qc.items():
            rows.append({'metric': f'qc_{stem}_{k}', 'value': v})
    return rows


def build_depth_profiles(channels, nuclear_labels, stain_binaries, z_size_um):
    """Per-z (depth) QC profiles for a 3D sample; ``None`` for 2D.

    Whole-mount imaging is prone to depth-dependent bias -- signal attenuation and
    bleaching, incomplete stain penetration, and objects clipped by the top/bottom
    of the stack. This returns, per z-plane: each channel's mean intensity, the
    nuclear foreground fraction (object density vs depth) and each stain's coverage
    fraction, plus the count of nuclei truncated by the first/last plane. Consumed by
    the QC report's depth page; cheap axis-reductions, so safe to always compute in 3D.
    """
    ref = next((np.asarray(img) for img in channels.values()
                if np.asarray(img).ndim == 3), None)
    if ref is None:
        return None
    Z, H, W = ref.shape
    plane = float(H * W)
    prof = {
        'n_z': int(Z),
        'z_um': [round(i * float(z_size_um or 1.0), 4) for i in range(Z)],
        'z_calibrated': bool(z_size_um),
        'channel_intensity_by_z': {
            stem: [round(float(v), 3) for v in np.asarray(img).mean(axis=(1, 2))]
            for stem, img in channels.items() if np.asarray(img).ndim == 3},
    }
    if nuclear_labels is not None and nuclear_labels.ndim == 3:
        fg = nuclear_labels > 0
        prof['nuclear_foreground_by_z'] = [round(float(v), 6)
                                           for v in fg.sum(axis=(1, 2)) / plane]
        regs = measure.regionprops(nuclear_labels)
        prof['n_nuclei'] = int(len(regs))
        prof['n_nuclei_z_truncated'] = int(sum(1 for r in regs
                                               if r.bbox[0] == 0 or r.bbox[3] >= Z))
    if stain_binaries:
        prof['stain_coverage_by_z'] = {
            stem: [round(float(v), 6) for v in np.asarray(sb).sum(axis=(1, 2)) / plane]
            for stem, sb in stain_binaries.items() if np.asarray(sb).ndim == 3}
    return prof


# =============================================================================
# CELL FILTER
# =============================================================================

# -----------------------------------------------------------------------------
# OUTLIER SCREEN
#
# One metric, one vote per channel.
#
# The metric is log10(size) -- area_um2 in 2D, volume_um3 in 3D -- for every channel's
# objects, whatever its role: nuclei, cytoplasm regions, or a stain's particles.
#
# Why the log. The robust z-score 0.6745*(x-median)/MAD is calibrated so |z| behaves
# like a standard normal deviate (0.6745 = 1/1.4826, the MAD->SD factor for normal
# data), which is what makes a threshold of 4 or 5 mean anything. Particle areas are
# strongly right-skewed (skewness ~5), and on the raw scale that calibration collapses:
# |z|>5 fires on ~13% of perfectly ordinary puncta, because there it merely means
# "area > 7.6x the median". Worse, size > 0 bounds the raw lower tail (z -> -0.76 as
# area -> 0), so a two-sided test is silently one-sided. log10 restores both the
# calibration and the lower tail, and is free for near-symmetric objects like nuclei.
#
# Why size alone. Shape is unresolvable at particle scale: solidity, eccentricity and
# convexity_defects all have MAD = 0 on puncta (so the mad==0 guard skips them anyway),
# and 4*pi*A/P**2 exceeds 1 for small digital objects whose perimeter is underestimated.
# One metric per channel also means every channel weighs in equally -- screening several
# columns and OR-ing them gives a channel more chances to flag, so more metrics silently
# means more removals.
#
# Two-sided: the upper tail catches merged blobs and doublet nuclei, the lower tail
# catches fragments.
# -----------------------------------------------------------------------------

def _build_cell_territory(nuclear_labels, cytoplasm_labels, nuc_to_cyto_map):
    """Label image carrying each cell's id over its whole territory (nucleus + its
    cytoplasm). Cytoplasm regions with no mapped nucleus are dropped, as elsewhere."""
    territory = np.asarray(nuclear_labels).astype(np.uint32, copy=True)
    if cytoplasm_labels is None or not nuc_to_cyto_map:
        return territory
    lut = np.zeros(int(cytoplasm_labels.max()) + 1, dtype=np.uint32)
    for nuc_id, cyto_id in nuc_to_cyto_map.items():
        if 0 < cyto_id < lut.size:
            lut[cyto_id] = nuc_id
    mapped = lut[cytoplasm_labels]
    fill = (territory == 0) & (mapped > 0)      # nuclei win; the two are disjoint anyway
    territory[fill] = mapped[fill]
    return territory


def _robust_z_log(sizes):
    """Two-sided robust z of log10(size). NaN where size <= 0 (never flags)."""
    sizes = np.asarray(sizes, dtype=float)
    logs = np.full(sizes.shape, np.nan)
    pos = sizes > 0
    logs[pos] = np.log10(sizes[pos])
    med = np.nanmedian(logs)
    mad = np.nanmedian(np.abs(logs - med))
    if not np.isfinite(mad) or mad == 0:
        return None, med, mad          # constant metric must never flag everything
    return 0.6745 * (logs - med) / mad, med, mad


def flag_size_outlier_cells(channel_objects, threshold):
    """One vote per channel: which cells each channel condemns, and why.

    ``channel_objects`` maps a channel name -> list of ``(cell_id, size)`` for that
    channel's segmented objects (one nucleus per cell, one cytoplasm per cell, many
    particles per cell). Within each channel the sizes are pooled, log10-transformed and
    screened with a two-sided robust z; a cell is flagged by that channel when it owns
    at least one flagged object. A cell is removed when ANY channel flags it, so every
    channel carries exactly one binary vote.

    Returns ``(flagged_by_channel, stats_by_channel)``; the stats describe the size
    distribution the screen actually saw, so an over-firing channel is visible in QC.
    """
    flagged_by_channel, stats_by_channel = {}, {}
    for channel, objects in channel_objects.items():
        cell_ids = np.array([c for c, _ in objects])
        sizes = np.array([s for _, s in objects], dtype=float)
        flagged = set()
        stats = {'n_objects': int(len(objects)), 'n_objects_flagged': 0,
                 'flag_rate': 0.0, 'n_cells_flagged': 0,
                 'median_size': None, 'mad_log10': None, 'z_min': None, 'z_max': None}
        if len(objects):
            z, med, mad = _robust_z_log(sizes)
            stats['median_size'] = float(10 ** med) if np.isfinite(med) else None
            stats['mad_log10'] = float(mad) if np.isfinite(mad) else None
            if z is not None:
                is_out = np.isfinite(z) & (np.abs(z) > threshold)
                flagged = {int(c) for c in cell_ids[is_out]}
                finite = z[np.isfinite(z)]
                stats.update(
                    n_objects_flagged=int(is_out.sum()),
                    flag_rate=float(is_out.mean()),
                    n_cells_flagged=len(flagged),
                    z_min=(float(finite.min()) if finite.size else None),
                    z_max=(float(finite.max()) if finite.size else None))
        flagged_by_channel[channel] = flagged
        stats_by_channel[channel] = stats
    return flagged_by_channel, stats_by_channel


def flag_size_outlier_rows(sizes, threshold):
    """Boolean array: two-sided log-size robust-z outliers. For particle mode, where
    the rows are the objects themselves and there are no cells to remove."""
    sizes = np.asarray(sizes, dtype=float)
    if sizes.size == 0:
        return np.zeros(0, dtype=bool)
    z, _, _ = _robust_z_log(sizes)
    if z is None:
        return np.zeros(sizes.shape, dtype=bool)
    return np.isfinite(z) & (np.abs(z) > threshold)


def apply_cell_filter(all_cells_df, config):
    filter_stain = config.get('filter_stain')
    filter_condition = config.get('filter_condition', 'contains')
    if not filter_stain:
        return all_cells_df.copy()
    # Keep cells by whole-cell presence of the selected stain's binary mask
    # (>= that stain's min object size, in px, over nucleus union cytoplasm).
    col = f'stain_{filter_stain}_cell_has_stain'
    if col not in all_cells_df.columns:
        return all_cells_df.copy()
    if filter_condition == 'contains':
        mask = all_cells_df[col] == True
    else:
        mask = (all_cells_df[col] == False) | all_cells_df[col].isna()
    return all_cells_df[mask].reset_index(drop=True)


# =============================================================================
# EXCEL OUTPUT
# =============================================================================

def save_feature_summary(analysis_path, sheets_dict):
    output_file = analysis_path / "Feature_summary.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in sheets_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


# =============================================================================
# AGGREGATION
# =============================================================================

def generate_aggregated_summary(folder_path):
    summary_files = collect_downstream_summaries(folder_path)
    if not summary_files:
        return
    print(f"\nGenerating aggregated summary for: {folder_path}")
    print(f"Found {len(summary_files)} downstream sample(s)")

    all_cells_dfs, filtered_cells_dfs, outlier_cells_dfs = [], [], []
    all_particles_dfs = []                                 # particle mode (no nuclei)
    sample_levels, filtered_sample_levels, outlier_sample_levels = [], [], []

    for sf in summary_files:
        sample_name = sf.parent.parent.name
        excel = pd.ExcelFile(sf)
        for sheet, target, id_col in [
            ('All_cells', all_cells_dfs, 'cell_id'),
            ('All_particles', all_particles_dfs, 'cell_id'),
            ('Filtered_cells', filtered_cells_dfs, 'cell_id'),
            ('Outlier_removed', outlier_cells_dfs, 'cell_id'),
        ]:
            if sheet in excel.sheet_names:
                df = _strip_summary_rows(pd.read_excel(sf, sheet_name=sheet), id_col)
                if not df.empty:
                    df.insert(0, 'sample_name', sample_name)
                    target.append(df)
        for sheet, target in [
            ('Sample_level', sample_levels),
            ('Filtered_sample_level', filtered_sample_levels),
            ('Outlier_removed_sample_level', outlier_sample_levels),
        ]:
            if sheet in excel.sheet_names:
                df = pd.read_excel(sf, sheet_name=sheet)
                if not df.empty:
                    wide = {'sample_name': sample_name}
                    for _, row in df.iterrows():
                        wide[row['metric']] = row['value']
                    target.append(wide)

    if not any([all_cells_dfs, all_particles_dfs, filtered_cells_dfs, outlier_cells_dfs,
                sample_levels, filtered_sample_levels, outlier_sample_levels]):
        print(f"No data to aggregate at: {folder_path}")
        return

    output_file = folder_path / "Aggregated_summary.xlsx"
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for dfs, sheet, id_col in [
            (all_cells_dfs, 'All_cells', 'cell_id'),
            (all_particles_dfs, 'All_particles', 'cell_id'),
            (filtered_cells_dfs, 'Filtered_cells', 'cell_id'),
            (outlier_cells_dfs, 'Outlier_removed', 'cell_id'),
        ]:
            if dfs:
                combined = append_summary_statistics(
                    pd.concat(dfs, ignore_index=True), id_col, include_sem=False)
                combined.to_excel(writer, sheet_name=sheet, index=False)
        for levels, sheet, id_col in [
            (sample_levels, 'Sample_level', 'sample_name'),
            (filtered_sample_levels, 'Filtered_sample_level', 'sample_name'),
            (outlier_sample_levels, 'Outlier_removed_sample_level', 'sample_name'),
        ]:
            if levels:
                df = append_summary_statistics(pd.DataFrame(levels), id_col)
                df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Aggregated summary saved to: {output_file}")


def generate_all_hierarchical_summaries(root_path):
    print("\n" + "=" * 80)
    print("GENERATING HIERARCHICAL SUMMARIES")
    print("=" * 80)
    all_dirs = []
    for dirpath, _, _ in os.walk(root_path):
        dp = Path(dirpath)
        if dp.name != "Analysis":
            all_dirs.append(dp)
    all_dirs.sort(key=lambda x: len(x.parts), reverse=True)
    for dp in all_dirs:
        if collect_downstream_summaries(dp):
            generate_aggregated_summary(dp)


# =============================================================================
# ORCHESTRATION: SEGMENTATION
# =============================================================================

def run_segmentation(image_folder, channels, config, cellpose_model=None, meta=None):
    """Segment nuclei and cytoplasm.  Saves masks to Analysis/.
    Returns (nuclear_labels, cytoplasm_labels).  `meta` carries 3D calibration so
    saved label stacks keep their voxel size.
    """
    image_folder = Path(image_folder)
    analysis_path = image_folder / "Analysis"
    analysis_path.mkdir(exist_ok=True)
    channel_roles = config['channel_roles']

    # --- Nuclear ---
    nuclear_stem = get_role_stem(channel_roles, 'nuclear')
    detection_image = channels[nuclear_stem]

    print("\n--- NUCLEAR SEGMENTATION ---")
    labeled, valid_labels, seg_info = segment_nuclei(detection_image, config)
    if not valid_labels:
        print("No valid nuclei found.")
        return None, None, None
    save_masks(analysis_path, 'nuclear', labeled, meta=meta)
    print(f"Nuclear: {seg_info['final_count']} objects "
          f"({seg_info['doublet_count']} doublets, {seg_info['split_count']} split)")

    # --- Cytoplasm ---
    cytoplasm_labels = None
    cyto_count = 0
    cytoplasm_source = config.get('cytoplasm_source', 'none')
    cyto_method = config.get('cytoplasm_method', 'cellpose')

    # Pre-segmented cytoplasm is loaded separately (process_sample); the 'none'
    # method disables cytoplasm. Only 'threshold'/'cellpose' segment here.
    if cytoplasm_source != 'none' and cyto_method not in ('presegmented', 'none'):
        if cytoplasm_source == 'membrane':
            stem = get_role_stem(channel_roles, 'membrane')
        else:  # 'channel'
            stem = get_role_stem(channel_roles, 'cytoplasm')

        if stem and stem in channels:
            print(f"\n--- CYTOPLASM SEGMENTATION ({cyto_method}, source={cytoplasm_source}) ---")
            if cyto_method == 'threshold':
                cytoplasm_labels, cyto_info = segment_cytoplasm_threshold(
                    channels[stem], labeled, config, source=cytoplasm_source,
                )
            else:
                cytoplasm_labels, cyto_info = segment_cytoplasm_cellpose(
                    channels[stem], labeled, config, model=cellpose_model,
                )
            save_masks(analysis_path, 'cytoplasm', cytoplasm_labels, meta=meta)
            cyto_count = cyto_info['n_cytoplasm']
            print(f"Cytoplasm: {cyto_count} regions")
        else:
            print(f"Warning: no channel for cytoplasm_source='{cytoplasm_source}'")

    seg_meta = {
        'nuclear_method': config.get('nuclear_method', 'threshold'),
        'nuclear': seg_info,
        'cytoplasm_source': cytoplasm_source,
        'cytoplasm_method': (config.get('cytoplasm_method', 'cellpose')
                             if cytoplasm_source != 'none' else None),
        'cytoplasm_count': cyto_count,
    }
    return labeled, cytoplasm_labels, seg_meta


# =============================================================================
# ORCHESTRATION: ANALYSIS
# =============================================================================

def run_analysis(image_folder, channels, nuclear_labels, cytoplasm_labels, config,
                 seg_meta=None, meta=None):
    """Extract features and quantify stains.  Saves Feature_summary.xlsx."""
    image_folder = Path(image_folder)
    analysis_path = image_folder / "Analysis"
    analysis_path.mkdir(exist_ok=True)
    channel_roles = config['channel_roles']
    stain_configs = config.get('stain_configs', {})
    channel_stems = sorted(channels.keys())
    is_3d = nuclear_labels.ndim == 3
    z_size = config.get('z_size_um')
    _px = config['pixel_size_um']
    # (z_um, y_um, x_um) voxel spacing for anisotropy-correct 3D alignment; None in 2D.
    align_spacing = ((z_size or _px), _px, _px) if is_3d else None

    nuclear_stem = get_role_stem(channel_roles, 'nuclear')
    detection_image = channels[nuclear_stem]

    # --- Nuclear features --- (GLCM texture is always computed)
    nuclear_features = extract_all_nuclear_features(
        nuclear_labels, detection_image, channels, config,
    )
    nuclear_alignment = compute_nuclear_alignment(nuclear_labels, spacing=align_spacing)
    print(f"Nuclear alignment: {nuclear_alignment:.4f}")

    # --- Cytoplasm features ---
    cytoplasm_features = None
    nuc_to_cyto_map = {}
    cyto_stem = None
    if cytoplasm_labels is not None:
        cyto_source = config.get('cytoplasm_source', 'none')
        if cyto_source == 'membrane':
            cyto_stem = get_role_stem(channel_roles, 'membrane')
        elif cyto_source == 'channel':
            cyto_stem = get_role_stem(channel_roles, 'cytoplasm')
        else:
            cyto_stem = None
        cyto_intensity = channels.get(cyto_stem, detection_image) if cyto_stem else detection_image
        # Skip cytoplasm texture for membrane-derived cytoplasm (membrane signal is
        # at boundaries, so its GLCM texture is not a meaningful cytoplasmic feature).
        cytoplasm_features = extract_all_cytoplasm_features(
            cytoplasm_labels, cyto_intensity, channels, config,
            compute_texture=(cyto_source != 'membrane'),
        )
        nuc_to_cyto_map = map_nuclei_to_cytoplasm(nuclear_labels, cytoplasm_labels)
        print(f"Extracted features for {len(cytoplasm_features)} cytoplasm regions")

    # Each cell's whole territory under its own id, so a stain particle can be assigned
    # to the cell owning most of its pixels (see quantify_stain_particles).
    cell_territory = (_build_cell_territory(nuclear_labels, cytoplasm_labels, nuc_to_cyto_map)
                      if config.get('remove_outliers', True) else None)

    # --- Stains --- ('none'-method stains stay measured channels, not segmented)
    stain_stems = active_stain_stems(channel_roles, stain_configs)
    stain_per_nucleus = {}
    stain_per_cytoplasm = {}
    stain_spatial_data = {}
    stain_particle_feats = {}       # stem -> per-particle size/shape (parent = cell_id)
    stain_filter_areas = {}         # stem -> mean particle area (px) = analysis-filter threshold
    stain_binaries = {}             # stem -> binary mask (3D only; for depth-QC profiles)

    for stem in stain_stems:
        if stem not in channels:
            continue
        sc = stain_configs.get(stem, DEFAULT_STAIN_CONFIG.copy())

        print(f"\n--- STAIN: {stem} ({sc.get('output_type','binary')}) ---")
        stain_mask, stain_info = segment_stain(
            channels[stem], sc,
            edge_margin=config.get('edge_exclusion_margin', 0))
        # Derive from the returned output_type so Cellpose stains (always labelled)
        # are handled correctly regardless of the threshold-mode output_type setting.
        is_binary = stain_info.get('output_type', 'binary') == 'binary'
        save_masks(analysis_path, f'stain_{stem}', stain_mask, meta=meta)

        stain_binary = stain_mask.astype(bool) if is_binary else (stain_mask > 0)

        # Per-image analysis-filter threshold: mean area (px) of one stain particle
        # over ALL particles of this channel. A cell "contains" the stain when its
        # own stain area reaches this (see build_merged_cells_dataframe).
        stain_filter_areas[stem] = mean_particle_area_px(stain_binary)
        if is_3d:
            stain_binaries[stem] = stain_binary          # for per-z coverage profiles

        per_nuc = quantify_stain_per_labels(
            stain_binary, nuclear_labels, config['pixel_size_um'], z_size_um=z_size)
        stain_per_nucleus[stem] = {d['label_id']: d for d in per_nuc}

        if cytoplasm_labels is not None:
            per_cyto = quantify_stain_per_labels(
                stain_binary, cytoplasm_labels, config['pixel_size_um'], z_size_um=z_size,
            )
            stain_per_cytoplasm[stem] = {d['label_id']: d for d in per_cyto}

        spatial = compute_stain_spatial_analysis(
            stain_binary, nuclear_labels, config['pixel_size_um'],
        )
        stain_spatial_data[stem] = {d['label_id']: d for d in spatial}

        # Per-particle sizes for the outlier screen: one row per *physical* particle,
        # owned by the cell holding most of its pixels. Only needed when outlier removal
        # is on, so skip the cost otherwise.
        if cell_territory is not None:
            stain_particle_feats[stem] = quantify_stain_particles(
                stain_binary, cell_territory, config['pixel_size_um'], z_size_um=z_size)

    # --- Build output ---
    all_cells_df = build_merged_cells_dataframe(
        nuclear_features, cytoplasm_features, nuc_to_cyto_map,
        stain_per_nucleus, stain_per_cytoplasm, channel_stems, stain_stems,
        stain_filter_areas=stain_filter_areas,
        nuclear_stem=nuclear_stem, cyto_stem=(cyto_stem or nuclear_stem), is_3d=is_3d,
    )
    sample_level_df = build_sample_level_dataframe(
        all_cells_df, nuclear_alignment, stain_spatial_data, stain_stems, is_3d=is_3d,
        nuc_to_cyto_map=nuc_to_cyto_map,
    )
    filtered_cells_df = apply_cell_filter(all_cells_df, config)

    # Per-image QC rows are acquisition-level (identical across all cell tiers).
    qc_df = pd.DataFrame(build_qc_metrics(channels, config))
    # Depth (per-z) QC profiles for 3D stacks -> QC report depth page.
    depth_profiles = (build_depth_profiles(channels, nuclear_labels, stain_binaries, z_size)
                      if is_3d else None)

    def _subset_sample_level(cells_df):
        """Sample-level sheet for a cell subset: recompute alignment + stain
        spatial on the surviving ids (so every row reflects the subset), then
        append the shared per-image QC rows."""
        ids = set(cells_df['cell_id'].tolist())
        sub_labels = np.where(np.isin(nuclear_labels, list(ids)), nuclear_labels, 0)
        alignment = compute_nuclear_alignment(sub_labels, spacing=align_spacing)
        spatial = {
            stem: {lid: d for lid, d in stain_spatial_data.get(stem, {}).items()
                   if lid in ids}
            for stem in stain_stems
        }
        # Recount nucleation over surviving cells only, like alignment/spatial above.
        sub_map = {nuc: cyto for nuc, cyto in nuc_to_cyto_map.items() if nuc in ids}
        df = build_sample_level_dataframe(
            cells_df, alignment, spatial, stain_stems, is_3d=is_3d,
            nuc_to_cyto_map=sub_map)
        if not qc_df.empty:
            df = pd.concat([df, qc_df], ignore_index=True)
        return df

    if not qc_df.empty:
        sample_level_df = pd.concat([sample_level_df, qc_df], ignore_index=True)
    filtered_sample_df = _subset_sample_level(filtered_cells_df)

    sheets = {
        'All_cells': append_summary_statistics(all_cells_df, 'cell_id', include_sem=False),
        'Sample_level': sample_level_df,
        'Filtered_cells': append_summary_statistics(filtered_cells_df, 'cell_id', include_sem=False),
        'Filtered_sample_level': filtered_sample_df,
    }

    # Outlier-removed tier: one vote per channel on the filtered set, giving the
    # progression all cells -> filtered -> outlier-removed. Two extra sheets that
    # mirror the existing tiers (cells + sample-level).
    outlier_removed_df = None
    outlier_stats = {}
    outlier_by_channel = {}
    if config.get('remove_outliers', True):
        thr = config.get('outlier_mad_threshold', 4.0)
        size_key = particle_size_key(3 if is_3d else 2)
        # Every channel contributes the same metric on its own objects: one nucleus and
        # one cytoplasm per cell, many particles per cell for a stain.
        channel_objects = {
            'nuclear': [(f['cell_id'], f[size_key]) for f in nuclear_features
                        if f.get(size_key) is not None],
        }
        if cytoplasm_features and nuc_to_cyto_map:
            cyto_to_nuc = {c: n for n, c in nuc_to_cyto_map.items()}
            channel_objects['cytoplasm'] = [
                (cyto_to_nuc[f['cell_id']], f[size_key]) for f in cytoplasm_features
                if f['cell_id'] in cyto_to_nuc and f.get(size_key) is not None]
        for stem, parts in stain_particle_feats.items():
            channel_objects[f'stain:{stem}'] = [(p['parent_label'], p[size_key]) for p in parts]

        per_channel_flagged, outlier_stats = flag_size_outlier_cells(channel_objects, thr)
        flagged_cells = set().union(*per_channel_flagged.values()) if per_channel_flagged else set()
        outlier_mask = filtered_cells_df['cell_id'].isin(flagged_cells)
        # Per-channel attribution, restricted to the filtered set. Channels overlap, so
        # these votes need not partition the removals.
        filt_ids = set(filtered_cells_df['cell_id'])
        outlier_by_channel = {ch: int(len(ids & filt_ids))
                              for ch, ids in per_channel_flagged.items()}
        outlier_removed_df = filtered_cells_df[~outlier_mask].reset_index(drop=True)
        sheets['Outlier_removed'] = append_summary_statistics(
            outlier_removed_df, 'cell_id', include_sem=False)
        sheets['Outlier_removed_sample_level'] = _subset_sample_level(outlier_removed_df)

    save_feature_summary(analysis_path, sheets)
    _write_qc_sidecar(analysis_path, image_folder.name, config, channels,
                      all_cells_df, filtered_cells_df, nuclear_alignment,
                      stain_stems, seg_meta, outlier_removed_df=outlier_removed_df,
                      outlier_by_channel=outlier_by_channel, outlier_stats=outlier_stats,
                      filter_areas=stain_filter_areas, depth_profiles=depth_profiles)
    n_out = ''
    if outlier_removed_df is not None:
        votes = ", ".join(f"{ch}={n}" for ch, n in sorted(outlier_by_channel.items()) if n)
        n_out = (f", {len(outlier_removed_df)} outlier-removed"
                 + (f" (flagged by {votes})" if votes else ""))
    print(f"\nFeature summary: {len(all_cells_df)} total, {len(filtered_cells_df)} filtered{n_out}")
    print(f"  -> {analysis_path / 'Feature_summary.xlsx'}")


def _write_qc_sidecar(analysis_path, sample_name, config, channels,
                      all_cells_df, filtered_cells_df, nuclear_alignment,
                      stain_stems, seg_meta, outlier_removed_df=None,
                      outlier_by_channel=None, outlier_stats=None, filter_areas=None,
                      depth_profiles=None):
    """Persist per-sample QC info not recoverable from the Excel alone (chiefly the
    doublet/split counts and a config snapshot), for the standalone batch QC report.
    Best-effort: a failure here never aborts the analysis."""
    import json
    from datetime import datetime
    align = (None if nuclear_alignment is None or np.isnan(nuclear_alignment)
             else float(nuclear_alignment))
    # Capture the Cellpose model + parameters for each detection that used Cellpose.
    cyto_src = config.get('cytoplasm_source', 'none')
    cp = {}
    if config.get('nuclear_method', 'threshold') == 'cellpose':
        cp['nuclear'] = config.get('nuclear_cellpose', {})
    if cyto_src != 'none' and config.get('cytoplasm_method', 'cellpose') == 'cellpose':
        cp['cytoplasm'] = {
            'model_type': config.get('cellpose_model_type'),
            'diameter': config.get('cellpose_diameter'),
            'cellprob_threshold': config.get('cellpose_cellprob_threshold'),
            'flow_threshold': config.get('cellpose_flow_threshold'),
            'niter': config.get('cellpose_niter'),
            'min_size': config.get('cellpose_min_size'),
            'max_size_fraction': config.get('cellpose_max_size_fraction'),
            'gpu': config.get('cellpose_gpu'),
            'invert': config.get('cellpose_invert'),
        }
    for stem in stain_stems:
        sc = config.get('stain_configs', {}).get(stem, {})
        if sc.get('method', 'threshold') == 'cellpose':
            cp[f'stain:{stem}'] = sc.get('cellpose', {})
    # Data-derived analysis-filter threshold (mean particle area) per stain, in px
    # and calibrated; recorded so the filter is reproducible from the sidecar alone.
    filter_thresh = None
    if filter_areas:
        px_um = config.get('pixel_size_um')
        z_um = config.get('z_size_um')
        is3d = any(np.asarray(img).ndim == 3 for img in channels.values())
        cal_key = 'mean_particle_volume_um3' if is3d else 'mean_particle_area_um2'
        factor = (px_um * px_um * (z_um or px_um) if is3d else px_um * px_um) if px_um else None
        filter_thresh = {
            stem: {'mean_particle_area_px': round(float(v), 2),
                   cal_key: (round(float(v) * factor, 4) if factor else None)}
            for stem, v in filter_areas.items()
        }
    meta = {
        'sample': sample_name,
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'dimensionality': ('3d' if any(np.asarray(img).ndim == 3 for img in channels.values())
                           else '2d'),
        'pixel_size_um': config.get('pixel_size_um'),
        'z_size_um': config.get('z_size_um'),
        'calibration': config.get('_calibration'),
        'num_threads': config.get('_num_workers'),
        'channel_roles': config.get('channel_roles', {}),
        'image_shape': {stem: list(np.asarray(img).shape) for stem, img in channels.items()},
        'nuclear_method': config.get('nuclear_method', 'threshold'),
        'cytoplasm_source': cyto_src,
        'cytoplasm_method': config.get('cytoplasm_method', 'cellpose'),
        'cellpose': cp or None,
        'filter_stain': config.get('filter_stain'),
        'filter_condition': config.get('filter_condition'),
        'filter_threshold_per_stain': filter_thresh,
        'remove_outliers': bool(config.get('remove_outliers', True)),
        'outlier_mad_threshold': config.get('outlier_mad_threshold'),
        # What the screen is: one metric, two-sided, one vote per channel.
        'outlier_screen': {
            'metric': ('log10(volume_um3)' if 'N_volume_um3' in all_cells_df.columns
                       else 'log10(area_um2)'),
            'statistic': 'robust_z = 0.6745*(x-median)/MAD',
            'sided': 'two',
            'threshold': config.get('outlier_mad_threshold'),
            'aggregation': 'cell removed if flagged by any channel',
        },
        'cell_count': int(len(all_cells_df)),
        'filtered_count': int(len(filtered_cells_df)),
        'outlier_removed_count': (None if outlier_removed_df is None
                                  else int(len(outlier_removed_df))),
        # Cells condemned per channel (channels overlap; these need not partition).
        'outlier_cells_flagged_by_channel': outlier_by_channel or None,
        # The size distribution each channel's screen actually saw.
        'outlier_size_distribution_by_channel': outlier_stats or None,
        'nuclear_alignment': align,
        'stain_stems': list(stain_stems),
        'segmentation': seg_meta,
        'depth_profiles': depth_profiles,
    }
    try:
        with open(Path(analysis_path) / 'qc_meta.json', 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, indent=2, default=str)
    except Exception as exc:
        print(f"  Warning: could not write qc_meta.json: {exc}")


# =============================================================================
# ORCHESTRATION: PARTICLE ANALYSIS  (no nuclear channel)
# =============================================================================

def _particle_sample_level(df, stain_stems, stain_labels, config, is_3d):
    """Per-stain sample-level rows for particle mode, computed on the particles in
    ``df`` (the full set, or an outlier-removed subset). ``stain_labels`` maps each
    stem to its label image, so coverage/alignment reflect exactly the kept
    particles. Mirrors the metric/value long form used by the cell path."""
    word, unit = ('volume', 'um3') if is_3d else ('area', 'um2')
    area_col = f'{word}_{unit}'                          # 'area_um2' | 'volume_um3'
    z = config.get('z_size_um')
    px = config['pixel_size_um']
    factor = px * px if z is None else px * px * z
    align_spacing = ((z or px), px, px) if is_3d else None
    rows = []
    for stem in stain_stems:
        labels = stain_labels.get(stem)
        if labels is None:
            continue
        img_px = int(np.asarray(labels).size)
        sub = df[df['stain'] == stem] if 'stain' in df.columns else df.iloc[0:0]
        keep_ids = set(int(v) for v in sub['label_id']) if 'label_id' in sub.columns else set()
        kept = (np.where(np.isin(labels, list(keep_ids)), labels, 0)
                if keep_ids else np.zeros_like(labels))
        areas = (sub[area_col].to_numpy(dtype=float)
                 if area_col in sub.columns else np.array([]))
        areas = areas[~np.isnan(areas)]
        stain_px = int(np.count_nonzero(kept > 0))
        align = compute_nuclear_alignment(kept, spacing=align_spacing)
        rows += [
            {'metric': f'stain_{stem}_particle_count', 'value': len(sub)},
            {'metric': f'stain_{stem}_{word}_total_{unit}', 'value': float(areas.sum())},
            {'metric': f'stain_{stem}_{word}_mean_{unit}',
             'value': float(areas.mean()) if areas.size else np.nan},
            {'metric': f'stain_{stem}_{word}_mean_{unit}_sem',
             'value': float(areas.std(ddof=1) / np.sqrt(areas.size)) if areas.size > 1 else np.nan},
            {'metric': f'stain_{stem}_coverage_fraction_image',
             'value': stain_px / img_px if img_px else np.nan},
            {'metric': f'stain_{stem}_alignment',
             'value': None if np.isnan(align) else float(align)},
        ]
    return pd.DataFrame(rows)


def run_particle_analysis(image_folder, channels, config, meta=None):
    """Nucleus-free 'particle mode': analyse each stain channel's segmented
    instances as standalone particles. Writes Feature_summary.xlsx with one row per
    particle (across all stains, `stain` column) + per-stain sample-level summaries,
    plus the outlier-removed tier. No cytoplasm and no cell filter (both need
    nuclei). Reuses segment_stain / extract_all_nuclear_features / the shared
    dataframe + Excel helpers."""
    image_folder = Path(image_folder)
    analysis_path = image_folder / "Analysis"
    analysis_path.mkdir(exist_ok=True)
    channel_roles = config['channel_roles']
    stain_configs = config.get('stain_configs', {})
    channel_stems = sorted(channels.keys())
    stain_stems = active_stain_stems(channel_roles, stain_configs)
    is_3d = any(np.asarray(img).ndim == 3 for img in channels.values())

    print("\n--- PARTICLE MODE (no nuclear channel) ---")
    rows = []
    stain_labels = {}
    seg_summary = {}
    next_id = 1
    for stem in stain_stems:
        if stem not in channels:
            continue
        sc = stain_configs.get(stem, DEFAULT_STAIN_CONFIG.copy())
        stain_mask, stain_info = segment_stain(
            channels[stem], sc, edge_margin=config.get('edge_exclusion_margin', 0))
        is_binary = stain_info.get('output_type', 'binary') == 'binary'
        save_masks(analysis_path, f'stain_{stem}', stain_mask, meta=meta)
        # Per-particle features require labelled instances (label a binary mask).
        labels = (measure.label(stain_mask, connectivity=2) if is_binary
                  else stain_mask)          # already an instance-label image
        stain_labels[stem] = labels
        feats = extract_all_nuclear_features(labels, channels[stem], channels, config)
        for f in feats:
            row = {'cell_id': next_id, 'stain': stem, 'label_id': int(f['cell_id'])}
            row.update(_particle_feature_columns(f, stem, channel_stems))
            rows.append(row)
            next_id += 1
        seg_summary[stem] = {'particle_count': len(feats),
                             'output_type': stain_info.get('output_type', 'binary')}
        print(f"  {stem}: {len(feats)} particles")

    particles_df = pd.DataFrame(rows)

    qc_df = pd.DataFrame(build_qc_metrics(channels, config))

    def _sample_level(cells_df):
        df = _particle_sample_level(cells_df, stain_stems, stain_labels, config, is_3d)
        if not qc_df.empty:
            df = pd.concat([df, qc_df], ignore_index=True)
        return df

    sheets = {
        'All_particles': append_summary_statistics(particles_df, 'cell_id', include_sem=False)
                         if not particles_df.empty else particles_df,
        'Sample_level': _sample_level(particles_df),
    }

    outlier_removed_df = None
    outlier_stats = outlier_by_channel = {}
    if config.get('remove_outliers', True) and not particles_df.empty:
        thr = config.get('outlier_mad_threshold', 4.0)
        size_key = particle_size_key(3 if is_3d else 2)
        # Same screen as cell mode, per stain channel. Here the objects ARE the rows,
        # so a flagged particle drops itself rather than a cell.
        channel_objects = {f'stain:{stem}': list(zip(grp['cell_id'], grp[size_key]))
                           for stem, grp in particles_df.groupby('stain')}
        flagged_by_channel, outlier_stats = flag_size_outlier_cells(channel_objects, thr)
        flagged = set().union(*flagged_by_channel.values()) if flagged_by_channel else set()
        outlier_by_channel = {ch: len(ids) for ch, ids in flagged_by_channel.items()}
        outlier_removed_df = particles_df[~particles_df['cell_id'].isin(flagged)].reset_index(drop=True)
        sheets['Outlier_removed'] = append_summary_statistics(
            outlier_removed_df, 'cell_id', include_sem=False)
        sheets['Outlier_removed_sample_level'] = _sample_level(outlier_removed_df)

    save_feature_summary(analysis_path, sheets)
    _write_qc_sidecar(analysis_path, image_folder.name, config, channels,
                      particles_df, particles_df, None, stain_stems,
                      {'mode': 'particle', 'stains': seg_summary},
                      outlier_removed_df=outlier_removed_df,
                      outlier_by_channel=outlier_by_channel, outlier_stats=outlier_stats)
    n_out = '' if outlier_removed_df is None else f", {len(outlier_removed_df)} outlier-removed"
    print(f"\nParticle summary: {len(particles_df)} particles across "
          f"{len(stain_stems)} stain(s){n_out}")
    print(f"  -> {analysis_path / 'Feature_summary.xlsx'}")


# =============================================================================
# INPUT DISCOVERY  (sample-organization strategies -- see IO_PLAN.md)
# =============================================================================

@dataclass
class SampleSpec:
    """One unit of work: a named sample, the dir its Analysis/ output goes under,
    and a loader yielding (channels, sample_meta). Decouples *what is a sample*
    (a folder vs a single multi-channel file) from *how it is processed*."""
    name: str
    sample_dir: Path
    load: Callable          # () -> (channels: dict, sample_meta: dict)


def _folder_samples(root, roles):
    """Folder-per-sample: each folder offering the nuclear channel (files or C/T
    splits) is one sample. This is the historical default."""
    return [SampleSpec(name=f.name, sample_dir=f,
                       load=(lambda ff=f: load_channels_with_meta(ff)))
            for f in find_processable_folders(root, roles)]


def _file_samples(root, roles):
    """File-per-sample: each multi-channel TIFF offering the primary channel (by
    bare c0/c1 name) is one sample. Output -> <file_parent>/<filestem>/Analysis/,
    so the existing hierarchical aggregation + QC report work unchanged. Primary =
    nuclear when assigned, else the first stain (particle mode)."""
    primary_stem = get_primary_stem(roles)
    specs = []
    if primary_stem is None:
        return specs
    for dirpath, _, _ in os.walk(root):
        if Path(dirpath).name == 'Analysis':
            continue
        for f in list_tiffs(dirpath):
            if primary_stem in file_channel_names(f):
                specs.append(SampleSpec(name=f.stem, sample_dir=f.parent / f.stem,
                                        load=(lambda ff=f: load_file_as_sample(ff))))
    return specs


def _detect_input_mode(root, roles):
    """Prefer folder-per-sample (back-compatible); fall back to file-per-sample
    only when no folder offers the primary channel but some single file does."""
    if find_processable_folders(root, roles):
        return 'folder_per_sample'
    if _file_samples(root, roles):
        return 'file_per_sample'
    return 'folder_per_sample'


def iter_samples(root, config):
    """Resolve the input organization into (mode, [SampleSpec, ...])."""
    root = Path(root)
    roles = config.get('channel_roles', {})
    mode = config.get('input_mode', 'auto')
    if mode == 'auto':
        mode = _detect_input_mode(root, roles)
    specs = (_file_samples(root, roles) if mode == 'file_per_sample'
             else _folder_samples(root, roles))
    return mode, specs


# =============================================================================
# ORCHESTRATION: COMBINED
# =============================================================================

def process_sample(spec, config, cellpose_model=None):
    image_folder = Path(spec.sample_dir)          # dir that will hold Analysis/
    image_folder.mkdir(parents=True, exist_ok=True)
    channel_roles = config['channel_roles']
    # Apply the CPU thread policy here too, so a direct call (e.g. Save Current Image)
    # gets the same parallelism as a full batch run.
    n_workers = set_worker_count(config.get('num_threads'))

    print("\n" + "=" * 80)
    print(f"PROCESSING: {spec.name}")
    print("=" * 80)

    channels, sample_meta = spec.load()

    nuclear_stem = get_role_stem(channel_roles, 'nuclear')
    primary_stem = get_primary_stem(channel_roles)
    if primary_stem is None or primary_stem not in channels:
        print("Error: no nuclear or stain channel found. Skipping.")
        return

    # Resolve per-sample voxel calibration (xy stays user-set; z + anisotropy from
    # the TIFF metadata) and inject it into a per-sample config copy. Drives µm³
    # volumes, anisotropy-aware Cellpose/watershed, and calibrated mask output.
    px, pz, aniso = resolve_calibration(config, sample_meta)
    # Provenance must read the ORIGINAL (user) config, before injection overwrites
    # pixel_size_um with the resolved value. Warn loudly when nothing is calibrated
    # so a silent 1 µm/px fallback can't quietly mislabel every µm² / µm³ value.
    cal = describe_calibration(config, sample_meta)
    config = _apply_sample_calibration(config, px, pz, aniso)
    config['_calibration'] = cal
    config['_num_workers'] = n_workers
    if not cal['calibrated']:
        print(f"  WARNING: no pixel size found for '{Path(image_folder).name}' "
              f"(config unset, no usable file metadata) -> using 1.0 um/px. All "
              f"area/volume/length values are UNCALIBRATED. Set the pixel size in the "
              f"UI or embed it in the image metadata.")
    save_meta = {'is_3d': sample_meta['is_3d'], 'pixel_size_um': px,
                 'z_size_um': pz, 'unit': sample_meta.get('unit')}
    if sample_meta['is_3d']:
        print(f"3D stack: voxel {px:.4g} x {px:.4g} x {pz:.4g} um "
              f"(anisotropy {aniso:.3g})")

    # No active nuclear segmentation (no nuclear role, or its method is 'none') ->
    # particle mode: analyse the stains as standalone particles (no nuclei / cytoplasm).
    nuclear_active = (nuclear_stem is not None
                      and config.get('nuclear_method', 'threshold') != 'none')
    if not nuclear_active:
        run_particle_analysis(image_folder, channels, config, meta=save_meta)
        return

    use_preseg_nuc = config.get('use_presegmented_nuclear', False)
    use_preseg_cyto = config.get('use_presegmented_cytoplasm', False)

    if use_preseg_nuc or use_preseg_cyto:
        preseg_nuc, preseg_cyto = load_presegmented_masks(image_folder, config)
    else:
        preseg_nuc, preseg_cyto = None, None

    # Determine nuclear labels
    seg_meta = None
    segmented_cyto = None
    if use_preseg_nuc and preseg_nuc is not None:
        nuclear_labels = preseg_nuc
        print(f"Using pre-segmented nuclear labels "
              f"({len(np.unique(nuclear_labels[nuclear_labels > 0]))} objects)")
    else:
        nuclear_labels, segmented_cyto, seg_meta = run_segmentation(
            image_folder, channels, config, cellpose_model, meta=save_meta)
        if nuclear_labels is None:
            return
        # run_segmentation handled cytoplasm too, UNLESS the cytoplasm method is
        # pre-segmented -- in that case keep the mask loaded above so it survives.
        if not use_preseg_cyto:
            preseg_cyto = None

    # Determine cytoplasm labels
    if use_preseg_cyto and preseg_cyto is not None:
        cytoplasm_labels = preseg_cyto
        print(f"Using pre-segmented cytoplasm labels "
              f"({len(np.unique(cytoplasm_labels[cytoplasm_labels > 0]))} objects)")
    elif not use_preseg_nuc:
        # Use the array run_segmentation just produced. Re-reading Analysis/
        # cytoplasm_labeled.tif here would resurrect a stale mask from an earlier
        # run whenever this run segmented no cytoplasm.
        cytoplasm_labels = segmented_cyto
    else:
        # Pre-segmented nuclear but need to segment cytoplasm
        cytoplasm_labels = None
        cytoplasm_source = config.get('cytoplasm_source', 'none')
        if cytoplasm_source != 'none' and config.get('cytoplasm_method', 'cellpose') != 'none':
            if cytoplasm_source == 'membrane':
                stem = get_role_stem(channel_roles, 'membrane')
            else:
                stem = get_role_stem(channel_roles, 'cytoplasm')
            if stem and stem in channels:
                if config.get('cytoplasm_method', 'cellpose') == 'threshold':
                    cytoplasm_labels, info = segment_cytoplasm_threshold(
                        channels[stem], nuclear_labels, config, source=cytoplasm_source,
                    )
                else:
                    cytoplasm_labels, info = segment_cytoplasm_cellpose(
                        channels[stem], nuclear_labels, config, model=cellpose_model,
                    )
                analysis_path = image_folder / "Analysis"
                analysis_path.mkdir(exist_ok=True)
                save_masks(analysis_path, 'cytoplasm', cytoplasm_labels, meta=save_meta)
                print(f"Cytoplasm: {info['n_cytoplasm']} regions")

    run_analysis(image_folder, channels, nuclear_labels, cytoplasm_labels, config,
                 seg_meta=seg_meta, meta=save_meta)


def process_single_image_set(image_folder, config, cellpose_model=None):
    """Back-compat: process a single folder as one sample (folder_per_sample)."""
    folder = Path(image_folder)
    spec = SampleSpec(name=folder.name, sample_dir=folder,
                      load=lambda: load_channels_with_meta(folder))
    return process_sample(spec, config, cellpose_model)


def run_batch(input_folder, config=None):
    if config is None:
        config = DEFAULT_CONFIG.copy()
    input_path = Path(input_folder)
    mode, specs = iter_samples(input_path, config)
    if not specs:
        print(f"Error: No processable samples found in {input_path}")
        return
    set_worker_count(config.get('num_threads'))
    print(f"Found {len(specs)} sample(s)  [input mode: {mode}]")
    print(thread_report())

    # Pre-warm the cytoplasm Cellpose model once for the batch (only when cytoplasm
    # actually uses Cellpose; threshold cytoplasm needs no model).
    cellpose_model = None
    cyto_source = config.get('cytoplasm_source', 'none')
    if (cyto_source != 'none'
            and config.get('cytoplasm_method', 'cellpose') == 'cellpose'
            and not config.get('use_presegmented_cytoplasm', False)):
        cellpose_model = get_cellpose_model(config)

    for spec in specs:
        process_sample(spec, config, cellpose_model)

    print("\n" + "=" * 80)
    print("ALL IMAGE SETS PROCESSED")
    print("=" * 80)
    generate_all_hierarchical_summaries(input_path)
    print("\nALL HIERARCHICAL SUMMARIES GENERATED")

    # Auto-generate the batch QC report. qc_report is imported lazily (so matplotlib
    # stays an optional dependency of the core) and any failure here is non-fatal:
    # the analysis outputs are already saved, so a missing matplotlib or a plotting
    # error must never invalidate the run.
    print("\n" + "=" * 80)
    print("GENERATING BATCH QC REPORT")
    print("=" * 80)
    try:
        import qc_report
        qc_report.generate(input_path)
    except Exception as exc:
        print(f"Warning: QC report not generated ({exc}).")
        print(f"  You can still create it manually: python qc_report.py \"{input_path}\"")