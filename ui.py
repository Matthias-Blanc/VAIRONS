import sys
import os
import colorsys
import numpy as np

# torch MUST be imported before PyQt6 (and before Cellpose pulls it in via the
# `from processing import ...` below). On Windows, initializing Qt's native
# libraries first can clash with torch's OpenMP/MKL DLLs and crash the app, so
# torch is imported here for its initialization side effect only. Do not remove.
import torch

from dataclasses import dataclass, field
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QFileDialog, QScrollArea, QButtonGroup,
    QSplitter, QMessageBox, QGraphicsView, QGraphicsScene,
    QStatusBar, QFormLayout, QFrame, QComboBox, QSlider, QTabWidget,
    QGridLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QPalette, QBrush, QIcon

from processing import (
    segment_nuclei,
    extract_all_nuclear_features,
    segment_cytoplasm_cellpose,
    segment_cytoplasm_threshold,
    segment_stain,
    extract_all_cytoplasm_features,
    quantify_stain_per_labels,
    mean_particle_area_px,
    compute_stain_spatial_analysis,
    compute_nuclear_alignment,
    map_nuclei_to_cytoplasm,
    detect_device,
    set_worker_count,
    resolve_worker_count,
    is_model_cached,
    get_cellpose_model_by_name,
)
from io_manager import (
    DEFAULT_CONFIG, DEFAULT_STAIN_CONFIG, load_channels_with_meta, discover_channels,
    run_batch, get_role_stem, resolve_calibration, _apply_sample_calibration,
    load_file_as_sample, build_merged_cells_dataframe, build_sample_level_dataframe,
    build_qc_metrics, active_stain_stems, SampleSpec, supported_image_extensions,
)


# =============================================================================
# BRANDING
# =============================================================================

LOGO_PATH = Path(__file__).resolve().parent / "LOGO.png"


def app_icon():
    """The Vairons mark for the window / taskbar, or an empty icon if the file is
    missing (never fatal). The logo is deliberately not drawn inside the UI itself."""
    return QIcon(str(LOGO_PATH)) if LOGO_PATH.exists() else QIcon()


# =============================================================================
# DARK PALETTE
# =============================================================================

def create_dark_palette():
    p = QPalette()
    CR = QPalette.ColorRole
    CG = QPalette.ColorGroup
    p.setColor(CR.Window, QColor(30, 30, 30))
    p.setColor(CR.WindowText, QColor(224, 224, 224))
    p.setColor(CR.Base, QColor(45, 45, 45))
    p.setColor(CR.AlternateBase, QColor(35, 35, 35))
    p.setColor(CR.ToolTipBase, QColor(224, 224, 224))
    p.setColor(CR.ToolTipText, QColor(30, 30, 30))
    p.setColor(CR.Text, QColor(224, 224, 224))
    p.setColor(CR.Button, QColor(45, 45, 45))
    p.setColor(CR.ButtonText, QColor(224, 224, 224))
    p.setColor(CR.BrightText, QColor(255, 255, 255))
    p.setColor(CR.Link, QColor(100, 149, 237))
    p.setColor(CR.Highlight, QColor(70, 130, 180))
    p.setColor(CR.HighlightedText, QColor(255, 255, 255))
    p.setColor(CG.Disabled, CR.WindowText, QColor(128, 128, 128))
    p.setColor(CG.Disabled, CR.Text, QColor(128, 128, 128))
    p.setColor(CG.Disabled, CR.ButtonText, QColor(128, 128, 128))
    return p


DARK_STYLESHEET = """
QGroupBox {
    border: 1px solid #3d3d3d; border-radius: 4px;
    margin-top: 8px; padding-top: 8px; font-weight: bold;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #2d2d2d; border: 1px solid #3d3d3d;
    border-radius: 3px; padding: 3px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border: 1px solid #4682b4;
}
QPushButton {
    background-color: #3d3d3d; border: 1px solid #4d4d4d;
    border-radius: 4px; padding: 6px 12px;
}
QPushButton:hover { background-color: #4d4d4d; }
QPushButton:pressed { background-color: #2d2d2d; }
QPushButton:disabled { background-color: #2a2a2a; color: #666666; }
QPushButton:checked { background-color: #4682b4; border: 1px solid #5a9fd4; }
QScrollArea { border: none; }
QScrollBar:vertical {
    background-color: #2d2d2d; width: 12px; border-radius: 6px;
}
QScrollBar::handle:vertical {
    background-color: #4d4d4d; border-radius: 6px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background-color: #5d5d5d; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QStatusBar { background-color: #252525; border-top: 1px solid #3d3d3d; }
QSplitter::handle { background-color: #3d3d3d; }
QSlider::groove:horizontal { background: #3d3d3d; height: 6px; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #4682b4; width: 14px; margin: -4px 0; border-radius: 7px;
}
QTabWidget::pane { border: 1px solid #3d3d3d; border-radius: 4px; top: -1px; }
QTabBar::tab {
    background: #2d2d2d; border: 1px solid #3d3d3d; border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
    padding: 4px 10px; margin-right: 2px;
}
QTabBar::tab:selected { background: #4682b4; }
QTabBar::tab:hover:!selected { background: #3d3d3d; }
"""


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_CHANNEL_COLORS = [
    (200, 200, 200), (0, 255, 0), (255, 0, 0), (0, 100, 255),
    (0, 255, 255), (255, 0, 255), (255, 255, 0),
]

MASK_COLORS = {
    'nuclear_labels': (255, 255, 0),
    'cytoplasm_labels': (0, 255, 200),
}


def channel_color(index):
    """Distinct display color for a channel; avoids collisions beyond 7 channels.

    The first entries match DEFAULT_CHANNEL_COLORS; higher indices are generated
    with golden-ratio hue spacing so any number of channels stays distinguishable.
    """
    if index < len(DEFAULT_CHANNEL_COLORS):
        return DEFAULT_CHANNEL_COLORS[index]
    h = (0.61803398875 * index) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))


ROLE_OPTIONS = ['none', 'nuclear', 'membrane', 'cytoplasm', 'stain']

CYTOPLASM_SOURCE_OPTIONS = [
    ('None', 'none'),
    ('Membrane channel', 'membrane'),
    ('Cytoplasm channel', 'channel'),
]

# Input organization (see IO_PLAN.md): folder-as-sample vs file-as-sample vs auto.
INPUT_MODE_OPTIONS = [
    ('Auto', 'auto'),
    ('Folder per sample', 'folder_per_sample'),
    ('File per sample', 'file_per_sample'),
]

CELLPOSE_MODEL_OPTIONS = ['cpsam', 'cpsam_v2', 'cpdino', 'cpdino-vitb']

# Segmentation method is independent of channel role: any detection (nuclei,
# stains/particles) can use either, each with its own parameters.
# 'none' disables that detection (no segmentation). For a stain it stays a measured
# channel; for nuclei it drops the sample to particle mode; for cytoplasm it is off.
SEGMENTATION_METHODS = ['threshold', 'cellpose', 'none']
# Nuclear & cytoplasm additionally accept pre-segmented label images as a method
# (loaded from the sample folder / its Analysis/). Stains have no pre-seg path.
SEGMENTATION_METHODS_PRESEG = ['threshold', 'cellpose', 'presegmented', 'none']

STAIN_OUTPUT_OPTIONS = ['binary', 'labeled']

# Hover help for the Cellpose parameters (shared by the nuclear/stain block and the
# cytoplasm block). Phrased as "what it does + which way to turn it".
CELLPOSE_TIPS = {
    'model': ("Cellpose model. 'cpsam' is the Cellpose-SAM generalist and is the "
              "recommended default; a path to a custom-trained model also works."),
    'diameter': ("Expected object diameter in pixels. 0 = no rescaling — the "
                 "recommended setting for cpsam, which is scale-invariant. (Cellpose "
                 "4.x has no size model, so 0 does not mean 'estimate'.) Set a value "
                 "only to rescale the image by 30/diameter before segmentation."),
    'cellprob': ("Cell-probability decision threshold (−6…6, default 0). Lower it to "
                 "recover dim / missed objects (more, larger masks); raise it to reject "
                 "background (fewer, tighter masks)."),
    'flow': ("Flow-error threshold (0…3, default 0.4). Masks whose recomputed flows "
             "disagree by more than this are discarded — lower = stricter (drops odd "
             "shapes), higher = more permissive. Applied in 2D only; ignored for 3D."),
    'niter': ("Dynamics iterations used to reconstruct masks. 0 = auto (scales with "
              "diameter). Increase for very large or long/thin objects that need more "
              "steps to fill."),
    'min_size': ("Objects smaller than this many pixels (voxels in 3D) are removed "
                 "after Cellpose segmentation."),
    'max_size_fraction': ("Cellpose discards any object covering more than this "
                          "fraction of the image (default 0.4 = 40%). Raise it towards "
                          "1.0 for cropped fields or sparse, very large cells, which "
                          "the default silently deletes."),
    'invert': ("Invert intensities before segmentation — use for dark objects on a "
               "bright background."),
}

FEATURE_CATEGORIES = ['Nuclear', 'Cytoplasm', 'Stain']


# =============================================================================
# GLASBEY LUT
# =============================================================================

def generate_glasbey_lut(n_colors=256):
    np.random.seed(42)
    lut = np.zeros((n_colors, 3), dtype=np.uint8)
    base = [
        [255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],
        [0,255,255],[255,128,0],[255,0,128],[128,255,0],[0,255,128],
        [128,0,255],[0,128,255],[255,128,128],[128,255,128],[128,128,255],
        [255,255,128],[255,128,255],[128,255,255],[192,64,64],[64,192,64],
        [64,64,192],[192,192,64],[192,64,192],[64,192,192],[255,192,64],
        [255,64,192],[192,255,64],[64,255,192],[192,64,255],[64,192,255],
    ]
    for i, c in enumerate(base):
        if i + 1 < n_colors:
            lut[i + 1] = c
    for i in range(len(base) + 1, n_colors):
        lut[i] = [np.random.randint(50, 256) for _ in range(3)]
    return lut


GLASBEY_LUT = generate_glasbey_lut(256)


def apply_glasbey_lut(labeled_image):
    mod = (labeled_image % 255) + 1
    mod[labeled_image == 0] = 0
    return GLASBEY_LUT[mod.astype(np.uint8)]


def normalize_to_uint8(image):
    mn, mx = image.min(), image.max()
    if mx == mn:
        return np.zeros_like(image, dtype=np.uint8)
    return ((image - mn) / (mx - mn) * 255).astype(np.uint8)


def array_to_qpixmap(array):
    if array.ndim == 2:
        h, w = array.shape
        qimg = QImage(array.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        h, w, c = array.shape
        qimg = QImage(array.data, w, h, w * c, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()
        if w:
            w.deleteLater()
        elif item.layout():
            clear_layout(item.layout())


# =============================================================================
# CUSTOM GRAPHICS VIEW
# =============================================================================

class ImageGraphicsView(QGraphicsView):
    pixel_hovered = pyqtSignal(int, int, int, int)
    cell_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QBrush(QColor(26, 26, 26)))
        self.setMouseTracking(True)
        self.pixmap_item = None
        self.highlight_item = None
        self.raw_image = None
        self.labels_image = None
        self.zoom_factor = 1.0
        self.panning = False
        self.pan_start = QPointF()

    def set_images(self, raw_image, labels_image):
        self.raw_image = raw_image
        self.labels_image = labels_image

    def set_pixmap(self, pixmap):
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.highlight_item = None
        self.setSceneRect(self.pixmap_item.boundingRect())

    def highlight_label(self, label_id):
        if self.highlight_item is not None:
            self.scene().removeItem(self.highlight_item)
            self.highlight_item = None
        if label_id <= 0 or self.labels_image is None:
            return
        mask = self.labels_image == label_id
        if not np.any(mask):
            return
        from skimage import measure as _measure
        from PyQt6.QtGui import QPainterPath
        from PyQt6.QtWidgets import QGraphicsPathItem
        contours = _measure.find_contours(mask.astype(float), 0.5)
        path = QPainterPath()
        for contour in contours:
            from PyQt6.QtCore import QPointF as _QP
            from PyQt6.QtGui import QPolygonF
            polygon = QPolygonF([_QP(p[1], p[0]) for p in contour])
            path.addPolygon(polygon)
            path.closeSubpath()
        self.highlight_item = QGraphicsPathItem(path)
        pen = QPen(QColor(255, 255, 0), 2)
        pen.setCosmetic(True)
        self.highlight_item.setPen(pen)
        self.highlight_item.setBrush(QBrush())
        self.scene().addItem(self.highlight_item)

    def fit_in_view(self):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self.zoom_factor = self.transform().m11()

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        self.zoom_factor *= factor

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            sp = self.mapToScene(event.position().toPoint())
            x, y = int(sp.x()), int(sp.y())
            if self.labels_image is not None:
                h, w = self.labels_image.shape
                if 0 <= x < w and 0 <= y < h:
                    self.cell_clicked.emit(int(self.labels_image[y, x]))
                    self.highlight_label(int(self.labels_image[y, x]))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = event.position() - self.pan_start
            self.pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
        else:
            sp = self.mapToScene(event.position().toPoint())
            x, y = int(sp.x()), int(sp.y())
            intensity, label_id = 0, 0
            if self.raw_image is not None:
                h, w = self.raw_image.shape[:2]
                if 0 <= x < w and 0 <= y < h:
                    intensity = int(self.raw_image[y, x])
            if self.labels_image is not None:
                h, w = self.labels_image.shape
                if 0 <= x < w and 0 <= y < h:
                    label_id = int(self.labels_image[y, x])
            self.pixel_hovered.emit(x, y, intensity, label_id)
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.fit_in_view()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)


# =============================================================================
# BATCH WORKER
# =============================================================================

# =============================================================================
# APPLY-PREVIEW COMPUTE  (pure: no widgets, safe to call from a worker thread)
# =============================================================================

@dataclass
class ApplyResult:
    """Everything one Apply produces. The worker fills it; the GUI thread renders it."""
    particle_mode: bool = False
    nuclear_labels: object = None
    nuclear_features: object = None
    cytoplasm_labels: object = None
    cytoplasm_features: object = None
    stain_results: dict = field(default_factory=dict)
    stain_per_nucleus: dict = field(default_factory=dict)
    stain_per_cytoplasm: dict = field(default_factory=dict)
    stain_spatial_nucleus: dict = field(default_factory=dict)
    info_parts: list = field(default_factory=list)
    sample_rows: list = field(default_factory=list)


class ApplyError(Exception):
    """A preview failure worth showing the user verbatim (e.g. missing pre-seg file)."""


def _cellpose_params_in_use(config):
    """(model_type, gpu) for every detection whose method is Cellpose."""
    pairs = []
    if config.get('nuclear_method', 'threshold') == 'cellpose':
        nc = config.get('nuclear_cellpose', {})
        pairs.append((nc.get('model_type', 'cpsam'), nc.get('gpu', True)))
    if (config.get('cytoplasm_source', 'none') != 'none'
            and config.get('cytoplasm_method', 'cellpose') == 'cellpose'):
        pairs.append((config.get('cellpose_model_type', 'cpsam'),
                      config.get('cellpose_gpu', True)))
    for sc in (config.get('stain_configs') or {}).values():
        if sc.get('method', 'threshold') == 'cellpose':
            cp = sc.get('cellpose', {})
            pairs.append((cp.get('model_type', 'cpsam'), cp.get('gpu', True)))
    return pairs


def _warm_cellpose_models(config, progress):
    """Load any Cellpose model this run needs, announcing it first.

    The first load pulls ~1.2 GB of weights and initialises the CUDA context -- several
    seconds during which the user otherwise sees no explanation.
    """
    for model_type, gpu in _cellpose_params_in_use(config):
        if not is_model_cached(model_type, gpu):
            progress(f"Loading Cellpose model '{model_type}' (first run, this takes a moment)...")
            get_cellpose_model_by_name(model_type, gpu)


def compute_cell_preview(channels, config, test_image_path, progress=lambda _m: None):
    """Nuclei -> features -> cytoplasm -> stains, for the standard (nuclear) preview."""
    channel_roles = config['channel_roles']
    nuclear_stem = get_role_stem(channel_roles, 'nuclear')
    detection_image = channels[nuclear_stem]
    res = ApplyResult(particle_mode=False)

    # --- Nuclear ---
    if config.get('use_presegmented_nuclear') and config.get('presegmented_nuclear_stem'):
        from io_manager import load_presegmented_masks
        progress("Loading pre-segmented nuclei...")
        preseg_nuc, _ = load_presegmented_masks(test_image_path, config)
        if preseg_nuc is None:
            raise ApplyError("Pre-segmented nuclear file not found")
        res.nuclear_labels = preseg_nuc
        res.info_parts.append(f"Nuclear: {len(np.unique(preseg_nuc[preseg_nuc > 0]))} (pre-seg)")
    else:
        progress("Segmenting nuclei...")
        labeled, _, seg_info = segment_nuclei(detection_image, config)
        res.nuclear_labels = labeled
        res.info_parts.append(f"Nuclear: {seg_info['final_count']}")

    progress("Nuclear features...")
    res.nuclear_features = extract_all_nuclear_features(
        res.nuclear_labels, detection_image, channels, config)

    # --- Cytoplasm ---
    if config.get('use_presegmented_cytoplasm'):
        preseg_cyto = None
        if config.get('presegmented_cytoplasm_stem'):
            from io_manager import load_presegmented_masks
            progress("Loading pre-segmented cytoplasm...")
            _, preseg_cyto = load_presegmented_masks(test_image_path, config)
        if preseg_cyto is not None:
            res.cytoplasm_labels = preseg_cyto
            res.info_parts.append(f"Cyto: {len(np.unique(preseg_cyto[preseg_cyto > 0]))} (pre-seg)")
        else:
            res.info_parts.append("Cyto: preseg not found")
    else:
        cyto_source = config['cytoplasm_source']
        if cyto_source != 'none' and config.get('cytoplasm_method', 'cellpose') != 'none':
            stem = get_role_stem(channel_roles,
                                 'membrane' if cyto_source == 'membrane' else 'cytoplasm')
            if stem and stem in channels:
                progress("Segmenting cytoplasm...")
                if config.get('cytoplasm_method', 'cellpose') == 'threshold':
                    res.cytoplasm_labels, ci = segment_cytoplasm_threshold(
                        channels[stem], res.nuclear_labels, config, source=cyto_source)
                else:
                    res.cytoplasm_labels, ci = segment_cytoplasm_cellpose(
                        channels[stem], res.nuclear_labels, config)
                res.info_parts.append(f"Cyto: {ci['n_cytoplasm']}")
            else:
                res.info_parts.append("Cyto: no channel")

    if res.cytoplasm_labels is not None:
        cyto_source = config['cytoplasm_source']
        cs = (get_role_stem(channel_roles, 'membrane') if cyto_source == 'membrane'
              else get_role_stem(channel_roles, 'cytoplasm') if cyto_source == 'channel'
              else None)
        cyto_int = channels.get(cs, detection_image) if cs else detection_image
        progress("Cytoplasm features...")
        res.cytoplasm_features = extract_all_cytoplasm_features(
            res.cytoplasm_labels, cyto_int, channels, config,
            compute_texture=(cyto_source != 'membrane'))

    # --- Stains ---
    stain_configs = config.get('stain_configs', {})
    for stem in [s for s, r in channel_roles.items() if r == 'stain']:
        if stem not in channels:
            continue
        sc = stain_configs.get(stem, DEFAULT_STAIN_CONFIG.copy())
        if sc.get('method', 'threshold') == 'none':
            continue                              # measured channel, not segmented

        progress(f"Stain: {stem}...")
        stain_mask, si = segment_stain(
            channels[stem], sc, edge_margin=config.get('edge_exclusion_margin', 0))
        is_binary = si.get('output_type', 'binary') == 'binary'
        res.stain_results[stem] = (stain_mask, is_binary)

        sb = stain_mask.astype(bool) if is_binary else (stain_mask > 0)
        per_nuc = quantify_stain_per_labels(
            sb, res.nuclear_labels, config['pixel_size_um'], z_size_um=config.get('z_size_um'))
        res.stain_per_nucleus[stem] = {d['label_id']: d for d in per_nuc}

        if res.cytoplasm_labels is not None:
            per_cyto = quantify_stain_per_labels(
                sb, res.cytoplasm_labels, config['pixel_size_um'],
                z_size_um=config.get('z_size_um'))
            res.stain_per_cytoplasm[stem] = {d['label_id']: d for d in per_cyto}

        spatial = compute_stain_spatial_analysis(sb, res.nuclear_labels, config['pixel_size_um'])
        res.stain_spatial_nucleus[stem] = {d['label_id']: d for d in spatial}

        count_key = 'object_count' if is_binary else 'final_count'
        res.info_parts.append(f"{stem}: {si.get(count_key, '?')} obj")

    return res


def compute_particle_preview(channels, config, progress=lambda _m: None):
    """Preview when there is no nuclear channel: segment each stain into instances.

    The FIRST stain drives click-to-feature (it is placed in `nuclear_labels` /
    `nuclear_features` so the existing feature panel works unchanged); the remaining
    stains are shown as mask layers. The batch run (`run_particle_analysis`) reports
    every stain's particles.
    """
    from skimage import measure as _measure
    channel_roles = config['channel_roles']
    stain_configs = config.get('stain_configs', {})
    stain_stems = [s for s, r in channel_roles.items()
                   if r == 'stain' and s in channels
                   and stain_configs.get(s, {}).get('method', 'threshold') != 'none']
    res = ApplyResult(particle_mode=True)
    primary = stain_stems[0] if stain_stems else None

    for stem in stain_stems:
        sc = stain_configs.get(stem, DEFAULT_STAIN_CONFIG.copy())
        progress(f"Stain: {stem}...")
        mask, info = segment_stain(
            channels[stem], sc, edge_margin=config.get('edge_exclusion_margin', 0))
        is_binary = info.get('output_type', 'binary') == 'binary'
        if stem == primary:
            labels = (_measure.label(mask, connectivity=2) if is_binary
                      else mask)    # already an instance-label image
            res.nuclear_labels = labels           # click target (features)
            progress(f"Particle features: {stem}...")
            res.nuclear_features = extract_all_nuclear_features(
                labels, channels[stem], channels, config)
            # Show the primary as a labelled mask under its own stain row.
            res.stain_results[stem] = (labels, False)
            res.info_parts.append(f"{stem}: {len(res.nuclear_features)} particles (clickable)")
        else:
            res.stain_results[stem] = (mask, is_binary)
            count_key = 'object_count' if is_binary else 'final_count'
            res.info_parts.append(f"{stem}: {info.get(count_key, '?')} obj")

    return res


def cell_sample_level_rows(res, channels, config):
    """Sample-level metrics for the nuclear preview, via the same io_manager builders
    the batch output uses. Pure: reads `res` + `channels`, never the widget tree."""
    if res.nuclear_labels is None or not res.nuclear_features:
        return []
    channel_roles = config['channel_roles']
    stain_configs = config.get('stain_configs', {})
    channel_stems = sorted(channels.keys())
    stain_stems = active_stain_stems(channel_roles, stain_configs)
    nuclear_stem = get_role_stem(channel_roles, 'nuclear')
    cyto_source = config.get('cytoplasm_source', 'none')
    cyto_stem = (get_role_stem(channel_roles, 'membrane') if cyto_source == 'membrane'
                 else get_role_stem(channel_roles, 'cytoplasm') if cyto_source == 'channel'
                 else None)
    nuc_to_cyto = (map_nuclei_to_cytoplasm(res.nuclear_labels, res.cytoplasm_labels)
                   if res.cytoplasm_labels is not None else {})
    is_3d = res.nuclear_labels.ndim == 3
    # Per-image analysis-filter threshold = mean particle area (px) per stain,
    # from the cached stain masks (matches the batch path in run_analysis).
    stain_filter_areas = {}
    for s in stain_stems:
        entry = res.stain_results.get(s)
        if entry is not None:
            mask, is_bin = entry
            sb = mask.astype(bool) if is_bin else (mask > 0)
            stain_filter_areas[s] = mean_particle_area_px(sb)
    all_cells_df = build_merged_cells_dataframe(
        res.nuclear_features, res.cytoplasm_features, nuc_to_cyto,
        res.stain_per_nucleus, res.stain_per_cytoplasm, channel_stems, stain_stems,
        stain_filter_areas=stain_filter_areas, nuclear_stem=nuclear_stem,
        cyto_stem=(cyto_stem or nuclear_stem), is_3d=is_3d)
    _pxa = config['pixel_size_um']
    align_spacing = ((config.get('z_size_um') or _pxa), _pxa, _pxa) if is_3d else None
    alignment = compute_nuclear_alignment(res.nuclear_labels, spacing=align_spacing)
    sdf = build_sample_level_dataframe(
        all_cells_df, alignment, res.stain_spatial_nucleus, stain_stems, is_3d=is_3d)
    rows = [(r.metric, r.value) for r in sdf.itertuples(index=False)]
    rows += [(d['metric'], d['value']) for d in build_qc_metrics(channels, config)]
    return rows


def particle_sample_level_rows(res, channels, config, is_3d):
    """Sample-level metrics for particle mode. Pure (see cell_sample_level_rows)."""
    from skimage import measure as _measure
    px = config.get('pixel_size_um', 1.0) or 1.0
    z = config.get('z_size_um')
    factor = px * px if z is None else px * px * z
    align_spacing = ((z or px), px, px) if is_3d else None
    word, unit = ('volume', 'um3') if is_3d else ('area', 'um2')
    rows = []
    for stem, (mask, is_bin) in res.stain_results.items():
        labels = _measure.label(mask, connectivity=2) if is_bin else mask
        n = int(labels.max())
        stain_px = int(np.count_nonzero(labels > 0))
        img_px = int(labels.size)
        align = compute_nuclear_alignment(labels, spacing=align_spacing)
        rows += [
            (f'stain_{stem}_particle_count', n),
            (f'stain_{stem}_{word}_total_{unit}', stain_px * factor),
            (f'stain_{stem}_coverage_fraction_image',
             stain_px / img_px if img_px else float('nan')),
            (f'stain_{stem}_alignment', float(align) if align == align else float('nan')),
        ]
    rows += [(d['metric'], d['value']) for d in build_qc_metrics(channels, config)]
    return rows


# =============================================================================
# WORKERS
# =============================================================================

class BatchWorker(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, input_folder, config):
        super().__init__()
        self.input_folder = input_folder
        self.config = config

    def run(self):
        try:
            run_batch(self.input_folder, self.config)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


class SaveCurrentWorker(QThread):
    """Process + save the single loaded test image (masks + Feature_summary.xlsx +
    qc_meta), reusing the batch per-sample pipeline so the output matches Run Batch."""
    finished = pyqtSignal(str)     # emits the Analysis/ output dir
    error = pyqtSignal(str)

    def __init__(self, spec, config):
        super().__init__()
        self.spec = spec
        self.config = config

    def run(self):
        try:
            from io_manager import process_sample
            process_sample(self.spec, self.config)
            self.finished.emit(str(Path(self.spec.sample_dir) / "Analysis"))
        except Exception as e:
            self.error.emit(str(e))


class ApplyWorker(QThread):
    """Run the Apply-on-current-image preview off the GUI thread.

    Segmentation of a single field takes seconds (the first Cellpose call also loads
    ~1.2 GB of weights and initialises the CUDA context), which is long enough for a
    synchronous Apply to freeze the window. Everything here is pure computation on the
    already-loaded channel arrays; the GUI thread applies the result in _apply_done.
    """
    progress = pyqtSignal(str)
    done = pyqtSignal(object)      # ApplyResult
    failed = pyqtSignal(str)

    def __init__(self, channels, config, test_image_path, is_3d, particle_mode):
        super().__init__()
        self.channels = channels
        self.config = config
        self.test_image_path = test_image_path
        self.is_3d = is_3d
        self.particle_mode = particle_mode

    def run(self):
        try:
            set_worker_count(self.config.get('num_threads'))
            emit = self.progress.emit
            _warm_cellpose_models(self.config, emit)
            if self.particle_mode:
                res = compute_particle_preview(self.channels, self.config, emit)
            else:
                res = compute_cell_preview(self.channels, self.config,
                                           self.test_image_path, emit)
            emit("Sample-level metrics...")
            # Best-effort, as before: a sample-level failure shows an "error" row
            # rather than discarding a segmentation that already succeeded.
            try:
                res.sample_rows = (
                    particle_sample_level_rows(res, self.channels, self.config, self.is_3d)
                    if self.particle_mode
                    else cell_sample_level_rows(res, self.channels, self.config))
            except Exception as exc:
                res.sample_rows = [("error", str(exc))]
            self.done.emit(res)
        except Exception as e:
            self.failed.emit(str(e))


# =============================================================================
# MAIN UI
# =============================================================================

class VaironsUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vairons")
        self.setWindowIcon(app_icon())
        # Keep the minimum small enough to fit laptop screens (13" macOS ~1280x800,
        # many Windows laptops 1366x768); open larger where the display allows.
        self.setMinimumSize(1080, 680)
        self.resize(1560, 920)
        self.setAcceptDrops(True)                # drag a folder / image in to load it

        self.config = DEFAULT_CONFIG.copy()
        self.channels = {}
        self.normalized_channels = {}
        self.test_image_path = None

        # 3D state (resolved from the loaded TIFF metadata)
        self.is_3d = False
        self.n_z = 1
        self.current_z = 0
        self.meta_pixel_size_um = None
        self.meta_z_size_um = None

        self.nuclear_labels = None
        self.cytoplasm_labels = None
        self.nuclear_features = None
        self.cytoplasm_features = None
        self.selected_cell_id = None

        self.stain_results = {}
        self.stain_per_nucleus = {}
        self.stain_per_cytoplasm = {}
        self.stain_spatial_nucleus = {}

        self.channel_role_combos = {}
        self.channel_checkboxes = {}          # per-channel "Show" (layer visibility)
        self.channel_contrast_sliders = {}    # per-channel display contrast (gain)
        self.channel_colors = {}
        self.mask_row_checkboxes = {}         # per-channel "Mask" visibility
        self.mask_row_sliders = {}            # per-channel mask opacity
        self.mask_data = {}
        self.stain_param_groups = {}
        self.stain_param_widgets = {}

        self.active_feature_category = 'Nuclear'
        # True while previewing a nucleus-free (particle) image: the feature panel
        # then shows the primary stain's particle features under a "Particle" tab.
        self.particle_mode = False
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        splitter.addWidget(self.create_parameter_panel())
        splitter.addWidget(self.create_display_panel())
        splitter.addWidget(self.create_feature_panel())
        splitter.setSizes([380, 680, 340])

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.coord_label = QLabel("X: -- Y: --")
        self.coord_label.setMinimumWidth(120)
        self.intensity_label = QLabel("Intensity: --")
        self.intensity_label.setMinimumWidth(120)
        self.hover_label_label = QLabel("Label: --")
        self.hover_label_label.setMinimumWidth(100)
        self.status_bar.addWidget(self.coord_label)
        self.status_bar.addWidget(self.intensity_label)
        self.status_bar.addWidget(self.hover_label_label)
        # Dimensionality (2D/3D), CPU threads and Cellpose device, right-aligned.
        self.dim_label = QLabel("")
        self.dim_label.setStyleSheet("color: #888888;")
        _ncpu = os.cpu_count() or 1
        _nworkers = resolve_worker_count(None)
        self.threads_label = QLabel(f"CPU threads detected: {_ncpu} (using {_nworkers})")
        self.threads_label.setStyleSheet("color: #888888;")
        self.threads_label.setToolTip(
            "Worker threads for the parallel feature-extraction and stain-quantification "
            "steps: always the detected logical cores minus one, leaving one for the OS "
            "and this window. Cellpose is unaffected (it uses the GPU / its own threads).")
        _dev, _gpu = detect_device()
        self.device_label = QLabel(f"Cellpose device: {_dev} ({'GPU' if _gpu else 'CPU'})")
        self.device_label.setStyleSheet("color: #888888;")
        self.status_bar.addPermanentWidget(self.dim_label)
        self.status_bar.addPermanentWidget(self.threads_label)
        self.status_bar.addPermanentWidget(self.device_label)

    # =========================================================================
    # PARAMETER PANEL
    # =========================================================================

    def create_parameter_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(330)

        container = QWidget()
        layout = QVBoxLayout(container)

        # --- Setup (paths + pixel calibration) ---
        path_group = QGroupBox("Setup")
        pl = QVBoxLayout(path_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Input Folder:"))
        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setPlaceholderText("Select folder for batch...")
        row.addWidget(self.input_folder_edit)
        btn = QPushButton("Browse")
        btn.clicked.connect(self.browse_input_folder)
        row.addWidget(btn)
        pl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Input mode:"))
        self.input_mode_combo = QComboBox()
        for display, _ in INPUT_MODE_OPTIONS:
            self.input_mode_combo.addItem(display)
        self.input_mode_combo.setToolTip(
            "Folder per sample: each folder is a sample (files/channels inside).\n"
            "File per sample: each multi-channel image file is a sample (c0/c1 channels).\n"
            "Auto: detect per batch.")
        row.addWidget(self.input_mode_combo)
        pl.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Test Image:"))
        self.test_image_edit = QLineEdit()
        self.test_image_edit.setPlaceholderText("Browse or drag an image / channel folder in...")
        self.test_image_edit.setToolTip("Tip: drag a channel folder or a multi-channel image "
                                        "onto the window to load it.")
        row.addWidget(self.test_image_edit)
        self.test_image_browse_btn = QPushButton("Browse")
        self.test_image_browse_btn.clicked.connect(self.browse_test_image)
        row.addWidget(self.test_image_browse_btn)
        pl.addLayout(row)

        # --- Pixel calibration (folded into Setup) ---
        # Auto-filled from the image metadata on load (editable \u2014 override if the
        # metadata is missing or wrong).
        self.pixel_size_spin = self._dspin(pl, "Pixel Size (\u00b5m):",
                                           self.config['pixel_size_um'], 0.0001, 10.0, 6, 0.001)
        # Z-step (depth), shown only for 3D stacks; auto-filled from the TIFF metadata.
        self.z_size_row = QWidget()
        zrl = QHBoxLayout(self.z_size_row)
        zrl.setContentsMargins(0, 0, 0, 0)
        zrl.addWidget(QLabel("Z Size (\u00b5m):"))
        self.z_size_spin = QDoubleSpinBox()
        self.z_size_spin.setRange(0.0001, 100.0)
        self.z_size_spin.setDecimals(4)
        self.z_size_spin.setSingleStep(0.01)
        self.z_size_spin.setValue(self.config['pixel_size_um'])
        self.z_size_spin.valueChanged.connect(self._update_dim_label)
        zrl.addWidget(self.z_size_spin)
        pl.addWidget(self.z_size_row)
        self.z_size_row.setVisible(False)
        layout.addWidget(path_group)
        # dim_label (2D/3D), threads_label (CPU threads) and device_label (Cellpose
        # device) live in the status bar at the very bottom -- see init_ui. The worker
        # count is not a user setting: it is always detected cores minus one.

        # --- Segmentation tabs (per-detection settings, below Setup) ---
        # Nuclear / Cytoplasm / one tab per stain channel. Pre-segmented masks are
        # exposed here as a *method*, not a separate panel.
        self.seg_tabs = QTabWidget()
        self.seg_tabs.setMinimumHeight(360)
        self.seg_tabs.setVisible(False)   # shown by _sync_detection_tabs once a role exists

        # Nuclear tab
        nuc_page, nl = self._make_tab_page()
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Method:"))
        self.nuclear_method_combo = QComboBox()
        self.nuclear_method_combo.addItems(SEGMENTATION_METHODS_PRESEG)
        self.nuclear_method_combo.setCurrentText(self.config.get('nuclear_method', 'threshold'))
        mrow.addWidget(self.nuclear_method_combo)
        nl.addLayout(mrow)

        # Threshold-method controls
        self.nuclear_threshold_widget = QWidget()
        ntl = QVBoxLayout(self.nuclear_threshold_widget)
        ntl.setContentsMargins(0, 0, 0, 0)
        self.threshold_min_spin = self._spin(ntl, "Threshold Min:", self.config['threshold_min'], 0, 65535)
        self.threshold_max_spin = self._spin(ntl, "Threshold Max:", self.config['threshold_max'], 0, 65535)
        self.connectivity_spin = self._spin(ntl, "Connectivity:", self.config['connectivity'], 1, 2)
        self.doublet_threshold_spin = self._dspin(ntl, "Doublet Threshold:", self.config['doublet_threshold'], 0.0, 1.0, 2, 0.05)
        self.watershed_distance_spin = self._spin(ntl, "Watershed Min Distance (px):", self.config['watershed_min_distance'], 1, 500)
        nl.addWidget(self.nuclear_threshold_widget)

        # Cellpose-method controls (own params, independent of role)
        self.nuclear_cellpose_widget, self.nuclear_cellpose_widgets = self._build_cellpose_widget(
            self.config.get('nuclear_cellpose', {}))
        nl.addWidget(self.nuclear_cellpose_widget)

        # Pre-segmented-method controls (label-image stem)
        self.nuclear_preseg_widget = QWidget()
        npl = QVBoxLayout(self.nuclear_preseg_widget)
        npl.setContentsMargins(0, 0, 0, 0)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Pre-seg stem:"))
        self.preseg_nuclear_stem_edit = QLineEdit()
        self.preseg_nuclear_stem_edit.setPlaceholderText("e.g., nuclear_labels or cp_masks")
        prow.addWidget(self.preseg_nuclear_stem_edit)
        npl.addLayout(prow)
        nhint = QLabel("Loads a label TIFF from the sample folder or its Analysis/.")
        nhint.setStyleSheet("color: #888888;")
        nhint.setWordWrap(True)
        npl.addWidget(nhint)
        nl.addWidget(self.nuclear_preseg_widget)

        # 'none'-method hint
        self.nuclear_none_label = QLabel(
            "No nuclear segmentation. With stain channels present the sample is "
            "analysed in particle mode (one row per stain particle).")
        self.nuclear_none_label.setStyleSheet("color: #888888;")
        self.nuclear_none_label.setWordWrap(True)
        nl.addWidget(self.nuclear_none_label)

        # (Min Object Size / Edge Exclusion Margin used to live here. Min Object Size
        # was removed; the global Edge Exclusion Margin is in Pixel Settings and
        # applies to every detection.)

        def _toggle_nuclear_method(_=None):
            m = self.nuclear_method_combo.currentText()
            self.nuclear_threshold_widget.setVisible(m == 'threshold')
            self.nuclear_cellpose_widget.setVisible(m == 'cellpose')
            self.nuclear_preseg_widget.setVisible(m == 'presegmented')
            self.nuclear_none_label.setVisible(m == 'none')
        self.nuclear_method_combo.currentTextChanged.connect(_toggle_nuclear_method)
        _toggle_nuclear_method()
        nl.addStretch()
        # Tab pages are kept as attributes and added/removed by _sync_detection_tabs
        # so the Nuclear/Cytoplasm tabs appear only once the role is assigned.
        self.nuclear_tab_page = nuc_page

        # Cytoplasm tab
        cyto_page, cl = self._make_tab_page()
        row = QHBoxLayout()
        row.addWidget(QLabel("Source:"))
        self.cytoplasm_source_combo = QComboBox()
        for display, _ in CYTOPLASM_SOURCE_OPTIONS:
            self.cytoplasm_source_combo.addItem(display)
        self.cytoplasm_source_combo.currentIndexChanged.connect(self._update_cyto_visibility)
        row.addWidget(self.cytoplasm_source_combo)
        cl.addLayout(row)

        # Method (cellpose / threshold seeded-watershed / pre-segmented), shown when source != none
        self.cyto_method_widget = QWidget()
        cmrow = QHBoxLayout(self.cyto_method_widget)
        cmrow.setContentsMargins(0, 0, 0, 0)
        cmrow.addWidget(QLabel("Method:"))
        self.cytoplasm_method_combo = QComboBox()
        self.cytoplasm_method_combo.addItems(SEGMENTATION_METHODS_PRESEG)
        self.cytoplasm_method_combo.setCurrentText(self.config.get('cytoplasm_method', 'cellpose'))
        self.cytoplasm_method_combo.currentTextChanged.connect(self._update_cyto_visibility)
        cmrow.addWidget(self.cytoplasm_method_combo)
        cl.addWidget(self.cyto_method_widget)
        self.cyto_method_widget.setVisible(False)

        # Cellpose params (shown when source != none and method == cellpose)
        self.cellpose_widget = QWidget()
        cpl = QVBoxLayout(self.cellpose_widget)
        cpl.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        cmlab = QLabel("Model:")
        cmlab.setToolTip(CELLPOSE_TIPS['model'])
        row.addWidget(cmlab)
        self.cellpose_model_combo = QComboBox()
        self.cellpose_model_combo.addItems(CELLPOSE_MODEL_OPTIONS)
        if self.config['cellpose_model_type'] in CELLPOSE_MODEL_OPTIONS:
            self.cellpose_model_combo.setCurrentText(self.config['cellpose_model_type'])
        self.cellpose_model_combo.setToolTip(CELLPOSE_TIPS['model'])
        row.addWidget(self.cellpose_model_combo)
        cpl.addLayout(row)

        c = self.config
        self.cellpose_diameter_spin = self._spin(cpl, "Diameter (0=no rescale):", c['cellpose_diameter'], 0, 1000,
                                                 tip=CELLPOSE_TIPS['diameter'])
        self.cellpose_cellprob_spin = self._dspin(cpl, "Cellprob Threshold:", c['cellpose_cellprob_threshold'], -6.0, 6.0, 1, 0.5,
                                                  tip=CELLPOSE_TIPS['cellprob'])
        self.cellpose_flow_spin = self._dspin(cpl, "Flow Threshold:", c['cellpose_flow_threshold'], 0.0, 3.0, 2, 0.1,
                                              tip=CELLPOSE_TIPS['flow'])
        self.cellpose_niter_spin = self._spin(cpl, "Niter (0=auto):", c['cellpose_niter'], 0, 2000,
                                              tip=CELLPOSE_TIPS['niter'])
        self.cellpose_min_size_spin = self._spin(cpl, "Min Size (px):", c['cellpose_min_size'], 0, 100000,
                                                 tip=CELLPOSE_TIPS['min_size'])
        self.cellpose_max_size_frac_spin = self._dspin(
            cpl, "Max Size Fraction:", c.get('cellpose_max_size_fraction', 0.4), 0.05, 1.0, 2, 0.05,
            tip=CELLPOSE_TIPS['max_size_fraction'])
        # GPU is used automatically when available (no user toggle).
        self.cellpose_invert_cb = QCheckBox("Invert image")
        self.cellpose_invert_cb.setChecked(c['cellpose_invert'])
        self.cellpose_invert_cb.setToolTip(CELLPOSE_TIPS['invert'])
        cpl.addWidget(self.cellpose_invert_cb)

        cl.addWidget(self.cellpose_widget)
        self.cellpose_widget.setVisible(False)

        # Threshold params (shown when source != none and method == threshold)
        self.cyto_threshold_widget = QWidget()
        ctl = QVBoxLayout(self.cyto_threshold_widget)
        ctl.setContentsMargins(0, 0, 0, 0)
        ct = self.config.get('cytoplasm_threshold', {})
        self.cyto_thr_min_spin = self._spin(ctl, "Threshold Min:", ct.get('threshold_min', 18), 0, 65535)
        self.cyto_thr_max_spin = self._spin(ctl, "Threshold Max:", ct.get('threshold_max', 255), 0, 65535)
        self.cyto_min_size_spin = self._spin(ctl, "Min Object Size (px):", ct.get('min_object_size', 0), 0, 100000)
        cl.addWidget(self.cyto_threshold_widget)
        self.cyto_threshold_widget.setVisible(False)

        # Pre-segmented params (label-image stem); Source still sets intensity channel
        self.cyto_preseg_widget = QWidget()
        cptl = QVBoxLayout(self.cyto_preseg_widget)
        cptl.setContentsMargins(0, 0, 0, 0)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Pre-seg stem:"))
        self.preseg_cyto_stem_edit = QLineEdit()
        self.preseg_cyto_stem_edit.setPlaceholderText("e.g., cytoplasm_labels")
        prow.addWidget(self.preseg_cyto_stem_edit)
        cptl.addLayout(prow)
        chint = QLabel("Loads a cytoplasm label TIFF; Source still selects the measured intensity channel.")
        chint.setStyleSheet("color: #888888;")
        chint.setWordWrap(True)
        cptl.addWidget(chint)
        cl.addWidget(self.cyto_preseg_widget)
        self.cyto_preseg_widget.setVisible(False)

        cl.addStretch()
        self.cyto_tab_page = cyto_page
        layout.addWidget(self.seg_tabs)

        # Analysis Filter and Outlier Removal live in the right (feature) panel;
        # see _build_filter_group / _build_outlier_group.

        # --- Actions ---
        action_group = QGroupBox("Actions")
        al = QVBoxLayout(action_group)
        self.apply_btn = QPushButton("Apply on Current Image")
        self.apply_btn.clicked.connect(self.apply_parameters)
        self.apply_btn.setEnabled(False)
        al.addWidget(self.apply_btn)
        self.save_current_btn = QPushButton("Save Current Image Result")
        self.save_current_btn.setToolTip(
            "Process the loaded test image and write its Analysis/ output "
            "(masks + Feature_summary.xlsx), exactly as Run Batch would.")
        self.save_current_btn.clicked.connect(self.save_current_image)
        self.save_current_btn.setEnabled(False)
        al.addWidget(self.save_current_btn)
        self.run_batch_btn = QPushButton("Run Batch")
        self.run_batch_btn.clicked.connect(self.run_batch_processing)
        al.addWidget(self.run_batch_btn)
        layout.addWidget(action_group)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    # =========================================================================
    # DISPLAY PANEL
    # =========================================================================

    def create_display_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Top: one row per channel -- name | role | Show (channel) | Mask | Opacity.
        # Height adapts to the channel count (up to a cap, then scrolls) via
        # _resize_channels_section, called after the grid is (re)built.
        top_group = QGroupBox("Channels")
        tgl = QVBoxLayout(top_group)
        self.channels_scroll = QScrollArea()
        self.channels_scroll.setWidgetResizable(True)
        self.channels_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        grid_container = QWidget()
        self.channel_grid = QGridLayout(grid_container)
        self.channel_grid.setContentsMargins(4, 4, 4, 4)
        self.channel_grid.setHorizontalSpacing(10)
        self.channel_grid.setVerticalSpacing(4)
        self.channel_grid.addWidget(QLabel("Load an image folder"), 0, 0)
        self.channels_scroll.setWidget(grid_container)
        tgl.addWidget(self.channels_scroll)
        self._resize_channels_section(0)
        layout.addWidget(top_group)

        # Image preview
        self.graphics_view = ImageGraphicsView()
        self.graphics_view.pixel_hovered.connect(self.on_pixel_hovered)
        self.graphics_view.cell_clicked.connect(self.on_cell_clicked)
        layout.addWidget(self.graphics_view, stretch=1)

        # Below the image: Fit View, then the (3D-only) Z navigation.
        controls_row = QHBoxLayout()
        controls_row.addStretch()
        fit_btn = QPushButton("Fit View")
        fit_btn.clicked.connect(lambda: self.graphics_view.fit_in_view())
        controls_row.addWidget(fit_btn)
        layout.addLayout(controls_row)

        # Z navigation (3D only): scroll through z-planes of the stack.
        self.z_control = QWidget()
        zrow = QHBoxLayout(self.z_control)
        zrow.setContentsMargins(0, 0, 0, 0)
        zrow.addWidget(QLabel("Z plane:"))
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.setRange(0, 0)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        zrow.addWidget(self.z_slider)
        self.z_pos_label = QLabel("0/0")
        self.z_pos_label.setMinimumWidth(50)
        zrow.addWidget(self.z_pos_label)
        layout.addWidget(self.z_control)
        self.z_control.setVisible(False)

        self.info_label = QLabel("Load a test image to preview parameters — "
                                 "or drag an image / channel folder anywhere here")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)
        return panel

    # =========================================================================
    # FEATURE PANEL
    # =========================================================================

    def _build_filter_group(self):
        filter_group = QGroupBox("Analysis Filter")
        fgl = QVBoxLayout(filter_group)
        fgl.addWidget(QLabel("Only analyze cells whose whole cell"))
        frow = QHBoxLayout()
        self.filter_condition_combo = QComboBox()
        self.filter_condition_combo.addItems(["contains", "doesn't contain"])
        frow.addWidget(self.filter_condition_combo)
        self.filter_stain_combo = QComboBox()
        self.filter_stain_combo.addItem("None")
        self.filter_stain_combo.setToolTip(
            "'contains': keep a cell whose whole-cell stain area is at least the mean\n"
            "area of one particle of this stain in the image (computed per image over\n"
            "all its particles). 'doesn't contain': keep the rest.")
        frow.addWidget(self.filter_stain_combo)
        fgl.addLayout(frow)
        # Edge exclusion margin (global): drops objects (nuclei and stain particles)
        # within N px of the lateral image border, for every method.
        self.edge_margin_spin = self._spin(
            fgl, "Edge Exclusion Margin (px):", self.config['edge_exclusion_margin'], 0, 100)
        return filter_group

    def _build_outlier_group(self):
        outlier_group = QGroupBox("Outlier Removal")
        ogl = QVBoxLayout(outlier_group)
        self.outlier_enable_cb = QCheckBox("Remove outliers (extra sheet)")
        self.outlier_enable_cb.setChecked(self.config.get('remove_outliers', True))
        self.outlier_enable_cb.setToolTip(
            "Adds an 'Outlier_removed' tier (all cells -> filtered -> outlier-removed).\n"
            "Every channel screens the same metric on its own objects — log10 of object\n"
            "size (area in 2D, volume in 3D) — with a two-sided robust z-score. Each\n"
            "channel casts one vote: nuclear/cytoplasm if that cell's own object is an\n"
            "outlier, a stain if the cell owns any outlier particle. A cell is dropped\n"
            "when ANY channel votes. Intensity, texture and shape are not screened.")
        ogl.addWidget(self.outlier_enable_cb)
        self.outlier_threshold_spin = self._dspin(
            ogl, "Robust-z threshold:", self.config.get('outlier_mad_threshold', 4.0),
            0.5, 20.0, 1, 0.5,
            tip=("Robust z = 0.6745*(x-median)/MAD on log10(size), so on well-behaved\n"
                 "data it reads like a standard-deviation count: 4 keeps ~99.99% of\n"
                 "objects. Lower to remove more aggressively. Screening the raw size\n"
                 "instead would flag ~13% of ordinary puncta at any threshold, because\n"
                 "particle sizes are strongly right-skewed."))
        return outlier_group

    def create_feature_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(300)
        layout = QVBoxLayout(panel)

        # Analysis filter + outlier removal sit above the selected-cell features.
        layout.addWidget(self._build_filter_group())
        layout.addWidget(self._build_outlier_group())

        # Sample-level features (whole-image summary), above the per-cell features.
        sample_header = QLabel("Sample-level features")
        sample_header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(sample_header)
        sample_scroll = QScrollArea()
        sample_scroll.setWidgetResizable(True)
        sample_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        sample_scroll.setMaximumHeight(200)
        sample_container = QWidget()
        self.sample_level_layout = QFormLayout(sample_container)
        self.sample_level_layout.setSpacing(4)
        self.sample_level_layout.setContentsMargins(5, 5, 5, 5)
        sample_scroll.setWidget(sample_container)
        layout.addWidget(sample_scroll)

        sep_sl = QFrame()
        sep_sl.setFrameShape(QFrame.Shape.HLine)
        sep_sl.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep_sl)

        header = QLabel("Selected Cell Features")
        header.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(header)
        self.selected_id_label = QLabel("No cell selected")
        self.selected_id_label.setStyleSheet("font-size: 13px; padding: 5px; color: #4682b4;")
        layout.addWidget(self.selected_id_label)

        cat_row = QHBoxLayout()
        self.feature_category_group = QButtonGroup(self)
        self.feature_category_group.setExclusive(True)
        self.feature_category_buttons = {}
        for cat in FEATURE_CATEGORIES:
            btn = QPushButton(cat)
            btn.setCheckable(True)
            btn.setMinimumWidth(60)
            btn.clicked.connect(lambda checked, c=cat: self._on_feature_category_changed(c))
            cat_row.addWidget(btn)
            self.feature_category_group.addButton(btn)
            self.feature_category_buttons[cat] = btn
        cat_row.addStretch()
        self.feature_category_buttons['Nuclear'].setChecked(True)
        # Category buttons appear only once a matching role is assigned (_sync_...).
        for btn in self.feature_category_buttons.values():
            btn.setVisible(False)
        layout.addLayout(cat_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.feature_container = QWidget()
        self.feature_layout = QFormLayout(self.feature_container)
        self.feature_layout.setSpacing(4)
        self.feature_layout.setContentsMargins(5, 5, 5, 5)
        scroll.setWidget(self.feature_container)
        layout.addWidget(scroll, stretch=1)

        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self.clear_selection)
        layout.addWidget(clear_btn)
        self._show_sample_level([])
        return panel

    def _on_feature_category_changed(self, category):
        self.active_feature_category = category
        self._refresh_feature_display()

    # =========================================================================
    # SAMPLE-LEVEL FEATURES  (whole-image summary, refreshed on Apply)
    # =========================================================================

    def _show_sample_level(self, rows):
        """Render a list of (metric, value) pairs in the sample-level panel. SEM
        metrics are not shown in the preview (they remain in the Excel output)."""
        rows = [(m, v) for (m, v) in rows if not str(m).endswith('_sem')]
        while self.sample_level_layout.count():
            item = self.sample_level_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        if not rows:
            lbl = QLabel("Apply on an image to compute sample-level features.")
            lbl.setStyleSheet("color: #888888;")
            lbl.setWordWrap(True)
            self.sample_level_layout.addRow(lbl)
            return
        for metric, value in rows:
            kl = QLabel(str(metric))
            kl.setStyleSheet("color: #cccccc;")
            if isinstance(value, float):
                vs = ("nan" if value != value else
                      (f"{value:.4e}" if (value != 0 and (abs(value) < 0.01 or abs(value) >= 1000))
                       else f"{value:.4f}"))
            else:
                vs = str(value)
            vl = QLabel(vs)
            vl.setStyleSheet("color: #ffffff;")
            vl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.sample_level_layout.addRow(kl, vl)

    # The sample-level rows themselves are computed in the ApplyWorker (see
    # cell_sample_level_rows / particle_sample_level_rows) -- map_nuclei_to_cytoplasm
    # alone can take seconds, which must not happen on the GUI thread. The window only
    # renders them, via _show_sample_level.

    # =========================================================================
    # ROLE-DEPENDENT UI  (tabs & feature categories appear with their role)
    # =========================================================================

    def _assigned_roles(self):
        return {c.currentText() for c in self.channel_role_combos.values()}

    def _sync_detection_tabs(self):
        """Show the Nuclear tab only when a nuclear role exists, and the Cytoplasm
        tab only when a membrane/cytoplasm role exists. Stain tabs are managed per
        stain in _on_role_changed. Pages are kept alive (removeTab, not delete) so
        get_current_config still reads their widgets."""
        roles = self._assigned_roles()
        nuc_idx = self.seg_tabs.indexOf(self.nuclear_tab_page)
        if 'nuclear' in roles and nuc_idx < 0:
            self.seg_tabs.insertTab(0, self.nuclear_tab_page, "Nuclear")
        elif 'nuclear' not in roles and nuc_idx >= 0:
            self.seg_tabs.removeTab(nuc_idx)

        has_cyto = ('membrane' in roles) or ('cytoplasm' in roles)
        cyto_idx = self.seg_tabs.indexOf(self.cyto_tab_page)
        if has_cyto and cyto_idx < 0:
            self.seg_tabs.insertTab(1 if 'nuclear' in roles else 0,
                                    self.cyto_tab_page, "Cytoplasm")
        elif not has_cyto and cyto_idx >= 0:
            self.seg_tabs.removeTab(cyto_idx)

        # Hide the whole tab strip until at least one detection tab exists.
        self.seg_tabs.setVisible(self.seg_tabs.count() > 0)

    def _sync_feature_categories(self):
        """Feature-category buttons appear only for assigned roles. In particle mode
        the Nuclear button is relabelled "Particle" and is the only one shown (it
        holds the primary stain's per-particle features)."""
        roles = self._assigned_roles()
        nuc_btn = self.feature_category_buttons['Nuclear']
        if self.particle_mode:
            nuc_btn.setText('Particle')
            vis = {'Nuclear': True, 'Cytoplasm': False, 'Stain': False}
        else:
            nuc_btn.setText('Nuclear')
            vis = {'Nuclear': 'nuclear' in roles,
                   'Cytoplasm': ('membrane' in roles) or ('cytoplasm' in roles),
                   'Stain': 'stain' in roles}
        for cat, btn in self.feature_category_buttons.items():
            btn.setVisible(vis[cat])
        visible = [c for c in FEATURE_CATEGORIES if vis[c]]
        if visible and self.active_feature_category not in visible:
            self.active_feature_category = visible[0]
            self.feature_category_buttons[visible[0]].setChecked(True)
            self._refresh_feature_display()

    def _sync_role_dependent_ui(self):
        self._sync_detection_tabs()
        self._sync_feature_categories()

    # =========================================================================
    # WIDGET HELPERS
    # =========================================================================

    def _spin(self, layout, label, default, lo, hi, tip=None):
        row = QHBoxLayout()
        lab = QLabel(label)
        row.addWidget(lab)
        sb = QSpinBox()
        sb.setRange(lo, hi)
        sb.setValue(default)
        row.addWidget(sb)
        layout.addLayout(row)
        if tip:
            lab.setToolTip(tip)
            sb.setToolTip(tip)
        return sb

    def _dspin(self, layout, label, default, lo, hi, dec, step, tip=None):
        row = QHBoxLayout()
        lab = QLabel(label)
        row.addWidget(lab)
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setDecimals(dec)
        sb.setSingleStep(step)
        sb.setValue(default)
        row.addWidget(sb)
        layout.addLayout(row)
        if tip:
            lab.setToolTip(tip)
            sb.setToolTip(tip)
        return sb

    def _make_tab_page(self):
        """A vertically-scrolling tab page. Returns (page_widget, inner_layout).

        Detection settings (esp. the tall Cellpose block) are added to
        ``inner_layout``; the surrounding scroll keeps ``seg_tabs`` a stable height.
        """
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(4, 4, 4, 4)
        scroll.setWidget(inner)
        return scroll, lay

    def _build_cellpose_widget(self, defaults):
        """A self-contained Cellpose parameter block (own widgets per detection).

        Returns (widget, widgets_dict) where widgets_dict is read by
        _read_cellpose_widget into a flat params dict for processing.run_cellpose_labels.
        """
        defaults = defaults or {}
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        widgets = {}

        row = QHBoxLayout()
        mlab = QLabel("Model:")
        mlab.setToolTip(CELLPOSE_TIPS['model'])
        row.addWidget(mlab)
        mc = QComboBox()
        mc.addItems(CELLPOSE_MODEL_OPTIONS)
        if defaults.get('model_type', 'cpsam') in CELLPOSE_MODEL_OPTIONS:
            mc.setCurrentText(defaults.get('model_type', 'cpsam'))
        mc.setToolTip(CELLPOSE_TIPS['model'])
        row.addWidget(mc)
        lay.addLayout(row)
        widgets['model_type'] = mc

        widgets['diameter'] = self._spin(lay, "Diameter (0=no rescale):", defaults.get('diameter', 0), 0, 1000,
                                         tip=CELLPOSE_TIPS['diameter'])
        widgets['cellprob_threshold'] = self._dspin(lay, "Cellprob Threshold:", defaults.get('cellprob_threshold', 0.0), -6.0, 6.0, 1, 0.5,
                                                    tip=CELLPOSE_TIPS['cellprob'])
        widgets['flow_threshold'] = self._dspin(lay, "Flow Threshold:", defaults.get('flow_threshold', 0.4), 0.0, 3.0, 2, 0.1,
                                                tip=CELLPOSE_TIPS['flow'])
        widgets['niter'] = self._spin(lay, "Niter (0=auto):", defaults.get('niter', 0), 0, 2000,
                                      tip=CELLPOSE_TIPS['niter'])
        widgets['min_size'] = self._spin(lay, "Cellpose Min Size (px):", defaults.get('min_size', 15), 0, 100000,
                                         tip=CELLPOSE_TIPS['min_size'])
        widgets['max_size_fraction'] = self._dspin(
            lay, "Max Size Fraction:", defaults.get('max_size_fraction', 0.4), 0.05, 1.0, 2, 0.05,
            tip=CELLPOSE_TIPS['max_size_fraction'])
        # GPU is used automatically when available (no user toggle).
        inv = QCheckBox("Invert image")
        inv.setChecked(defaults.get('invert', False))
        inv.setToolTip(CELLPOSE_TIPS['invert'])
        lay.addWidget(inv)
        widgets['invert'] = inv
        return w, widgets

    def _read_cellpose_widget(self, widgets):
        return {
            'model_type': widgets['model_type'].currentText(),
            'diameter': widgets['diameter'].value(),
            'cellprob_threshold': widgets['cellprob_threshold'].value(),
            'flow_threshold': widgets['flow_threshold'].value(),
            'niter': widgets['niter'].value(),
            'min_size': widgets['min_size'].value(),
            'max_size_fraction': widgets['max_size_fraction'].value(),
            'gpu': True,                       # auto: backend clamps to the detected device
            'invert': widgets['invert'].isChecked(),
        }

    # =========================================================================
    # VISIBILITY TOGGLES
    # =========================================================================

    def _update_cyto_visibility(self, _=None):
        source = CYTOPLASM_SOURCE_OPTIONS[self.cytoplasm_source_combo.currentIndex()][1]
        on = source != 'none'
        method = self.cytoplasm_method_combo.currentText()
        self.cyto_method_widget.setVisible(on)
        self.cellpose_widget.setVisible(on and method == 'cellpose')
        self.cyto_threshold_widget.setVisible(on and method == 'threshold')
        self.cyto_preseg_widget.setVisible(on and method == 'presegmented')

    # =========================================================================
    # FILTER COMBO
    # =========================================================================

    def _rebuild_filter_combos(self):
        current = self.filter_stain_combo.currentText()
        self.filter_stain_combo.clear()
        self.filter_stain_combo.addItem("None")
        for stem, combo in self.channel_role_combos.items():
            if combo.currentText() == 'stain':
                self.filter_stain_combo.addItem(stem)
        idx = self.filter_stain_combo.findText(current)
        if idx >= 0:
            self.filter_stain_combo.setCurrentIndex(idx)

    # =========================================================================
    # CHANNEL ROLES
    # =========================================================================

    def populate_channel_roles(self, stems):
        # Drop any stain tabs left over from a previously-loaded image.
        for old_stem, page in list(self.stain_param_groups.items()):
            idx = self.seg_tabs.indexOf(page)
            if idx >= 0:
                self.seg_tabs.removeTab(idx)
            page.deleteLater()
        self.stain_param_groups.clear()
        self.stain_param_widgets.clear()
        self.particle_mode = False

        # Rebuild the per-channel grid: name | role | contrast | show | mask | opacity.
        clear_layout(self.channel_grid)
        self.channel_role_combos.clear()
        self.channel_checkboxes.clear()
        self.channel_contrast_sliders.clear()
        self.channel_colors.clear()
        self.mask_row_checkboxes.clear()
        self.mask_row_sliders.clear()

        for col, htext in enumerate(("Channel", "Role", "Contrast", "Show", "Mask", "Opacity")):
            hl = QLabel(htext)
            hl.setStyleSheet("font-weight: bold; color: #aaaaaa;")
            self.channel_grid.addWidget(hl, 0, col)

        for i, stem in enumerate(stems):
            color = channel_color(i)
            self.channel_colors[stem] = color
            r = i + 1

            name = QLabel(stem)
            name.setStyleSheet(f"color: rgb({color[0]},{color[1]},{color[2]});")
            self.channel_grid.addWidget(name, r, 0)

            combo = QComboBox()
            combo.addItems(ROLE_OPTIONS)
            combo.currentTextChanged.connect(lambda text, s=stem: self._on_role_changed(s, text))
            self.channel_grid.addWidget(combo, r, 1)
            self.channel_role_combos[stem] = combo

            csl = QSlider(Qt.Orientation.Horizontal)
            csl.setRange(0, 100)
            csl.setValue(50)                           # 50 = 1.0 gain (unchanged)
            csl.setMinimumWidth(80)
            csl.setToolTip("Display contrast for this channel")
            csl.valueChanged.connect(self.update_display)
            self.channel_grid.addWidget(csl, r, 2)
            self.channel_contrast_sliders[stem] = csl

            ccb = QCheckBox()
            ccb.setChecked(True)
            ccb.setToolTip(f"Show the {stem} channel")
            ccb.stateChanged.connect(self.update_display)
            self.channel_grid.addWidget(ccb, r, 3, Qt.AlignmentFlag.AlignCenter)
            self.channel_checkboxes[stem] = ccb

            mcb = QCheckBox()
            mcb.setChecked(True)
            mcb.setEnabled(False)                      # enabled once a mask exists
            mcb.setToolTip("Show this channel's mask (after Apply)")
            mcb.stateChanged.connect(self.update_display)
            self.channel_grid.addWidget(mcb, r, 4, Qt.AlignmentFlag.AlignCenter)
            self.mask_row_checkboxes[stem] = mcb

            sld = QSlider(Qt.Orientation.Horizontal)
            sld.setRange(0, 100)
            sld.setValue(50)
            sld.setMinimumWidth(90)
            sld.setEnabled(False)
            sld.setToolTip("Mask opacity")
            sld.valueChanged.connect(self.update_display)
            self.channel_grid.addWidget(sld, r, 5)
            self.mask_row_sliders[stem] = sld

        self.channel_grid.setColumnStretch(2, 1)
        self.channel_grid.setColumnStretch(5, 1)
        self._resize_channels_section(len(stems))

        if len(stems) == 1:
            self.channel_role_combos[stems[0]].setCurrentText('nuclear')

        self._rebuild_filter_combos()
        self._sync_role_dependent_ui()

    def _on_role_changed(self, stem, new_role):
        self.particle_mode = False
        if stem in self.stain_param_groups:
            page = self.stain_param_groups.pop(stem)
            self.stain_param_widgets.pop(stem, None)
            idx = self.seg_tabs.indexOf(page)
            if idx >= 0:
                self.seg_tabs.removeTab(idx)
            page.deleteLater()
        if new_role == 'stain':
            self._create_stain_params(stem)
        self._rebuild_filter_combos()
        self._sync_role_dependent_ui()

    def _create_stain_params(self, stem):
        d = DEFAULT_STAIN_CONFIG.copy()
        page, gl = self._make_tab_page()
        w = {}

        # Method selector (independent of role)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Method:"))
        mc = QComboBox()
        mc.addItems(SEGMENTATION_METHODS)
        mc.setCurrentText(d.get('method', 'threshold'))
        mrow.addWidget(mc)
        gl.addLayout(mrow)
        w['method'] = mc

        # Threshold-method controls
        tw = QWidget()
        tl = QVBoxLayout(tw)
        tl.setContentsMargins(0, 0, 0, 0)
        w['threshold_min'] = self._spin(tl, "Threshold Min:", d['threshold_min'], 0, 65535)
        w['threshold_max'] = self._spin(tl, "Threshold Max:", d['threshold_max'], 0, 65535)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Output Type:"))
        oc = QComboBox()
        oc.addItems(STAIN_OUTPUT_OPTIONS)
        orow.addWidget(oc)
        tl.addLayout(orow)
        w['output_type'] = oc
        lw = QWidget()
        ll = QVBoxLayout(lw)
        ll.setContentsMargins(0, 0, 0, 0)
        w['doublet_threshold'] = self._dspin(ll, "Doublet Threshold:", d['doublet_threshold'], 0.0, 1.0, 2, 0.05)
        w['watershed_min_distance'] = self._spin(ll, "Watershed Min Distance:", d['watershed_min_distance'], 1, 500)
        tl.addWidget(lw)
        w['labeled_widget'] = lw
        lw.setVisible(False)
        oc.currentTextChanged.connect(lambda t, _w=lw: _w.setVisible(t == 'labeled'))
        gl.addWidget(tw)
        w['threshold_widget'] = tw

        # Cellpose-method controls (own params)
        cw, cwidgets = self._build_cellpose_widget(d.get('cellpose', {}))
        gl.addWidget(cw)
        w['cellpose_widget'] = cw
        w['cellpose_widgets'] = cwidgets

        # Object size filtering is handled per method (Cellpose "Min Size"); there is
        # no separate stain min-size.
        none_hint = QLabel("Not segmented — reported as a per-cell mean intensity only.")
        none_hint.setStyleSheet("color: #888888;")
        none_hint.setWordWrap(True)
        gl.addWidget(none_hint)
        w['none_hint'] = none_hint

        def _toggle_stain_method(_=None, _mc=mc, _tw=tw, _cw=cw, _hint=none_hint):
            m = _mc.currentText()
            _tw.setVisible(m == 'threshold')
            _cw.setVisible(m == 'cellpose')
            _hint.setVisible(m == 'none')
        mc.currentTextChanged.connect(_toggle_stain_method)
        _toggle_stain_method()

        gl.addStretch()
        idx = self.seg_tabs.addTab(page, f"Stain: {stem}")
        self.seg_tabs.setCurrentIndex(idx)
        self.stain_param_groups[stem] = page
        self.stain_param_widgets[stem] = w

    # =========================================================================
    # MASK TOGGLES
    # =========================================================================

    def _mask_color(self, key):
        """Color for a mask layer; stain masks reuse their own channel's color."""
        if key in MASK_COLORS:
            return MASK_COLORS[key]
        if key.startswith('stain_'):
            return self.channel_colors.get(key[len('stain_'):], (255, 255, 255))
        return (255, 255, 255)

    # Per-row height + header + group/scroll chrome, and the cap (rows before it
    # starts scrolling). Keeps the Channels section as tall as it needs to be.
    _CHANNEL_ROW_PX = 30
    _CHANNEL_ROWS_CAP = 8

    def _resize_channels_section(self, n_channels):
        """Size the Channels scroll area to the number of channels (capped)."""
        rows = min(max(n_channels, 1) + 1, self._CHANNEL_ROWS_CAP + 1)   # + header row
        self.channels_scroll.setFixedHeight(rows * self._CHANNEL_ROW_PX + 14)

    def _mask_key_for_stem(self, stem):
        """Mask-data key produced by a channel given its current role (or None)."""
        combo = self.channel_role_combos.get(stem)
        role = combo.currentText() if combo else 'none'
        if role == 'nuclear':
            return 'nuclear_labels'
        if role in ('membrane', 'cytoplasm'):
            return 'cytoplasm_labels'
        if role == 'stain':
            return f'stain_{stem}'
        return None

    def _mask_checkbox_for_key(self, key):
        """The per-channel Mask checkbox that controls mask-data ``key`` (or None)."""
        for stem, mcb in self.mask_row_checkboxes.items():
            if self._mask_key_for_stem(stem) == key:
                return mcb
        return None

    def rebuild_mask_toggles(self):
        """Refresh available masks + enable each channel row's Mask/Opacity controls.

        The per-channel widgets already exist (built with the grid); this only
        repopulates ``mask_data`` and toggles each row's enabled state by whether
        that channel's role currently has a mask. In particle mode ``nuclear_labels``
        holds the primary stain (for click only) and is not shown as its own mask.
        """
        self.mask_data.clear()
        if self.nuclear_labels is not None and not self.particle_mode:
            self.mask_data['nuclear_labels'] = self.nuclear_labels
        if self.cytoplasm_labels is not None:
            self.mask_data['cytoplasm_labels'] = self.cytoplasm_labels
        for stem, (mask, _) in self.stain_results.items():
            self.mask_data[f'stain_{stem}'] = mask

        for stem, mcb in self.mask_row_checkboxes.items():
            key = self._mask_key_for_stem(stem)
            avail = key is not None and key in self.mask_data
            mcb.setEnabled(avail)
            self.mask_row_sliders[stem].setEnabled(avail)

    # =========================================================================
    # 3D / Z NAVIGATION
    # =========================================================================

    def _plane(self, arr):
        """Current 2D plane of a (possibly 3D) array, for display/interaction."""
        if self.is_3d and getattr(arr, 'ndim', 2) == 3:
            return arr[min(self.current_z, arr.shape[0] - 1)]
        return arr

    def _on_z_changed(self, value):
        self.current_z = value
        self.z_pos_label.setText(f"{value + 1}/{self.n_z}")
        self.update_display()

    def _update_dim_label(self, _=None):
        if not self.is_3d:
            self.dim_label.setText("2D image")
            return
        px = self.pixel_size_spin.value()
        aniso = (self.z_size_spin.value() / px) if px else 1.0
        self.dim_label.setText(f"3D stack: {self.n_z} planes · anisotropy {aniso:.3g}")

    def _apply_metadata_pixel_size(self):
        """Fill the XY pixel-size field from the image metadata when a plausible
        value is present. The field stays editable so the user can override a
        missing or wrong calibration."""
        px = self.meta_pixel_size_um
        if px is not None and np.isfinite(px) and px > 0:
            self.pixel_size_spin.blockSignals(True)
            self.pixel_size_spin.setValue(float(px))
            self.pixel_size_spin.blockSignals(False)

    def _configure_dimensionality_ui(self):
        """Show/hide 3D controls and auto-fill xy/z calibration after a load."""
        self._apply_metadata_pixel_size()
        self.z_control.setVisible(self.is_3d)
        self.z_size_row.setVisible(self.is_3d)
        if self.is_3d:
            self.z_slider.blockSignals(True)
            self.z_slider.setRange(0, max(0, self.n_z - 1))
            self.z_slider.setValue(self.current_z)
            self.z_slider.blockSignals(False)
            self.z_pos_label.setText(f"{self.current_z + 1}/{self.n_z}")
            # xy stays user-authoritative; only z-step is taken from the metadata.
            if self.meta_z_size_um:
                self.z_size_spin.setValue(float(self.meta_z_size_um))
        # Doublet/watershed splitting is 2D-only (3D relies on Cellpose, plan D2).
        for w in (self.doublet_threshold_spin, self.watershed_distance_spin):
            w.setEnabled(not self.is_3d)
        self._update_dim_label()

    # =========================================================================
    # COMPOSITING
    # =========================================================================

    def _contrast_gain(self, stem):
        """Display gain from a channel's contrast slider (0-100, 50 = 1.0 = unchanged)."""
        sld = self.channel_contrast_sliders.get(stem)
        return (sld.value() / 50.0) if sld is not None else 1.0

    def composite_display(self):
        if not self.channels or not self.normalized_channels:
            return None
        first = self._plane(next(iter(self.normalized_channels.values())))
        h, w = first.shape
        composite = np.zeros((h, w, 3), dtype=np.float64)

        any_ch = False
        for stem, cb in self.channel_checkboxes.items():
            if cb.isChecked() and stem in self.normalized_channels:
                any_ch = True
                gray = self._plane(self.normalized_channels[stem]).astype(np.float64) / 255.0
                gray = np.clip(gray * self._contrast_gain(stem), 0.0, 1.0)
                color = self.channel_colors.get(stem, (200, 200, 200))
                for c in range(3):
                    composite[:, :, c] += gray * (color[c] / 255.0)

        result = (np.clip(composite * 255, 0, 255) if any_ch
                  else np.zeros((h, w, 3))).astype(np.float64)

        # Each channel row's mask is blended with its own opacity, in row order.
        for stem in self.channel_checkboxes:
            mcb = self.mask_row_checkboxes.get(stem)
            if mcb is None or not mcb.isChecked():
                continue
            key = self._mask_key_for_stem(stem)
            if key is None or key not in self.mask_data:
                continue
            data = self._plane(self.mask_data[key])
            px = data > 0
            if not np.any(px):
                continue
            is_bin = data.dtype == bool or (data.dtype != np.uint16 and data.max() <= 1)
            if is_bin:
                base_color = self._mask_color(key)
                layer = np.zeros((h, w, 3), dtype=np.float64)
                for c in range(3):
                    layer[:, :, c][px] = base_color[c]
            else:
                layer = apply_glasbey_lut(data).astype(np.float64)
            alpha = self.mask_row_sliders[stem].value() / 100.0
            result[px] = (1 - alpha) * result[px] + alpha * layer[px]

        return np.ascontiguousarray(np.clip(result, 0, 255).astype(np.uint8))

    # =========================================================================
    # DISPLAY
    # =========================================================================

    def update_display(self, _=None):
        if not self.channels:
            return
        display = self.composite_display()
        if display is None:
            return
        self.graphics_view.set_pixmap(array_to_qpixmap(display))

        # Which labels a click resolves against: in particle mode the primary
        # stain; otherwise cytoplasm (if its mask is shown) else nuclei.
        active_labels = None
        if self.particle_mode and self.nuclear_labels is not None:
            active_labels = self.nuclear_labels
        else:
            cb = self._mask_checkbox_for_key('cytoplasm_labels')
            if self.cytoplasm_labels is not None and cb and cb.isChecked():
                active_labels = self.cytoplasm_labels
            if active_labels is None and self.nuclear_labels is not None:
                cb = self._mask_checkbox_for_key('nuclear_labels')
                if cb and cb.isChecked():
                    active_labels = self.nuclear_labels

        raw_for_hover = None
        for stem, cb in self.channel_checkboxes.items():
            if cb.isChecked() and stem in self.channels:
                raw_for_hover = self.channels[stem]
                break
        self.graphics_view.set_images(
            self._plane(raw_for_hover) if raw_for_hover is not None else None,
            self._plane(active_labels) if active_labels is not None else None)
        if self.selected_cell_id and self.selected_cell_id > 0:
            self.graphics_view.highlight_label(self.selected_cell_id)

    # =========================================================================
    # BROWSING / LOADING
    # =========================================================================

    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_edit.setText(folder)

    def _input_mode(self):
        return INPUT_MODE_OPTIONS[self.input_mode_combo.currentIndex()][1]

    def browse_test_image(self):
        # In file-per-sample mode a single multi-channel file is the sample.
        if self._input_mode() == 'file_per_sample':
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Sample Image", "", "TIFF images (*.tif *.tiff)")
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if path:
            self.test_image_edit.setText(path)
            self.load_test_image(path)

    # --- Drag & drop: drop a channel folder or a single multi-channel image ------

    def _droppable(self, path):
        """True if ``path`` is a folder or a supported image file."""
        p = Path(path)
        return p.is_dir() or (p.is_file() and p.suffix.lower() in supported_image_extensions())

    def _resolve_dropped_target(self, paths):
        """Choose the sample target from dropped items: a single file/folder loads
        directly; several image files sharing one parent load that parent as a folder."""
        paths = [Path(p) for p in paths if p]
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0]
        files = [p for p in paths if p.is_file()]
        parents = {p.parent for p in files}
        if files and len(files) == len(paths) and len(parents) == 1:
            return next(iter(parents))           # dropped loose channel files -> their folder
        return paths[0]

    def _drop_local_paths(self, event):
        md = event.mimeData()
        return [u.toLocalFile() for u in md.urls() if u.toLocalFile()] if md.hasUrls() else []

    def dragEnterEvent(self, event):
        if any(self._droppable(p) for p in self._drop_local_paths(event)):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        target = self._resolve_dropped_target(self._drop_local_paths(event))
        if target is None or not self._droppable(target):
            event.ignore()
            QMessageBox.warning(
                self, "Unsupported drop",
                "Drop a folder of channel images, or a single multi-channel image "
                f"({', '.join(sorted(supported_image_extensions()))}).")
            return
        event.acceptProposedAction()
        self.test_image_edit.setText(str(target))
        try:
            self.load_test_image(target)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"Could not load {Path(target).name}:\n{e}")

    def load_test_image(self, path):
        path = Path(path)
        if path.is_file():
            # File-per-sample: one multi-channel file is a sample (bare c0/c1 channels).
            self.channels, sample_meta = load_file_as_sample(path)
            stems = sorted(self.channels.keys())
            if not stems:
                QMessageBox.warning(self, "Error", "No channels found in file")
                return
        else:
            discovered = discover_channels(path)
            if not discovered:
                QMessageBox.warning(self, "Error", "No .tif/.tiff files found")
                return
            stems = [s for s, _ in discovered]
            self.channels, sample_meta = load_channels_with_meta(path)
        self.populate_channel_roles(stems)
        self.normalized_channels = {s: normalize_to_uint8(a) for s, a in self.channels.items()}
        self.test_image_path = path

        # Dimensionality + voxel calibration auto-detected from the TIFF metadata.
        self.is_3d = bool(sample_meta.get('is_3d'))
        self.meta_pixel_size_um = sample_meta.get('pixel_size_um')
        self.meta_z_size_um = sample_meta.get('z_size_um')
        self.n_z = self.channels[stems[0]].shape[0] if self.is_3d else 1
        self.current_z = self.n_z // 2
        self._configure_dimensionality_ui()

        self.nuclear_labels = None
        self.cytoplasm_labels = None
        self.nuclear_features = None
        self.cytoplasm_features = None
        self.stain_results.clear()
        self.stain_per_nucleus.clear()
        self.stain_per_cytoplasm.clear()
        self.stain_spatial_nucleus.clear()
        self.rebuild_mask_toggles()
        self._show_sample_level([])

        self.apply_btn.setEnabled(True)
        self.save_current_btn.setEnabled(True)
        ph, pw = self._plane(self.channels[stems[0]]).shape
        dim = f"3D {self.n_z}×{ph}×{pw}" if self.is_3d else f"{pw}×{ph}"
        self.info_label.setText(
            f"Loaded: {self.test_image_path.name} ({dim}) — {len(stems)} channel(s)")
        self.clear_selection()
        self.update_display()
        self.graphics_view.fit_in_view()

    # =========================================================================
    # CONFIG
    # =========================================================================

    def get_current_config(self):
        channel_roles = {s: c.currentText() for s, c in self.channel_role_combos.items()}
        source_idx = self.cytoplasm_source_combo.currentIndex()
        cyto_source = CYTOPLASM_SOURCE_OPTIONS[source_idx][1]

        # Pre-segmented is a *method* choice; translate to the backend's flags.
        nuclear_method = self.nuclear_method_combo.currentText()
        cyto_method = self.cytoplasm_method_combo.currentText()
        use_preseg_nuc = (nuclear_method == 'presegmented')
        use_preseg_cyto = (cyto_method == 'presegmented' and cyto_source != 'none')

        stain_configs = {}
        for stem, w in self.stain_param_widgets.items():
            stain_configs[stem] = {
                'method': w['method'].currentText(),
                'cellpose': self._read_cellpose_widget(w['cellpose_widgets']),
                'threshold_min': w['threshold_min'].value(),
                'threshold_max': w['threshold_max'].value(),
                'min_object_size': 0,          # stain min-size removed (Cellpose min_size covers it)
                'output_type': w['output_type'].currentText(),
                'doublet_threshold': w['doublet_threshold'].value(),
                'watershed_min_distance': w['watershed_min_distance'].value(),
            }

        fs = self.filter_stain_combo.currentText()

        return {
            'input_mode': self._input_mode(),
            'num_threads': None,          # auto: detected logical cores minus one
            'pixel_size_um': self.pixel_size_spin.value(),
            'z_size_um': (self.z_size_spin.value() if self.is_3d else None),
            'nuclear_method': nuclear_method,
            'nuclear_cellpose': self._read_cellpose_widget(self.nuclear_cellpose_widgets),
            'threshold_min': self.threshold_min_spin.value(),
            'threshold_max': self.threshold_max_spin.value(),
            # Nuclear min-object-size filter removed entirely (no size filtering).
            'min_object_size': 0,
            'connectivity': self.connectivity_spin.value(),
            'edge_exclusion_margin': self.edge_margin_spin.value(),
            'doublet_threshold': self.doublet_threshold_spin.value(),
            'watershed_min_distance': self.watershed_distance_spin.value(),
            'cytoplasm_source': cyto_source,
            'cytoplasm_method': cyto_method,
            'cytoplasm_threshold': {
                'threshold_min': self.cyto_thr_min_spin.value(),
                'threshold_max': self.cyto_thr_max_spin.value(),
                'min_object_size': self.cyto_min_size_spin.value(),
            },
            'cellpose_model_type': self.cellpose_model_combo.currentText(),
            'cellpose_diameter': self.cellpose_diameter_spin.value(),
            'cellpose_cellprob_threshold': self.cellpose_cellprob_spin.value(),
            'cellpose_flow_threshold': self.cellpose_flow_spin.value(),
            'cellpose_niter': self.cellpose_niter_spin.value(),
            'cellpose_min_size': self.cellpose_min_size_spin.value(),
            'cellpose_max_size_fraction': self.cellpose_max_size_frac_spin.value(),
            'cellpose_gpu': True,
            'cellpose_invert': self.cellpose_invert_cb.isChecked(),
            'channel_roles': channel_roles,
            'stain_configs': stain_configs,
            'filter_stain': fs if fs != 'None' else None,
            'filter_condition': 'contains' if self.filter_condition_combo.currentIndex() == 0 else 'doesnt_contain',
            'remove_outliers': self.outlier_enable_cb.isChecked(),
            'outlier_mad_threshold': self.outlier_threshold_spin.value(),
            'use_presegmented_nuclear': use_preseg_nuc,
            'presegmented_nuclear_stem': self.preseg_nuclear_stem_edit.text().strip(),
            'use_presegmented_cytoplasm': use_preseg_cyto,
            'presegmented_cytoplasm_stem': self.preseg_cyto_stem_edit.text().strip(),
        }

    # =========================================================================
    # APPLY
    # =========================================================================

    def apply_parameters(self):
        """Validate, snapshot the config, and hand the work to an ApplyWorker.

        Nothing heavy runs here: segmenting one field takes seconds (and the first
        Cellpose call also loads its weights), which would otherwise freeze the window.
        """
        if not self.channels or self._apply_busy():
            return

        config = self.get_current_config()
        # Resolve voxel calibration and inject it (z-size + anisotropy) exactly like
        # the batch path, so the 3D preview uses correct um^3 + do_3D parameters.
        sample_meta = {'is_3d': self.is_3d,
                       'pixel_size_um': self.meta_pixel_size_um,
                       'z_size_um': self.meta_z_size_um}
        px, pz, aniso = resolve_calibration(config, sample_meta)
        config = _apply_sample_calibration(config, px, pz, aniso)
        channel_roles = config['channel_roles']

        nuclear_stem = get_role_stem(channel_roles, 'nuclear')
        stain_roles = [s for s, r in channel_roles.items() if r == 'stain' and s in self.channels]
        # No active nuclear segmentation (no nuclear role, or its method is 'none')
        # -> particle mode.
        particle_mode = (nuclear_stem is None
                         or config.get('nuclear_method', 'threshold') == 'none')
        if particle_mode:
            if not stain_roles:
                QMessageBox.warning(self, "Error", "Assign a channel as 'nuclear' or 'stain'")
                return
        elif nuclear_stem not in self.channels:
            QMessageBox.warning(self, "Error", "Nuclear channel not found in loaded image")
            return

        self._set_apply_busy(True)
        self.info_label.setText("Processing...")
        self.apply_worker = ApplyWorker(self.channels, config, self.test_image_path,
                                        self.is_3d, particle_mode)
        self.apply_worker.progress.connect(self._apply_progress)
        self.apply_worker.done.connect(self._apply_done)
        self.apply_worker.failed.connect(self._apply_failed)
        self.apply_worker.start()

    def _apply_busy(self):
        w = getattr(self, 'apply_worker', None)
        return w is not None and w.isRunning()

    def _set_apply_busy(self, busy):
        """Lock the controls that would corrupt an in-flight preview -- notably anything
        that can swap out self.channels while the worker is reading it."""
        for btn in (self.apply_btn, self.save_current_btn, self.run_batch_btn,
                    self.test_image_browse_btn):
            btn.setEnabled(not busy)
        self.setAcceptDrops(not busy)          # drag-and-drop also loads a new image

    def _apply_progress(self, msg):
        self.info_label.setText(msg)

    def _apply_done(self, res):
        self.particle_mode = res.particle_mode
        self.nuclear_labels = res.nuclear_labels
        self.nuclear_features = res.nuclear_features
        self.cytoplasm_labels = res.cytoplasm_labels
        self.cytoplasm_features = res.cytoplasm_features
        self.stain_results = dict(res.stain_results)
        self.stain_per_nucleus = dict(res.stain_per_nucleus)
        self.stain_per_cytoplasm = dict(res.stain_per_cytoplasm)
        self.stain_spatial_nucleus = dict(res.stain_spatial_nucleus)

        self.rebuild_mask_toggles()
        self._sync_feature_categories()        # shows the "Particle" tab in particle mode
        self._show_sample_level(res.sample_rows)
        self.info_label.setText(" | ".join(res.info_parts) if res.info_parts else "No stains")
        self.clear_selection()
        self.update_display()
        self._set_apply_busy(False)

    def _apply_failed(self, msg):
        self.info_label.setText("Processing failed")
        self._set_apply_busy(False)
        QMessageBox.critical(self, "Error", f"Processing failed: {msg}")

    # =========================================================================
    # HOVER / CLICK / FEATURES
    # =========================================================================

    def on_pixel_hovered(self, x, y, intensity, label_id):
        zpart = f" Z: {self.current_z}" if self.is_3d else ""
        self.coord_label.setText(f"X: {x} Y: {y}{zpart}")
        self.intensity_label.setText(f"Intensity: {intensity}")
        self.hover_label_label.setText(f"Label: {label_id}")

    def on_cell_clicked(self, label_id):
        if label_id <= 0:
            self.clear_selection()
            return
        self.selected_cell_id = label_id
        self.selected_id_label.setText(f"Cell ID: {label_id}")
        self._refresh_feature_display()

    def _refresh_feature_display(self):
        while self.feature_layout.count():
            item = self.feature_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        lid = self.selected_cell_id
        if not lid or lid <= 0:
            h = QLabel("Click on a cell to view features")
            h.setStyleSheet("color: #888888;")
            self.feature_layout.addRow(h)
            return

        cat = self.active_feature_category

        if cat == 'Nuclear':
            f = next((x for x in (self.nuclear_features or []) if x.get('cell_id') == lid), None)
            if f is None:
                self.feature_layout.addRow(QLabel("No nuclear features"))
                return
            morph = ['area_um2', 'volume_um3', 'perimeter_um', 'surface_area_um2',
                     'circularity', 'sphericity', 'eccentricity', 'elongation', 'flatness',
                     'solidity', 'major_axis_length_um', 'minor_axis_length_um',
                     'convexity_defects', 'centroid_x', 'centroid_y', 'centroid_z']
            intens = ['intensity_mean', 'intensity_sd']
            tex = ['texture_contrast', 'texture_homogeneity', 'texture_energy', 'texture_correlation']
            self._add_section("Morphology", f, morph)
            self._add_section("Intensity", f, intens)
            self._add_section("Texture", f, tex)
            ch = sorted(k for k in f if k.endswith('_mean') and k not in intens)
            if ch:
                self._add_section("Channels", f, ch)

        elif cat == 'Cytoplasm':
            f = next((x for x in (self.cytoplasm_features or []) if x.get('cell_id') == lid), None)
            if f is None:
                self.feature_layout.addRow(QLabel("No cytoplasm features"))
                return
            morph = ['area_um2', 'volume_um3', 'perimeter_um', 'surface_area_um2',
                     'circularity', 'sphericity', 'eccentricity', 'elongation', 'flatness',
                     'solidity', 'major_axis_length_um', 'minor_axis_length_um',
                     'convexity_defects', 'centroid_x', 'centroid_y', 'centroid_z']
            intens = ['intensity_mean', 'intensity_sd']
            tex = ['texture_contrast', 'texture_homogeneity', 'texture_energy', 'texture_correlation']
            self._add_section("Morphology", f, morph)
            self._add_section("Intensity", f, intens)
            self._add_section("Texture", f, tex)
            ch = sorted(k for k in f if k.endswith('_mean') and k not in intens)
            if ch:
                self._add_section("Channels", f, ch)

        elif cat == 'Stain':
            has_any = False
            for stem in sorted(self.stain_per_nucleus.keys()):
                nuc_data = self.stain_per_nucleus.get(stem, {}).get(lid)
                cyto_data = self.stain_per_cytoplasm.get(stem, {}).get(lid)
                spatial = self.stain_spatial_nucleus.get(stem, {}).get(lid)
                if nuc_data is None and cyto_data is None and spatial is None:
                    continue
                has_any = True
                h = QLabel(f"Stain: {stem}")
                h.setStyleSheet("font-weight: bold; margin-top: 8px; color: #4682b4;")
                self.feature_layout.addRow(h)
                if nuc_data:
                    self._add_section("Per Nucleus", nuc_data,
                                      ['stain_measure_cal', 'stain_particle_count', 'stain_coverage_fraction'])
                if spatial:
                    self._add_section("Spatial (Nucleus)", spatial,
                                      ['n_stain_particles', 'particle_alignment',
                                       'peak_count_major', 'freq_major_axis',
                                       'peak_count_minor', 'freq_minor_axis'])
                if cyto_data:
                    self._add_section("Per Cytoplasm", cyto_data,
                                      ['stain_measure_cal', 'stain_particle_count', 'stain_coverage_fraction'])

            if not has_any:
                self.feature_layout.addRow(QLabel("No stain data"))

    def _add_feature_row(self, key, value):
        kl = QLabel(f"{key}:")
        kl.setStyleSheet("color: #cccccc;")
        if isinstance(value, float):
            vs = f"{value:.4e}" if (abs(value) < 0.01 or abs(value) >= 1000) else f"{value:.4f}"
        else:
            vs = str(value)
        vl = QLabel(vs)
        vl.setStyleSheet("color: #ffffff;")
        vl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.feature_layout.addRow(kl, vl)

    def _add_section(self, name, features, keys):
        lbl = QLabel(name)
        lbl.setStyleSheet("font-weight: bold; margin-top: 8px; color: #aaaaaa;")
        self.feature_layout.addRow(lbl)
        for key in keys:
            if key in features and key != 'cell_id':
                self._add_feature_row(key, features[key])

    def clear_selection(self):
        self.selected_cell_id = None
        self.selected_id_label.setText("No cell selected")
        self.graphics_view.highlight_label(0)
        while self.feature_layout.count():
            item = self.feature_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        h = QLabel("Click on a cell to view features")
        h.setStyleSheet("color: #888888;")
        self.feature_layout.addRow(h)

    # =========================================================================
    # SAVE CURRENT / BATCH
    # =========================================================================

    def save_current_image(self):
        if not self.channels or self.test_image_path is None:
            QMessageBox.warning(self, "Error", "Load a test image first")
            return
        config = self.get_current_config()
        if not any(r in ('nuclear', 'stain') for r in config['channel_roles'].values()):
            QMessageBox.warning(self, "Error", "Assign at least one channel as 'nuclear' or 'stain'")
            return

        path = Path(self.test_image_path)
        if path.is_file():
            spec = SampleSpec(name=path.stem, sample_dir=path.parent / path.stem,
                              load=(lambda p=path: load_file_as_sample(p)))
        else:
            spec = SampleSpec(name=path.name, sample_dir=path,
                              load=(lambda p=path: load_channels_with_meta(p)))

        self.save_current_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.run_batch_btn.setEnabled(False)
        self.info_label.setText("Saving current image result...")
        self.save_worker = SaveCurrentWorker(spec, config)
        self.save_worker.finished.connect(self._save_current_finished)
        self.save_worker.error.connect(self._save_current_error)
        self.save_worker.start()

    def _save_current_finished(self, out_dir):
        self.save_current_btn.setEnabled(True)
        self.apply_btn.setEnabled(bool(self.channels))
        self.run_batch_btn.setEnabled(True)
        self.info_label.setText("Saved current image result")
        QMessageBox.information(self, "Saved", f"Result written to:\n{out_dir}")

    def _save_current_error(self, msg):
        self.save_current_btn.setEnabled(True)
        self.apply_btn.setEnabled(bool(self.channels))
        self.run_batch_btn.setEnabled(True)
        self.info_label.setText("Save failed")
        QMessageBox.critical(self, "Error", f"Save failed:\n{msg}")

    # =========================================================================
    # BATCH
    # =========================================================================

    def run_batch_processing(self):
        input_folder = self.input_folder_edit.text()
        if not input_folder or not Path(input_folder).exists():
            QMessageBox.warning(self, "Error", "Please select a valid input folder")
            return
        config = self.get_current_config()
        roles = config['channel_roles'].values()
        # Nuclear is optional: a stain-only batch runs in particle mode.
        if not any(r in ('nuclear', 'stain') for r in roles):
            QMessageBox.warning(self, "Error", "Assign at least one channel as 'nuclear' or 'stain'")
            return

        reply = QMessageBox.question(
            self, "Confirm Batch",
            f"Start batch processing on:\n{input_folder}\n\nThis may take a while.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self.run_batch_btn.setEnabled(False)
        self.apply_btn.setEnabled(False)
        self.info_label.setText("Batch processing...")

        self.batch_worker = BatchWorker(input_folder, config)
        self.batch_worker.finished.connect(self.batch_finished)
        self.batch_worker.error.connect(self.batch_error)
        self.batch_worker.start()

    def batch_finished(self):
        self.run_batch_btn.setEnabled(True)
        self.apply_btn.setEnabled(bool(self.channels))
        self.info_label.setText("Batch processing completed!")
        QMessageBox.information(
            self, "Complete",
            "Batch processing completed!\n\n"
            "A QC report (Batch_QC_report.pdf) was generated at the batch root "
            "(see the console if generation was skipped).")

    def batch_error(self, error_msg):
        self.run_batch_btn.setEnabled(True)
        self.apply_btn.setEnabled(bool(self.channels))
        self.info_label.setText("Batch failed")
        QMessageBox.critical(self, "Error", f"Batch failed:\n{error_msg}")


def main():
    # Windows groups a bare python.exe process under python's own taskbar icon and
    # ignores setWindowIcon there. Claiming an explicit AppUserModelID first makes the
    # taskbar button ours. Harmless if it fails, and a no-op off Windows.
    if sys.platform == 'win32':
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('Vairons.App')
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("Vairons")
    app.setWindowIcon(app_icon())
    # Fusion + an explicit dark palette gives an identical look on Windows and macOS
    # (rather than each platform's native widget style).
    app.setStyle('Fusion')
    app.setPalette(create_dark_palette())
    app.setStyleSheet(DARK_STYLESHEET)
    window = VaironsUI()
    window.show()
    window.raise_()               # bring to front (macOS when launched from a terminal)
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()