import json
import os
import warnings
import argparse
from pathlib import Path
import sys
import textwrap

import numpy as np
import matplotlib
if '--headless' in sys.argv or '--save-image' in sys.argv:
    matplotlib.use('Agg')
else:
    if os.environ.get('WAYLAND_DISPLAY'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'wayland')
    matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, Patch

from config import PlateConfig, SimulationConfig
from geometry import build_mesh
from solver import solve
from field import (
    compute_potential_field_distributed,
    make_grid_xz,
)
from presets import PRESETS
from utils import EPSILON_0
from validation import validate_config, ValidationError

N_GRID = 35
CMAP_PHI = 'RdBu_r'
COLOR_P1 = '#1565C0'
COLOR_P2 = '#B71C1C'
LENGTH_RANGE = (0.020, 0.30)
WIDTH_RANGE = (0.020, 0.30)
THICKNESS_RANGE = (0.001, 0.02)
GAP_RANGE = (0.003, 0.12)
VOLTAGE_RANGE = (1.0, 500.0)
N_ELEMENTS_RANGE = (3, 50)
EPSILON_RANGE = (1.0, 10.0)
Y_SLICE_RANGE = (-0.15, 0.15)

def _plate_mask(pts, cfg):
    mask = np.zeros(len(pts), dtype=bool)
    p2, z2 = cfg.plate2, cfg.plate2_z_center
    mask |= ((np.abs(pts[:, 0]) <= p2.length / 2) &
             (np.abs(pts[:, 1]) <= p2.width / 2) &
             (np.abs(pts[:, 2] - z2) <= p2.thickness / 2))
    p1, z1 = cfg.plate1, cfg.plate1_z_center
    mask |= ((np.abs(pts[:, 0]) <= p1.length / 2) &
             (np.abs(pts[:, 1]) <= p1.width / 2) &
             (np.abs(pts[:, 2] - z1) <= p1.thickness / 2))
    return mask


def _plate_visible(plate, z_center, y_slice):
    return abs(y_slice) <= plate.width / 2


def _wrap_lines(lines, width):
    wrapped = []
    for line in lines:
        if not line:
            wrapped.append('')
            continue
        wrapped.extend(textwrap.wrap(line, width=width, break_long_words=False, break_on_hyphens=False))
    return '\n'.join(wrapped)


def _require_range(name, value, bounds):
    lo, hi = bounds
    if value < lo or value > hi:
        raise ValidationError(f"{name} должно быть в диапазоне [{lo}, {hi}], получено {value}")


def _validate_app_limits(cfg, y_slice):
    _require_range('L1', cfg.plate1.length, LENGTH_RANGE)
    _require_range('W1', cfg.plate1.width, WIDTH_RANGE)
    _require_range('T1', cfg.plate1.thickness, THICKNESS_RANGE)
    _require_range('L2', cfg.plate2.length, LENGTH_RANGE)
    _require_range('W2', cfg.plate2.width, WIDTH_RANGE)
    _require_range('T2', cfg.plate2.thickness, THICKNESS_RANGE)
    _require_range('gap', cfg.gap, GAP_RANGE)
    _require_range('|V|', abs(cfg.voltage), VOLTAGE_RANGE)
    _require_range('n_elements', cfg.n_elements, N_ELEMENTS_RANGE)
    _require_range('epsilon_r', cfg.epsilon_r, EPSILON_RANGE)
    _require_range('y-slice', y_slice, Y_SLICE_RANGE)


def _clone_config(cfg):
    return SimulationConfig(
        plate1=PlateConfig(
            length=cfg.plate1.length,
            width=cfg.plate1.width,
            thickness=cfg.plate1.thickness,
            potential=cfg.plate1.potential,
        ),
        plate2=PlateConfig(
            length=cfg.plate2.length,
            width=cfg.plate2.width,
            thickness=cfg.plate2.thickness,
            potential=cfg.plate2.potential,
        ),
        gap=cfg.gap,
        epsilon_r=cfg.epsilon_r,
        n_elements=cfg.n_elements,
    )


def _config_from_args(args):
    cfg = _clone_config(PRESETS[args.preset])
    if args.l1 is not None:
        cfg.plate1.length = args.l1
    if args.w1 is not None:
        cfg.plate1.width = args.w1
    if args.t1 is not None:
        cfg.plate1.thickness = args.t1
    if args.l2 is not None:
        cfg.plate2.length = args.l2
    if args.w2 is not None:
        cfg.plate2.width = args.w2
    if args.t2 is not None:
        cfg.plate2.thickness = args.t2
    if args.gap is not None:
        cfg.gap = args.gap
    if args.eps is not None:
        cfg.epsilon_r = args.eps
    if args.n is not None:
        cfg.n_elements = args.n
    if args.voltage is not None:
        cfg.plate1.potential = args.voltage / 2
        cfg.plate2.potential = -args.voltage / 2
    return cfg


def run_simulation(cfg):
    all_warns = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        validate_config(cfg)
        all_warns += [str(x.message) for x in w]
    mesh = build_mesh(cfg)
    result = solve(cfg, mesh)
    return mesh, result, all_warns


def build_summary(cfg, mesh, result, warns):
    s = min(cfg.plate1.length * cfg.plate1.width, cfg.plate2.length * cfg.plate2.width)
    c_ideal = EPSILON_0 * cfg.epsilon_r * s / cfg.gap
    return {
        'voltage': cfg.voltage,
        'gap': cfg.gap,
        'epsilon_r': cfg.epsilon_r,
        'n_elements': cfg.n_elements,
        'n_total': int(len(mesh.centers)),
        'capacitance_f': float(result.capacitance),
        'capacitance_pf': float(result.capacitance * 1e12),
        'c_ideal_f': float(c_ideal),
        'c_ideal_pf': float(c_ideal * 1e12),
        'capacitance_ratio': float(result.capacitance / c_ideal),
        'q1': float(result.Q_plate1),
        'q2': float(result.Q_plate2),
        'residual_norm': float(result.residual_norm),
        'condition_number': float(result.condition_number),
        'warnings': warns,
    }


def print_summary(summary, as_json=False):
    if as_json:
        print(json.dumps(summary, ensure_ascii=False))
        return
    print(f"C = {summary['capacitance_pf']:.3f} пФ")
    print(f"C_ideal = {summary['c_ideal_pf']:.3f} пФ")
    print(f"C/C_ideal = {summary['capacitance_ratio']:.3f}")
    print(f"Q1 = {summary['q1']:.3e} Кл")
    print(f"Q2 = {summary['q2']:.3e} Кл")
    print(f"N = {summary['n_total']}")
    if summary['warnings']:
        for msg in summary['warnings']:
            print(f"! {msg}")


def save_snapshot(cfg, output_path, y_slice=0.0):
    app = App(start_config=cfg)
    if abs(float(app._sl_y.val) - y_slice) > 1e-12:
        app._sl_y.set_val(y_slice)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    app._fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(app._fig)

class App:
    def __init__(self, start_preset='fringe', start_config=None):
        self._cfg = _clone_config(start_config if start_config is not None else PRESETS[start_preset])
        self._mesh = None
        self._result = None
        self._cb2d = None

        self._build_figure()
        self._build_sliders()
        self._build_controls()
        self._recompute()

    def _build_figure(self):
        self._fig = plt.figure(figsize=(14, 9))
        self._fig.suptitle('Метод моментов — плоский конденсатор', fontsize=13)
        self._plot_rect = [0.05, 0.44, 0.76, 0.50]
        self._cbar_rect = [0.83, 0.44, 0.025, 0.50]
        self._slider_cols = [0.05, 0.28, 0.51]
        self._slider_rows = [0.33, 0.255, 0.180, 0.105]
        self._slider_width = 0.19
        self._slider_height = 0.028
        self._button_rect = [0.77, 0.31, 0.16, 0.045]
        self._text_rect = [0.73, 0.05, 0.24, 0.22]
        self._ax2d = self._fig.add_axes(self._plot_rect)
        self._ax2d_cbar = self._fig.add_axes(self._cbar_rect)
        self._ax2d.set_navigate(False)
        self._ax2d_cbar.set_navigate(False)
    def _build_sliders(self):
        cfg = self._cfg
        sh = self._slider_height
        sw = self._slider_width
        cx = self._slider_cols
        ry = self._slider_rows

        def sl(col, row, label, lo, hi, val, step=None, fmt='%.4g'):
            ax = self._fig.add_axes([cx[col], ry[row], sw, sh])
            s = mwidgets.Slider(ax, '', lo, hi, valinit=val, valstep=step)
            s.valtext.set_text(fmt % val)
            ax.set_title(label, fontsize=8.5, pad=2, loc='left', color='#222222')
            return s

        V0 = abs(cfg.voltage)
        hkw = dict(fontsize=9, fontweight='bold', va='bottom', transform=self._fig.transFigure)
        hy = ry[0] + sh + 0.015
        self._fig.text(cx[0] + sw / 2, hy, 'Пластина 1', ha='center', color=COLOR_P1, **hkw)
        self._fig.text(cx[1] + sw / 2, hy, 'Пластина 2', ha='center', color=COLOR_P2, **hkw)
        self._fig.text(cx[2] + sw / 2, hy, 'Система', ha='center', color='#444444', **hkw)
        self._sl_L1 = sl(0, 0, 'L₁, м', LENGTH_RANGE[0], LENGTH_RANGE[1], cfg.plate1.length)
        self._sl_W1 = sl(0, 1, 'W₁, м', WIDTH_RANGE[0], WIDTH_RANGE[1], cfg.plate1.width)
        self._sl_T1 = sl(0, 2, 'T₁, м', THICKNESS_RANGE[0], THICKNESS_RANGE[1], cfg.plate1.thickness)
        self._sl_y = sl(0, 3, 'Y разрез, м', Y_SLICE_RANGE[0], Y_SLICE_RANGE[1], 0.0, fmt='%.3f')
        self._sl_L2 = sl(1, 0, 'L₂, м', LENGTH_RANGE[0], LENGTH_RANGE[1], cfg.plate2.length)
        self._sl_W2 = sl(1, 1, 'W₂, м', WIDTH_RANGE[0], WIDTH_RANGE[1], cfg.plate2.width)
        self._sl_T2 = sl(1, 2, 'T₂, м', THICKNESS_RANGE[0], THICKNESS_RANGE[1], cfg.plate2.thickness)
        self._sl_gap = sl(2, 0, 'Зазор, м', GAP_RANGE[0], GAP_RANGE[1], cfg.gap)
        self._sl_V = sl(2, 1, 'V, В', VOLTAGE_RANGE[0], VOLTAGE_RANGE[1], V0, fmt='%.1f')
        self._sl_n = sl(2, 2, 'N эл.', N_ELEMENTS_RANGE[0], N_ELEMENTS_RANGE[1], cfg.n_elements, step=1, fmt='%d')
        self._sl_eps = sl(2, 3, 'ε_r', EPSILON_RANGE[0], EPSILON_RANGE[1], cfg.epsilon_r, fmt='%.2f')

    def _build_controls(self):
        ax_btn = self._fig.add_axes(self._button_rect)
        self._btn = mwidgets.Button(ax_btn, 'Пересчитать', color='#E3F2FD', hovercolor='#90CAF9')
        self._btn.on_clicked(lambda _: self._recompute())
        ax_txt = self._fig.add_axes(self._text_rect)
        ax_txt.axis('off')
        self._ax_txt = ax_txt
        self._txt = ax_txt.text(0.0, 1.0, 'Нажмите Пересчитать', transform=ax_txt.transAxes,
                                 fontsize=8.0, va='top', ha='left', family='monospace', wrap=True)

    def _get_config(self):
        V = float(self._sl_V.val)
        return SimulationConfig(
            plate1=PlateConfig(
                length=float(self._sl_L1.val),
                width=float(self._sl_W1.val),
                thickness=float(self._sl_T1.val),
                potential=+V / 2,
            ),
            plate2=PlateConfig(
                length=float(self._sl_L2.val),
                width=float(self._sl_W2.val),
                thickness=float(self._sl_T2.val),
                potential=-V / 2,
            ),
            gap=float(self._sl_gap.val),
            epsilon_r=float(self._sl_eps.val),
            n_elements=int(self._sl_n.val),
        )

    def _recompute(self):
        cfg = self._get_config()
        y_slice = float(self._sl_y.val)
        self._fig.suptitle('Считаю...', fontsize=13, color='gray')
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

        try:
            _validate_app_limits(cfg, y_slice)
            mesh, result, all_warns = run_simulation(cfg)
        except ValidationError as e:
            return self._show_error(f'Ошибка входных данных:\n{e}')
        except Exception as e:
            return self._show_error(f'Ошибка расчёта:\n{e}')

        self._cfg = cfg
        self._mesh = mesh
        self._result = result

        X, Z, pts = make_grid_xz(cfg, mesh, n_points=N_GRID, y_slice=y_slice)
        phi_flat, E_flat = compute_potential_field_distributed(pts, mesh, result, cfg.epsilon_r)
        phi = phi_flat.reshape(N_GRID, N_GRID)
        Ex = E_flat[:, 0].reshape(N_GRID, N_GRID)
        Ez = E_flat[:, 2].reshape(N_GRID, N_GRID)
        inside = _plate_mask(pts, cfg).reshape(N_GRID, N_GRID)
        Ex[inside] = np.nan
        Ez[inside] = np.nan

        self._draw_2d(cfg, X, Z, phi, Ex, Ez, inside, y_slice)
        self._update_info(cfg, mesh, result, all_warns)
        self._fig.suptitle('Метод моментов — плоский конденсатор', fontsize=13, color='black')
        self._fig.canvas.draw_idle()

    def _draw_2d(self, cfg, X, Z, phi, Ex, Ez, inside, y_slice=0.0):
        ax = self._ax2d
        ax.clear()
        x_1d = X[0, :]
        z_1d = Z[:, 0]

        phi_valid = phi[np.isfinite(phi) & ~inside]
        if len(phi_valid) == 0:
            phi_valid = phi[np.isfinite(phi)]
        if len(phi_valid) == 0:
            ax.set_title('Нет данных для отображения', fontsize=10)
            return

        vmax_phi = np.abs(phi_valid).max()
        levels = np.linspace(-vmax_phi, vmax_phi, 22)
        norm = Normalize(vmin=-vmax_phi, vmax=vmax_phi)

        cf = ax.contourf(X, Z, phi, levels=levels, cmap=CMAP_PHI, norm=norm, extend='both')
        self._ax2d.set_position(self._plot_rect)
        self._ax2d_cbar.set_position(self._cbar_rect)
        sm = ScalarMappable(norm=norm, cmap=CMAP_PHI)
        sm.set_array([])
        if self._cb2d is None:
            self._ax2d_cbar.clear()
            self._cb2d = self._fig.colorbar(sm, cax=self._ax2d_cbar)
            self._cb2d.set_label('φ, В')
        else:
            self._cb2d.update_normal(sm)
            self._cb2d.set_label('φ, В')
        self._cb2d.ax.set_position(self._cbar_rect)

        Ex_s = np.ma.masked_invalid(Ex)
        Ez_s = np.ma.masked_invalid(Ez)
        E_mag = np.ma.sqrt(Ex_s**2 + Ez_s**2)
        safe = np.ma.where(E_mag > 0, E_mag, 1.0)
        try:
            ax.streamplot(x_1d, z_1d, Ex_s / safe, Ez_s / safe, color='white', linewidth=0.9,
                          density=1.3, arrowsize=0.9, arrowstyle='->')
        except Exception:
            pass

        p2, z2 = cfg.plate2, cfg.plate2_z_center
        if _plate_visible(p2, z2, y_slice):
            ax.add_patch(Rectangle(
                (-p2.length / 2, z2 - p2.thickness / 2),
                p2.length, p2.thickness,
                linewidth=1.5, edgecolor=COLOR_P2, facecolor=COLOR_P2, alpha=1.0,
            ))

        p1, z1 = cfg.plate1, cfg.plate1_z_center
        if _plate_visible(p1, z1, y_slice):
            ax.add_patch(Rectangle(
                (-p1.length / 2, z1 - p1.thickness / 2),
                p1.length, p1.thickness,
                linewidth=1.5, edgecolor=COLOR_P1, facecolor=COLOR_P1, alpha=1.0,
            ))

        ax.set_xlabel('X, м')
        ax.set_ylabel('Z, м')
        ax.set_title(f'Потенциал φ и силовые линии E  (разрез Y = {y_slice*100:.1f} см)', fontsize=10)
        ax.set_aspect('auto')

        legend_elems = [
            Patch(facecolor=COLOR_P1, alpha=0.7, label=f'Пл. 1  φ = {cfg.plate1.potential:+.0f} В'),
            Patch(facecolor=COLOR_P2, alpha=0.7, label=f'Пл. 2  φ = {cfg.plate2.potential:+.0f} В'),
        ]
        ax.legend(handles=legend_elems, loc='upper right', fontsize=8)

    def _update_info(self, cfg, mesh, result, warns):
        S = min(cfg.plate1.length * cfg.plate1.width, cfg.plate2.length * cfg.plate2.width)
        C_ideal = EPSILON_0 * cfg.epsilon_r * S / cfg.gap

        lines = [
            f'C         = {result.capacitance*1e12:.3f} пФ',
            f'C_ideal   = {C_ideal*1e12:.3f} пФ',
            f'C/C_ideal = {result.capacitance/C_ideal:.3f}',
            f'Q₁        = {result.Q_plate1:.3e} Кл',
            f'N элем.   = {len(mesh.centers)}',
        ]
        if warns:
            lines.append('')
            for w in warns[:3]:
                lines.append('! ' + w)

        self._txt.set_text(_wrap_lines(lines, width=34))
        self._txt.set_color('black')

    def _show_error(self, msg):
        self._ax2d.clear()
        wrapped = _wrap_lines(msg.splitlines(), width=70)
        self._ax2d.text(0.5, 0.5, wrapped, transform=self._ax2d.transAxes,
                         ha='center', va='center', fontsize=9, color='red', wrap=True)
        self._txt.set_text(_wrap_lines(msg.splitlines(), width=34))
        self._txt.set_color('red')
        self._fig.suptitle('Ошибка', fontsize=13, color='red')
        self._fig.canvas.draw_idle()

def main():
    parser = argparse.ArgumentParser(description='Плоский конденсатор — метод моментов')
    parser.add_argument('--preset', default='fringe', choices=list(PRESETS.keys()),
                        help='Начальный пресет (по умолчанию: fringe)')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--l1', type=float)
    parser.add_argument('--w1', type=float)
    parser.add_argument('--t1', type=float)
    parser.add_argument('--l2', type=float)
    parser.add_argument('--w2', type=float)
    parser.add_argument('--t2', type=float)
    parser.add_argument('--gap', type=float)
    parser.add_argument('--voltage', type=float)
    parser.add_argument('--n', type=int)
    parser.add_argument('--eps', type=float)
    parser.add_argument('--y-slice', type=float, default=0.0)
    parser.add_argument('--save-image')
    args = parser.parse_args()
    cfg = _config_from_args(args)
    if args.headless:
        try:
            _validate_app_limits(cfg, args.y_slice)
            mesh, result, warns = run_simulation(cfg)
        except ValidationError as e:
            print(str(e))
            raise SystemExit(2)
        summary = build_summary(cfg, mesh, result, warns)
        if args.save_image:
            save_snapshot(cfg, args.save_image, y_slice=args.y_slice)
            summary['image_path'] = str(Path(args.save_image).resolve())
        print_summary(summary, as_json=args.json)
        return
    app = App(start_preset=args.preset, start_config=cfg)
    plt.show()

if __name__ == '__main__':
    main()
