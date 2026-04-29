from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def gaussian_2d(
    xx: np.ndarray,
    yy: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    angle: float,
) -> np.ndarray:
    ca = np.cos(angle)
    sa = np.sin(angle)
    x = xx - cx
    y = yy - cy
    xr = ca * x + sa * y
    yr = -sa * x + ca * y
    return np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))


def make_rolling_hills_heightfield(
    rows: int,
    cols: int,
    seed: int,
    half_length: float,
    half_width: float,
    terrain_center_x: float,
    flat_until_x: float,
    hill_count: int,
    valley_count: int,
    undulation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    local_x = np.linspace(-half_length, half_length, cols, dtype=np.float32)
    world_x = local_x + terrain_center_x
    y = np.linspace(-half_width, half_width, rows, dtype=np.float32)
    xx, yy = np.meshgrid(world_x, y)

    height = np.zeros_like(xx, dtype=np.float32)

    # Large smooth hills.
    terrain_min_x = terrain_center_x - half_length
    terrain_max_x = flat_until_x - 0.25
    for _ in range(hill_count):
        cx = rng.uniform(terrain_min_x + 0.35, terrain_max_x)
        cy = rng.uniform(-half_width * 0.55, half_width * 0.55)
        sx = rng.uniform(0.22, 0.55)
        sy = rng.uniform(0.14, 0.36)
        angle = rng.uniform(-0.9, 0.9)
        amp = rng.uniform(0.45, 1.00)
        height += amp * gaussian_2d(xx, yy, cx, cy, sx, sy, angle)

    # Smooth basins carved into the hills, so the surface goes up and down.
    for _ in range(valley_count):
        cx = rng.uniform(terrain_min_x + 0.45, terrain_max_x)
        cy = rng.uniform(-half_width * 0.55, half_width * 0.55)
        sx = rng.uniform(0.25, 0.62)
        sy = rng.uniform(0.16, 0.42)
        angle = rng.uniform(-0.9, 0.9)
        amp = rng.uniform(0.22, 0.58)
        height -= amp * gaussian_2d(xx, yy, cx, cy, sx, sy, angle)

    # Broad non-linear terrain waves. These make saddles and shoulders instead of straight ramps.
    height += undulation * 0.35 * np.sin(2.0 * np.pi * (0.37 * xx + 0.72 * yy + 0.11))
    height += undulation * 0.24 * np.sin(2.0 * np.pi * (0.61 * xx - 0.34 * yy + 0.47))
    height += undulation * 0.18 * np.cos(2.0 * np.pi * (0.19 * xx + 1.13 * yy - 0.23))

    # Keep edges and reset zone smooth.
    entry_blend = smoothstep((flat_until_x - xx) / 0.48)
    side_blend = smoothstep((half_width - np.abs(yy)) / 0.13)
    height *= entry_blend * side_blend

    # Put the launch zone at zero height and keep valleys as low regions, not holes below the floor.
    height -= np.percentile(height, 4.0)
    height = np.maximum(height, 0.0)
    height *= entry_blend * side_blend
    height /= float(height.max() + 1e-9)
    return height.astype(np.float32), world_x, y


def write_xml(
    base_xml: str,
    out_xml: str,
    heightmap_path: str,
    half_length: float,
    half_width: float,
    height_scale: float,
    negative_height: float,
    terrain_center_x: float,
) -> None:
    with open(base_xml, "r", encoding="utf-8") as file:
        xml = file.read()

    heightmap_xml_path = heightmap_path.replace("\\", "/")
    asset_block = (
        '    <material name="rolling_hills_mat" rgba="0.23 0.31 0.20 1" reflectance="0"/>\n'
        f'    <hfield name="rolling_hills_hfield" file="{heightmap_xml_path}" '
        f'size="{half_length:.6g} {half_width:.6g} {height_scale:.6g} {negative_height:.6g}"/>\n'
    )
    xml = xml.replace("  </asset>", asset_block + "  </asset>", 1)

    old_ground = (
        '    <geom name="ground" type="plane" pos="0 0 0" size="9 9 0.1" '
        'material="ground_mat" friction="0.1 0.08 0.01" contype="2" conaffinity="1"/>'
    )
    new_ground = (
        '    <geom name="ground" type="plane" pos="0 0 -0.008" size="12 12 0.1" '
        'rgba="0.17 0.22 0.15 1" friction="0.16 0.09 0.02" contype="2" conaffinity="1"/>\n'
        "\n"
        "    <!-- Continuous rolling hills: one curved hfield surface, no obstacle geoms. -->\n"
        '    <geom name="rolling_hills_terrain"\n'
        '          type="hfield"\n'
        '          hfield="rolling_hills_hfield"\n'
        f'          pos="{terrain_center_x:.6g} 0 0"\n'
        '          material="rolling_hills_mat"\n'
        '          friction="0.20 0.11 0.025"\n'
        '          contype="2"\n'
        '          conaffinity="1"/>'
    )
    if old_ground not in xml:
        raise RuntimeError("Could not find the base ground geom to replace.")

    xml = xml.replace(old_ground, new_ground, 1)
    xml = xml.replace('<mujoco model="jansen_assembly">', '<mujoco model="jansen_rolling_hills">', 1)

    out_dir = os.path.dirname(out_xml)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_xml, "w", encoding="utf-8", newline="\n") as file:
        file.write(xml)


def write_preview(path: str, heightfield: np.ndarray, world_x: np.ndarray, y: np.ndarray, height_scale: float) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    height_m = heightfield * height_scale
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.0), dpi=170, height_ratios=[1.35, 1.0, 1.0])

    image = axes[0].imshow(
        height_m,
        origin="lower",
        aspect="auto",
        extent=(float(world_x.min()), float(world_x.max()), float(y.min()), float(y.max())),
        cmap="terrain",
    )
    axes[0].set_title("Rolling Hills Heightfield")
    axes[0].set_ylabel("y position")
    fig.colorbar(image, ax=axes[0], label="height (m)")

    center_row = height_m[height_m.shape[0] // 2]
    off_center_row = height_m[int(height_m.shape[0] * 0.68)]
    axes[1].plot(world_x, center_row, color="#355f32", linewidth=1.7)
    axes[1].set_ylabel("center height (m)")
    axes[1].grid(True, color="#cbd4c5", linewidth=0.6, alpha=0.7)

    axes[2].plot(world_x, off_center_row, color="#6d7333", linewidth=1.7)
    axes[2].set_xlabel("x position, robot trains toward negative x")
    axes[2].set_ylabel("off-center height (m)")
    axes[2].grid(True, color="#d0d2bd", linewidth=0.6, alpha=0.7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a continuous rolling-hills hfield terrain for Jansen.")
    parser.add_argument("--base-xml", default="jansen_assembly_red_articulated.xml")
    parser.add_argument("--out-xml", default="jansen_assembly_rolling_hills.xml")
    parser.add_argument("--out-png", default="terrain_assets/rolling_hills_heightmap.png")
    parser.add_argument("--preview", default="plots/rolling_hills_preview.png")
    parser.add_argument("--rows", type=int, default=180)
    parser.add_argument("--cols", type=int, default=640)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--half-length", type=float, default=2.15)
    parser.add_argument("--half-width", type=float, default=0.62)
    parser.add_argument("--height-scale", type=float, default=0.026)
    parser.add_argument("--negative-height", type=float, default=0.012)
    parser.add_argument("--terrain-center-x", type=float, default=-2.0)
    parser.add_argument("--flat-until-x", type=float, default=-0.35)
    parser.add_argument("--hills", type=int, default=9)
    parser.add_argument("--valleys", type=int, default=7)
    parser.add_argument("--undulation", type=float, default=0.85)
    args = parser.parse_args()

    heightfield, world_x, y = make_rolling_hills_heightfield(
        rows=args.rows,
        cols=args.cols,
        seed=args.seed,
        half_length=args.half_length,
        half_width=args.half_width,
        terrain_center_x=args.terrain_center_x,
        flat_until_x=args.flat_until_x,
        hill_count=args.hills,
        valley_count=args.valleys,
        undulation=args.undulation,
    )

    png_dir = os.path.dirname(args.out_png)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)
    imageio.imwrite(args.out_png, (heightfield * 255.0).astype(np.uint8))

    write_preview(args.preview, heightfield, world_x, y, args.height_scale)
    write_xml(
        base_xml=args.base_xml,
        out_xml=args.out_xml,
        heightmap_path=args.out_png,
        half_length=args.half_length,
        half_width=args.half_width,
        height_scale=args.height_scale,
        negative_height=args.negative_height,
        terrain_center_x=args.terrain_center_x,
    )

    print(f"Saved terrain XML: {args.out_xml}")
    print(f"Saved heightmap: {args.out_png}")
    print(f"Saved preview: {args.preview}")
    print(f"Height scale: {args.height_scale:.4f} m | hills: {args.hills} | valleys: {args.valleys}")


if __name__ == "__main__":
    main()
