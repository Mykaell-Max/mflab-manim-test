"""Reconstrução AMR refinada e visualização 3D de saídas do MFSim.

Fluxo:
  1. Lê todos os níveis Chombo/MFSim do HDF5.
  2. Reamostra-os em uma grade uniforme, do nível grosseiro ao fino.
  3. O nível mais refinado sempre sobrescreve os anteriores.
  4. Salva um cache VTI por timestep.
  5. Reproduz corte, streamlines e isosuperfície da esteira em PyVista.

O cache torna a primeira execução mais demorada, mas as seguintes muito
mais rápidas. A reamostragem é explícita e o espaçamento final permanece
registrado no VTI.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates


# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------

OUTPUT_DIR = Path(r"C:\Users\max00\Downloads\output\output")

# 0.004: recomendado para validar (~4,3 milhões de pontos/frame).
# 0.002: preserva o dx mais fino no domínio inteiro (~34 milhões/frame).
TARGET_DX = 0.004

CACHE_DIR_NAME = "mflab_amr_cache"
CACHE_FLOAT_TYPE = np.float32
COMPONENTS_TO_CACHE = ("u", "v", "w", "pressure", "dwall_s")

FRAME_DURATION_MS = 250
LOOP_COUNT = 100000
WAKE_SPEED = 0.80
STREAMLINE_SEEDS_Y = 5
STREAMLINE_SEEDS_Z = 5

BACKGROUND = "#07131F"
TEXT_COLOR = "#F1F6F9"
GEOMETRY_COLOR = "#D7E4EC"
STREAMLINE_COLOR = "#76D7FF"
WAKE_COLOR = "#FFB454"
COLORMAP = "viridis"


# -----------------------------------------------------------------------------
# Leitura Chombo/MFSim
# -----------------------------------------------------------------------------

def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def timestep_number(path: Path) -> int:
    return int(path.name.split(".")[-2])


def discover_hdf5_files() -> list[Path]:
    files = sorted(
        OUTPUT_DIR.glob("ns_output_ct.*.hdf5"),
        key=timestep_number,
    )
    if not files:
        raise FileNotFoundError(
            f"Nenhum ns_output_ct.*.hdf5 encontrado em {OUTPUT_DIR}"
        )
    return files


def target_geometry(hdf: h5py.File):
    root = hdf["level_0"]
    domain_attribute = root.attrs["prob_domain"]

    # Dependendo da versão do h5py/HDF5, prob_domain pode chegar como
    # um vetor de seis inteiros ou como um escalar de dtype estruturado.
    field_names = getattr(domain_attribute.dtype, "names", None)
    if field_names:
        lo = np.array(
            [
                domain_attribute["lo_i"],
                domain_attribute["lo_j"],
                domain_attribute["lo_k"],
            ],
            dtype=int,
        )
        hi = np.array(
            [
                domain_attribute["hi_i"],
                domain_attribute["hi_j"],
                domain_attribute["hi_k"],
            ],
            dtype=int,
        )
    else:
        domain = np.asarray(domain_attribute, dtype=int).reshape(-1)
        if domain.size != 6:
            raise ValueError(
                "prob_domain deveria conter seis índices, mas contém "
                f"{domain.size}: {domain_attribute!r}"
            )
        lo = domain[:3]
        hi = domain[3:]

    coarse_dx = np.asarray(root.attrs["vec_dx"], dtype=float)
    origin = np.asarray(hdf.attrs["origin"], dtype=float)
    physical_min = origin + lo * coarse_dx
    physical_max = origin + (hi + 1) * coarse_dx

    dimensions = np.rint(
        (physical_max - physical_min) / TARGET_DX
    ).astype(int) + 1

    reconstructed_max = physical_min + (dimensions - 1) * TARGET_DX
    if not np.allclose(reconstructed_max, physical_max, atol=1.0e-10):
        raise ValueError(
            "TARGET_DX não divide exatamente o domínio. "
            f"Máximo esperado {physical_max}, obtido {reconstructed_max}."
        )

    return physical_min, physical_max, dimensions


def decode_level(hdf: h5py.File, level_index: int):
    group = hdf[f"level_{level_index}"]
    boxes = group["boxes"][:]
    raw = group["data:datatype=0"][:]
    spacing = np.asarray(group.attrs["vec_dx"], dtype=float)
    global_origin = np.asarray(hdf.attrs["origin"], dtype=float)

    component_count = int(group.attrs["num_components"])
    component_names = [
        decode_name(group.attrs[f"component_{index}"])
        for index in range(component_count)
    ]
    component_indices = {
        name: component_names.index(name)
        for name in COMPONENTS_TO_CACHE
    }

    cursor = 0
    decoded_blocks = []

    for box_index, box in enumerate(boxes):
        lo = np.array(
            [box["lo_i"], box["lo_j"], box["lo_k"]], dtype=int
        )
        hi = np.array(
            [box["hi_i"], box["hi_j"], box["hi_k"]], dtype=int
        )

        point_dimensions = hi - lo + 2
        points_per_component = int(np.prod(point_dimensions))
        record_size = points_per_component * component_count
        record = raw[cursor : cursor + record_size]

        if record.size != record_size:
            raise ValueError(
                f"Nível {level_index}, bloco {box_index}: "
                f"esperados {record_size} valores, encontrados {record.size}."
            )
        cursor += record_size

        all_components = record.reshape(
            (component_count, points_per_component)
        )
        arrays = {
            name: all_components[index].reshape(
                tuple(point_dimensions), order="F"
            )
            for name, index in component_indices.items()
        }

        decoded_blocks.append(
            {
                "origin": global_origin + lo * spacing,
                "spacing": spacing,
                "dimensions": point_dimensions,
                "arrays": arrays,
            }
        )

    if cursor != raw.size:
        raise ValueError(
            f"Nível {level_index}: consumidos {cursor} valores de "
            f"{raw.size}. O layout do arquivo não corresponde ao esperado."
        )

    return decoded_blocks


# -----------------------------------------------------------------------------
# Composição AMR em grade uniforme
# -----------------------------------------------------------------------------

def target_interval(block_min, block_max, target_origin, dimensions):
    start = np.ceil(
        (block_min - target_origin) / TARGET_DX - 1.0e-9
    ).astype(int)
    stop = np.floor(
        (block_max - target_origin) / TARGET_DX + 1.0e-9
    ).astype(int)

    start = np.maximum(start, 0)
    stop = np.minimum(stop, dimensions - 1)
    return start, stop


def interpolate_block(array, source_indices):
    coordinates = np.meshgrid(
        source_indices[0],
        source_indices[1],
        source_indices[2],
        indexing="ij",
        sparse=False,
    )
    return map_coordinates(
        array,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )


def compose_uniform_grid(hdf5_path: Path) -> pv.ImageData:
    with h5py.File(hdf5_path, "r") as hdf:
        physical_time = float(hdf.attrs["time"])
        target_origin, _, dimensions = target_geometry(hdf)
        number_of_levels = int(hdf.attrs["num_levels"])

        arrays = {
            name: np.full(
                tuple(dimensions),
                np.nan,
                dtype=CACHE_FLOAT_TYPE,
            )
            for name in COMPONENTS_TO_CACHE
        }
        refinement_level = np.full(
            tuple(dimensions), -1, dtype=np.int8
        )

        # A ordem é essencial: blocos finos sobrescrevem os grosseiros.
        for level_index in range(number_of_levels):
            blocks = decode_level(hdf, level_index)
            print(
                f"    nível {level_index}: {len(blocks)} blocos",
                flush=True,
            )

            for block in blocks:
                block_min = block["origin"]
                block_max = (
                    block["origin"]
                    + (block["dimensions"] - 1) * block["spacing"]
                )
                start, stop = target_interval(
                    block_min,
                    block_max,
                    target_origin,
                    dimensions,
                )

                if np.any(stop < start):
                    continue

                target_indices = [
                    np.arange(start[axis], stop[axis] + 1)
                    for axis in range(3)
                ]
                target_coordinates = [
                    target_origin[axis]
                    + target_indices[axis] * TARGET_DX
                    for axis in range(3)
                ]
                source_indices = [
                    (target_coordinates[axis] - block["origin"][axis])
                    / block["spacing"][axis]
                    for axis in range(3)
                ]

                destination = np.ix_(
                    target_indices[0],
                    target_indices[1],
                    target_indices[2],
                )

                for name in COMPONENTS_TO_CACHE:
                    source = block["arrays"][name]
                    if name == "dwall_s":
                        # 1e40 é sentinela, não uma distância física.
                        source = np.where(
                            np.abs(source) < 1.0e20, source, np.nan
                        )

                    sampled = interpolate_block(source, source_indices)
                    arrays[name][destination] = sampled.astype(
                        CACHE_FLOAT_TYPE, copy=False
                    )

                refinement_level[destination] = level_index

    missing_velocity = sum(
        int(np.isnan(arrays[name]).sum()) for name in ("u", "v", "w")
    )
    if missing_velocity:
        raise ValueError(
            f"A reconstrução deixou {missing_velocity} valores de velocidade "
            "sem cobertura."
        )

    velocity = np.stack(
        (arrays["u"], arrays["v"], arrays["w"]), axis=-1
    )
    speed = np.linalg.norm(velocity, axis=-1).astype(CACHE_FLOAT_TYPE)

    grid = pv.ImageData(
        dimensions=tuple(int(value) for value in dimensions),
        spacing=(TARGET_DX, TARGET_DX, TARGET_DX),
        origin=tuple(float(value) for value in target_origin),
    )
    grid.point_data["velocity"] = velocity.reshape((-1, 3), order="F")
    grid.point_data["velocity_magnitude"] = speed.ravel(order="F")
    grid.point_data["pressure"] = arrays["pressure"].ravel(order="F")
    grid.point_data["dwall_s"] = arrays["dwall_s"].ravel(order="F")
    grid.point_data["amr_level"] = refinement_level.ravel(order="F")
    grid.field_data["physical_time"] = np.array([physical_time])
    grid.field_data["target_dx"] = np.array([TARGET_DX])

    return grid


def cache_directory() -> Path:
    dx_label = str(TARGET_DX).replace(".", "p")
    return OUTPUT_DIR / f"{CACHE_DIR_NAME}_dx_{dx_label}"


def cache_path_for(hdf5_path: Path) -> Path:
    return cache_directory() / f"ct.{timestep_number(hdf5_path):09d}.vti"


def preprocess(files: list[Path], force: bool = False):
    destination = cache_directory()
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Cache: {destination}")
    for index, path in enumerate(files, start=1):
        cached = cache_path_for(path)
        if cached.exists() and not force:
            print(f"[{index:02d}/{len(files):02d}] reutilizando {cached.name}")
            continue

        print(f"[{index:02d}/{len(files):02d}] reconstruindo {path.name}")
        start = time.perf_counter()
        grid = compose_uniform_grid(path)
        grid.save(cached, binary=True)
        elapsed = time.perf_counter() - start

        points = grid.n_points
        print(
            f"    {points:,} pontos • {elapsed:.1f} s • "
            f"salvo em {cached.name}"
        )


# -----------------------------------------------------------------------------
# Visualização
# -----------------------------------------------------------------------------

def discover_cache_files() -> list[Path]:
    files = sorted(cache_directory().glob("ct.*.vti"))
    if not files:
        raise FileNotFoundError(
            "Nenhum cache VTI encontrado. Execute primeiro com --preprocess."
        )
    return files


def physical_time(grid) -> float:
    return float(np.asarray(grid.field_data["physical_time"])[0])


def global_speed_range(cache_files: list[Path]):
    minimum = np.inf
    maximum = -np.inf
    for path in cache_files:
        grid = pv.read(path)
        values = np.asarray(grid.point_data["velocity_magnitude"])
        finite = values[np.isfinite(values) & (np.abs(values) < 1.0e20)]
        minimum = min(minimum, float(finite.min()))
        maximum = max(maximum, float(finite.max()))
    return minimum, maximum


def central_slice(grid):
    return grid.slice(normal=(0, 1, 0), origin=grid.center)


def sphere_surface(grid):
    values = np.asarray(grid.point_data["dwall_s"])
    finite = values[np.isfinite(values)]
    if not finite.size or not (finite.min() <= 0 <= finite.max()):
        return None
    surface = grid.contour([0.0], scalars="dwall_s")
    return surface.clean() if surface.n_points else None


def wake_surface(grid):
    values = np.asarray(grid.point_data["velocity_magnitude"])
    if values.min() > WAKE_SPEED or values.max() < WAKE_SPEED:
        return None
    result = grid.contour([WAKE_SPEED], scalars="velocity_magnitude")
    return result if result.n_points else None


def streamlines(grid):
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds
    x_length = xmax - xmin
    y_margin = 0.25 * (ymax - ymin)
    z_margin = 0.25 * (zmax - zmin)

    seeds = pv.PointSet(
        np.array(
            [
                [xmin + 2 * TARGET_DX, y, z]
                for y in np.linspace(
                    ymin + y_margin,
                    ymax - y_margin,
                    STREAMLINE_SEEDS_Y,
                )
                for z in np.linspace(
                    zmin + z_margin,
                    zmax - z_margin,
                    STREAMLINE_SEEDS_Z,
                )
            ]
        )
    )
    return grid.streamlines_from_source(
        seeds,
        vectors="velocity",
        integration_direction="forward",
        integrator_type=45,
        max_length=x_length * 1.10,
        initial_step_length=TARGET_DX * 0.5,
        terminal_speed=1.0e-8,
        compute_vorticity=False,
    )


def view(cache_files: list[Path]):
    speed_range = global_speed_range(cache_files)
    print(f"Escala global |u|: {speed_range[0]:.6f} a {speed_range[1]:.6f}")

    initial = pv.read(cache_files[0])
    reference = pv.read(cache_files[len(cache_files) // 2])

    plotter = pv.Plotter(window_size=(1600, 900))
    plotter.set_background(BACKGROUND)
    plotter.enable_parallel_projection()

    lookup_table = pv.LookupTable()
    lookup_table.apply_cmap(COLORMAP, n_values=256)
    lookup_table.scalar_range = speed_range
    lookup_table.nan_opacity = 0.0

    def add_dynamic_actors(grid, render=False):
        section = central_slice(grid)
        section_actor = plotter.add_mesh(
            section,
            name="velocity_slice",
            scalars="velocity_magnitude",
            cmap=COLORMAP,
            clim=speed_range,
            opacity=0.82,
            lighting=False,
            show_scalar_bar=False,
            reset_camera=False,
            render=render,
        )
        section_actor.mapper.lookup_table = lookup_table
        section_actor.mapper.scalar_range = speed_range

        lines = streamlines(grid)
        if lines.n_points:
            plotter.add_mesh(
                lines,
                name="streamlines",
                color=STREAMLINE_COLOR,
                opacity=0.60,
                line_width=2,
                render_lines_as_tubes=False,
                lighting=False,
                show_scalar_bar=False,
                reset_camera=False,
                render=render,
            )

        wake = wake_surface(grid)
        if wake is not None:
            plotter.add_mesh(
                wake,
                name="wake",
                color=WAKE_COLOR,
                opacity=0.28,
                smooth_shading=True,
                show_scalar_bar=False,
                reset_camera=False,
                render=render,
            )
        else:
            plotter.remove_actor("wake", reset_camera=False, render=False)

        return section_actor

    section_actor = add_dynamic_actors(initial)

    sphere = sphere_surface(reference)
    if sphere is not None:
        plotter.add_mesh(
            sphere,
            name="sphere",
            color=GEOMETRY_COLOR,
            smooth_shading=True,
            metallic=0.10,
            roughness=0.40,
        )

    plotter.add_scalar_bar(
        title="|u|",
        mapper=section_actor.mapper,
        n_labels=5,
        fmt="%.4f",
        color=TEXT_COLOR,
        position_x=0.86,
        position_y=0.18,
        width=0.055,
        height=0.58,
        vertical=True,
    )
    plotter.add_text(
        "MFLab • Reconstrução AMR refinada",
        position="upper_left",
        font_size=15,
        color=TEXT_COLOR,
    )
    time_actor = plotter.add_text(
        f"t = {physical_time(initial):.6f}",
        position=(20, 20),
        font_size=12,
        color=TEXT_COLOR,
    )
    plotter.add_text(
        f"dx = {TARGET_DX:g} • esteira: |u| = {WAKE_SPEED:.2f}",
        position="lower_right",
        font_size=10,
        color=TEXT_COLOR,
    )
    plotter.add_axes(
        xlabel="x", ylabel="y", zlabel="z", color=TEXT_COLOR
    )

    center = np.asarray(initial.center)
    length = initial.length
    plotter.camera_position = [
        center + np.array([0.08, -0.95, 0.20]) * length,
        center,
        (0, 0, 1),
    ]
    plotter.camera.zoom(1.25)

    plotter.show(
        auto_close=False,
        interactive=True,
        interactive_update=True,
    )

    try:
        for _ in range(LOOP_COUNT):
            for index, path in enumerate(cache_files):
                grid = pv.read(path)
                add_dynamic_actors(grid, render=False)
                lookup_table.scalar_range = speed_range
                time_actor.SetInput(
                    f"t = {physical_time(grid):.6f}  •  "
                    f"frame {index + 1}/{len(cache_files)}"
                )
                plotter.update(force_redraw=True)
                time.sleep(FRAME_DURATION_MS / 1000.0)
    except (KeyboardInterrupt, RuntimeError):
        pass

    if plotter.render_window is not None:
        plotter.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstrói AMR do MFSim e visualiza o escoamento."
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="gera/reutiliza os caches VTI",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="abre a visualização usando os caches",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reconstrói caches existentes",
    )
    args = parser.parse_args()

    # Sem flags, executa o fluxo completo.
    run_preprocess = args.preprocess or not (args.preprocess or args.view)
    run_view = args.view or not (args.preprocess or args.view)

    if run_preprocess:
        preprocess(discover_hdf5_files(), force=args.force)

    if run_view:
        view(discover_cache_files())


if __name__ == "__main__":
    main()
