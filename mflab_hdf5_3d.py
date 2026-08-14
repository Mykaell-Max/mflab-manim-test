from pathlib import Path
import time

import h5py
import numpy as np
import pyvista as pv


# Altere apenas esta pasta. Todos os HDF5 temporais serão encontrados.
OUTPUT_DIR = Path(r"C:\Users\max00\Downloads\output\output")

# True: abre uma janela interativa. False: salva somente uma imagem.
INTERACTIVE = True
SCREENSHOT = Path("mflab_cfd_3d.png")
FRAME_DURATION_MS = 350
LOOP_COUNT = 100000  # continua até a janela ser fechada

BACKGROUND = "#07131F"
GEOMETRY_COLOR = "#D7E4EC"
TEXT_COLOR = "#F1F6F9"
COLORMAP = "viridis"
SHOW_REFERENCE_SLICE = True
SHOW_STREAMLINES = True
SHOW_WAKE_ISOSURFACE = True
STREAMLINE_SEEDS_Y = 5
STREAMLINE_SEEDS_Z = 5
WAKE_SPEED = 0.80
SLICE_NORMAL = (0, 1, 0)
STREAMLINE_COLOR = "#78D7FF"
WAKE_COLOR = "#FFB454"


def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def read_level(hdf, level_index):
    """Reconstrói os blocos estruturados de um nível Chombo/MFSim."""
    group = hdf[f"level_{level_index}"]
    boxes = group["boxes"][:]
    raw = group["data:datatype=0"][:]

    component_count = int(group.attrs["num_components"])
    component_names = [
        decode_name(group.attrs[f"component_{i}"])
        for i in range(component_count)
    ]

    spacing = tuple(float(v) for v in group.attrs["vec_dx"])
    global_origin = tuple(float(v) for v in hdf.attrs["origin"])

    blocks = pv.MultiBlock()
    cursor = 0

    for box_index, box in enumerate(boxes):
        lo = np.array([box["lo_i"], box["lo_j"], box["lo_k"]], dtype=int)
        hi = np.array([box["hi_i"], box["hi_j"], box["hi_k"]], dtype=int)

        cell_dimensions = hi - lo + 1
        point_dimensions = cell_dimensions + 1
        points_per_component = int(np.prod(point_dimensions))
        record_size = points_per_component * component_count

        record = raw[cursor : cursor + record_size]
        if record.size != record_size:
            raise ValueError(
                f"Nível {level_index}, bloco {box_index}: dados insuficientes. "
                f"Esperados {record_size}, encontrados {record.size}."
            )
        cursor += record_size

        block_origin = tuple(
            global_origin[axis] + lo[axis] * spacing[axis]
            for axis in range(3)
        )

        grid = pv.ImageData(
            dimensions=tuple(int(v) for v in point_dimensions),
            spacing=spacing,
            origin=block_origin,
        )

        # FArrayBox/Chombo: componentes contíguas, com i variando mais rápido.
        component_data = record.reshape(
            (component_count, points_per_component)
        )

        for component_index, component_name in enumerate(component_names):
            grid.point_data[component_name] = component_data[component_index]

        velocity = np.column_stack(
            (
                grid.point_data["u"],
                grid.point_data["v"],
                grid.point_data["w"],
            )
        )
        grid.point_data["velocity"] = velocity
        grid.point_data["velocity_magnitude"] = np.linalg.norm(
            velocity, axis=1
        )
        grid.field_data["amr_level"] = np.array([level_index])

        blocks[f"level_{level_index}_box_{box_index}"] = grid

    if cursor != raw.size:
        raise ValueError(
            f"Nível {level_index}: foram consumidos {cursor} valores, "
            f"mas o dataset contém {raw.size}."
        )

    return blocks


def collect_all_levels(hdf):
    levels = []
    number_of_levels = int(hdf.attrs["num_levels"])

    for level_index in range(number_of_levels):
        blocks = read_level(hdf, level_index)
        levels.append(blocks)
        print(f"Nível {level_index}: {len(blocks)} blocos reconstruídos")

    return levels


def finite_range(levels, scalar_name):
    minimum = np.inf
    maximum = -np.inf

    for blocks in levels:
        for block in blocks:
            values = np.asarray(block.point_data[scalar_name])
            # O MFSim pode usar valores enormes como sentinelas.
            values = values[np.isfinite(values) & (np.abs(values) < 1.0e20)]
            if values.size:
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))

    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError(f"Não há valores finitos para {scalar_name}.")

    return minimum, maximum


def timestep_number(path):
    return int(path.name.split(".")[-2])


def discover_timesteps():
    files = sorted(
        OUTPUT_DIR.glob("ns_output_ct.*.hdf5"),
        key=timestep_number,
    )
    if not files:
        raise FileNotFoundError(
            f"Nenhum arquivo ns_output_ct.*.hdf5 encontrado em {OUTPUT_DIR}"
        )
    return files


def global_speed_range(files):
    """Obtém uma escala fixa para impedir mudanças de cor entre frames."""
    minimum = np.inf
    maximum = -np.inf

    print("Calculando escala global de velocidade...")
    for index, path in enumerate(files, start=1):
        with h5py.File(path, "r") as hdf:
            base_level = read_level(hdf, 0)
        local_minimum, local_maximum = finite_range(
            [base_level], "velocity_magnitude"
        )
        minimum = min(minimum, local_minimum)
        maximum = max(maximum, local_maximum)
        print(
            f"  [{index:02d}/{len(files):02d}] {path.name}: "
            f"{local_minimum:.6f} a {local_maximum:.6f}"
        )

    # Evita uma lookup table degenerada em campos inicialmente uniformes.
    span = maximum - minimum
    reference = max(abs(minimum), abs(maximum), 1.0)
    if span <= reference * 1.0e-12:
        padding = reference * 0.01
        minimum -= padding
        maximum += padding

    return minimum, maximum


def load_base_frame(path, center):
    with h5py.File(path, "r") as hdf:
        physical_time = float(hdf.attrs["time"])
        base_level = read_level(hdf, 0)

    sections = []
    for block in base_level:
        section = block.slice(normal=SLICE_NORMAL, origin=center)
        if section.n_points:
            sections.append(section)

    return base_level, sections, physical_time


def create_streamlines(base_level):
    """Integra trajetórias 3D a partir de sementes próximas à entrada."""
    volume = base_level.combine(merge_points=True)
    volume.set_active_vectors("velocity")

    xmin, xmax, ymin, ymax, zmin, zmax = volume.bounds
    x_length = xmax - xmin
    y_margin = 0.25 * (ymax - ymin)
    z_margin = 0.25 * (zmax - zmin)

    seed_points = np.array(
        [
            [xmin + 0.015 * x_length, y, z]
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
    seeds = pv.PointSet(seed_points)

    lines = volume.streamlines_from_source(
        seeds,
        vectors="velocity",
        integration_direction="forward",
        integrator_type=45,
        max_length=x_length * 1.15,
        initial_step_length=x_length / 500.0,
        terminal_speed=1.0e-8,
        compute_vorticity=False,
    )

    if "velocity" in lines.point_data:
        lines.point_data["velocity_magnitude"] = np.linalg.norm(
            lines.point_data["velocity"], axis=1
        )

    return lines


def create_wake_surface(base_level):
    """Extrai uma isosuperfície 3D representando o déficit da esteira."""
    volume = base_level.combine(merge_points=True)
    values = np.asarray(volume.point_data["velocity_magnitude"])
    finite = values[np.isfinite(values)]

    if not finite.size or not (finite.min() <= WAKE_SPEED <= finite.max()):
        return None

    surface = volume.contour(
        [WAKE_SPEED], scalars="velocity_magnitude"
    )
    return surface if surface.n_points else None


def create_scene(files, levels, physical_time, speed_range):
    plotter = pv.Plotter(
        off_screen=not INTERACTIVE,
        window_size=(1600, 900),
    )
    plotter.set_background(BACKGROUND)

    # O nível 0 cobre todo o domínio sem lacunas. Ele é usado no corte
    # para evitar sobreposição visual dos diferentes níveis AMR.
    base_level = levels[0]
    complete_domain = base_level.combine()
    center = np.asarray(complete_domain.center)

    # Uma única LUT compartilhada impede o VTK de recalcular a escala
    # quando os arrays e as streamlines são substituídos.
    lookup_table = pv.LookupTable()
    lookup_table.apply_cmap(COLORMAP, n_values=256)
    lookup_table.scalar_range = speed_range

    # Corte longitudinal pelo centro do domínio. Mantemos os datasets
    # para trocar seus valores durante a animação sem perder a câmera.
    section_datasets = []
    section_actors = []
    for block in base_level:
        section = block.slice(normal=SLICE_NORMAL, origin=center)
        if section.n_points:
            actor = plotter.add_mesh(
                section,
                scalars="velocity_magnitude",
                cmap=COLORMAP,
                clim=speed_range,
                opacity=0.68 if SHOW_REFERENCE_SLICE else 0.0,
                show_scalar_bar=False,
            )
            actor.mapper.lookup_table = lookup_table
            actor.mapper.scalar_range = speed_range
            section_datasets.append(section)
            section_actors.append(actor)

    streamline_actor = None
    if SHOW_STREAMLINES:
        initial_lines = create_streamlines(base_level)
        if initial_lines.n_points:
            streamline_actor = plotter.add_mesh(
                initial_lines,
                name="streamlines",
                color=STREAMLINE_COLOR,
                opacity=0.72,
                line_width=2,
                render_lines_as_tubes=False,
                lighting=False,
                show_scalar_bar=False,
            )

    if SHOW_WAKE_ISOSURFACE:
        initial_wake = create_wake_surface(base_level)
        if initial_wake is not None:
            plotter.add_mesh(
                initial_wake,
                name="wake",
                color=WAKE_COLOR,
                opacity=0.32,
                smooth_shading=True,
                show_scalar_bar=False,
            )

    # Extrai dwall_s = 0 prioritariamente dos níveis mais refinados.
    sphere_parts = []
    for blocks in reversed(levels):
        for block in blocks:
            values = np.asarray(block.point_data["dwall_s"])
            finite = values[np.isfinite(values)]
            if finite.size and finite.min() <= 0 <= finite.max():
                surface = block.contour([0.0], scalars="dwall_s")
                if surface.n_points:
                    sphere_parts.append(surface)

    if sphere_parts:
        sphere = pv.MultiBlock(sphere_parts).combine().clean()
        plotter.add_mesh(
            sphere,
            color=GEOMETRY_COLOR,
            smooth_shading=True,
            metallic=0.15,
            roughness=0.42,
        )
    else:
        print("Aviso: nenhuma superfície dwall_s = 0 foi encontrada.")

    plotter.add_scalar_bar(
        title="|u|",
        mapper=section_actors[0].mapper,
        n_labels=5,
        fmt="%.4f",
        color=TEXT_COLOR,
        title_font_size=18,
        label_font_size=14,
        position_x=0.86,
        position_y=0.18,
        width=0.055,
        height=0.58,
        vertical=True,
    )

    plotter.add_text(
        "MFLab • Escoamento ao redor de uma esfera",
        position="upper_left",
        font_size=15,
        color=TEXT_COLOR,
    )
    time_actor = plotter.add_text(
        f"t = {physical_time:.6f}",
        position=(20, 20),
        font_size=12,
        color=TEXT_COLOR,
    )
    plotter.add_text(
        f"Plano central: |u|  •  Esteira 3D: |u| = {WAKE_SPEED:.2f}",
        position="lower_right",
        font_size=10,
        color=TEXT_COLOR,
    )

    plotter.add_axes(
        xlabel="x",
        ylabel="y",
        zlabel="z",
        color=TEXT_COLOR,
    )

    # Vista lateral levemente elevada: o eixo x do escoamento permanece
    # horizontal e a projeção paralela evita a antiga aparência radial.
    length = complete_domain.length
    plotter.camera_position = [
        center + np.array([0.10, -0.95, 0.24]) * length,
        center,
        (0, 0, 1),
    ]
    plotter.enable_parallel_projection()
    plotter.camera.zoom(1.28)

    if INTERACTIVE:
        def update_frame(step):
            path = files[step]
            new_base_level, new_sections, new_time = load_base_frame(
                path, center
            )

            if len(new_sections) != len(section_datasets):
                raise RuntimeError(
                    "A topologia do nível base mudou entre os timesteps."
                )

            for target, source in zip(section_datasets, new_sections):
                # A geometria permanece fixa. Atualizar apenas o array evita
                # trocar o campo ativo ou a lookup table do mapper.
                target.point_data["velocity_magnitude"][:] = (
                    source.point_data["velocity_magnitude"]
                )
                target.GetPointData().GetArray(
                    "velocity_magnitude"
                ).Modified()
                target.GetPointData().Modified()
                target.Modified()

            # Alguns backends do VTK recalculam o range quando o array muda.
            # Reaplicamos a mesma escala científica em todos os frames.
            for actor in section_actors:
                actor.mapper.scalar_range = speed_range
                actor.mapper.lookup_table.scalar_range = speed_range

            if SHOW_STREAMLINES:
                new_lines = create_streamlines(new_base_level)
                if new_lines.n_points:
                    new_actor = plotter.add_mesh(
                        new_lines,
                        name="streamlines",
                        color=STREAMLINE_COLOR,
                        opacity=0.72,
                        line_width=2,
                        render_lines_as_tubes=False,
                        lighting=False,
                        show_scalar_bar=False,
                        reset_camera=False,
                        render=False,
                    )

            if SHOW_WAKE_ISOSURFACE:
                new_wake = create_wake_surface(new_base_level)
                if new_wake is not None:
                    plotter.add_mesh(
                        new_wake,
                        name="wake",
                        color=WAKE_COLOR,
                        opacity=0.32,
                        smooth_shading=True,
                        show_scalar_bar=False,
                        reset_camera=False,
                        render=False,
                    )
                else:
                    plotter.remove_actor(
                        "wake", reset_camera=False, render=False
                    )

            lookup_table.scalar_range = speed_range

            time_actor.SetInput(
                f"t = {new_time:.6f}  •  frame {step + 1}/{len(files)}"
            )
            plotter.render()

        # O loop explícito é mais consistente no Windows que add_timer_event.
        # interactive_update manté a janela responsiva para girar/zoom enquanto
        # os dados mudam.
        plotter.show(
            auto_close=False,
            interactive=True,
            interactive_update=True,
        )

        try:
            for _ in range(LOOP_COUNT):
                for step in range(len(files)):
                    update_frame(step)
                    plotter.update(force_redraw=True)
                    time.sleep(FRAME_DURATION_MS / 1000.0)
        except (KeyboardInterrupt, RuntimeError):
            pass

        if plotter.render_window is not None:
            plotter.screenshot(str(SCREENSHOT))
            plotter.show()
    else:
        plotter.show(screenshot=str(SCREENSHOT), auto_close=True)

    print(f"Imagem salva em: {SCREENSHOT.resolve()}")


def main():
    files = discover_timesteps()
    first_file = files[0]
    print(f"{len(files)} timesteps encontrados.")

    with h5py.File(first_file, "r") as hdf:
        physical_time = float(hdf.attrs["time"])
        levels = collect_all_levels(hdf)

    speed_range = global_speed_range(files)
    print(f"Tempo físico: {physical_time}")
    print("Intervalo global de velocidade:", speed_range)
    create_scene(files, levels, physical_time, speed_range)


if __name__ == "__main__":
    main()