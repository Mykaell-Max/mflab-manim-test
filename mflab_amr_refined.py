"""Reconstrução AMR refinada e visualização 3D de saídas do MFSim.

Três estágios independentes:

  1. --preprocess  HDF5 Chombo/MFSim -> grade uniforme (.vti por timestep).
  2. --geometry    .vti -> geometria leve por timestep (.vtp em scene/).
  3. --view        carrega a geometria pronta e anima.

A separação existe por um motivo de desempenho: corte, streamlines e
isosuperfícies custam segundos sobre milhões de pontos e não podem ser
recalculados dentro do laço de renderização. O estágio --view apenas troca
o dataset já pronto de cada mapper, então a câmera permanece responsiva.

Todas as escalas (|u| e o limiar de Q) são globais e ficam registradas em
scene/scene.json, para que os frames permaneçam comparáveis entre si.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates


# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------

# Máquina de produção (Linux). Use --data para qualquer outro caminho.
OUTPUT_DIR = Path("/home/max/Downloads/output")

# 0.004: recomendado para validar (~4,3 milhões de pontos/frame).
# 0.002: preserva o dx mais fino no domínio inteiro (~34 milhões/frame).
TARGET_DX = 0.004

# vec_dx do level_0 neste caso. Usado só no relatório de --inspect, para
# traduzir amr_level em dx nativo; não entra em nenhum cálculo.
BASE_LEVEL_DX = 0.032

# Recorte: reamostra só a caixa que contém refinamento de nível >= CROP_LEVEL,
# unida sobre todos os timesteps processados. None = domínio inteiro.
# Fora dessa caixa o AMR não refinou, então uma grade fina ali apenas
# interpola dado grosseiro a custo alto.
CROP_LEVEL = None
CROP_MARGIN_CELLS = 4

CACHE_DIR_NAME = "mflab_amr_cache"
CACHE_FLOAT_TYPE = np.float32
COMPONENTS_TO_CACHE = ("u", "v", "w", "pressure", "dwall_s")

# --- esteira -----------------------------------------------------------------
# "q": isosuperfície de Q-criterion, revela os vórtices desprendidos.
# "speed": isosuperfície de |u|, mantida para comparação.
WAKE_MODE = "q"
WAKE_SPEED = 0.80

# Limiar da isosuperfície em unidades normalizadas Q* = Q / (U/D)², com U a
# velocidade de referência do escoamento livre e D o diâmetro da esfera.
# Normalizar assim mantém o limiar comparável entre resoluções e casos; um
# percentil, não: o conjunto de Q positivo é uma fração ínfima do domínio,
# e qualquer percentil alto sobre ele aterrissa junto ao máximo.
REFERENCE_VELOCITY = 1.0
Q_STAR_LEVEL = 0.10
# Níveis apenas reportados, para escolher Q_STAR_LEVEL com dado na mão.
Q_STAR_REPORT = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)

# Faixa excluída ao redor do sólido, em diâmetros de esfera. Em unidades
# físicas, não em células: assim o mesmo valor significa a mesma coisa em
# qualquer resolução. A camada limite tem o Q mais alto de todo o campo; sem
# excluí-la, a isosuperfície vira uma casca colada no corpo em vez de mostrar
# a esteira. O que se remove aqui é físico, e a legenda registra isso.
Q_MASK_DIAMETERS = 0.12

# --- corte -------------------------------------------------------------------
# |u| é um escalar ruim para o corte em escoamento externo: quase todo o
# domínio está no escoamento livre, então o plano vira uma chapa de uma cor
# só. A vorticidade é praticamente nula fora da esteira, então o corte
# escurece até o fundo e só a esteira acende.
SLICE_SCALAR = "vorticity_magnitude"  # ou "velocity_magnitude"
SLICE_COLORMAP = "magma"
SLICE_PERCENTILE = 99.5  # topo da escala do corte, global e fixo
SLICE_SAMPLES_PER_FRAME = 200_000
# Fração da escala em que o corte sai de transparente para opaco. Sem isso o
# plano vira um retângulo preto flutuando, com bordas duras, mais escuro que
# o próprio fundo. Com a rampa, ele só existe onde há vorticidade.
SLICE_FADE_FRACTION = 0.12

# --- corpo imerso ------------------------------------------------------------
# "analytic": desenha a esfera ajustada, se o ajuste for confirmado.
# "contour": desenha sempre o contorno bruto de dwall_s = 0.
SPHERE_MODE = "analytic"
SPHERE_FIT_TOLERANCE = 0.35  # resíduo RMS máximo aceito, em células

# --- streamlines -------------------------------------------------------------
# Poucas linhas, semeadas junto ao corpo. Anéis largos a montante produzem
# retas paralelas que só ocluem a cena: a montante o escoamento é uniforme.
SHOW_STREAMLINES = True
STREAMLINE_RINGS = (0.55, 0.95, 1.40)
STREAMLINE_PER_RING = 8
STREAMLINE_UPSTREAM_RADII = 3.0
STREAMLINE_TUBE_FRACTION = 0.0016  # fração do comprimento do domínio

# --- animação e câmera -------------------------------------------------------
FRAME_DURATION_MS = 250
# Levemente a montante e acima: a esteira se afasta para o fundo da imagem
# em vez de ficar comprimida contra a esfera.
CAMERA_DIRECTION = (-0.25, -1.00, 0.32)
# Enquadramento em diâmetros de esfera, não no domínio inteiro: o recorte
# tem 13 D de comprimento e o assunto ocupa uns poucos D, então enquadrar a
# caixa toda deixa o objeto perdido no meio da tela.
CAMERA_FRAME_DIAMETERS = 5.5
CAMERA_FOCUS_OFFSET_D = 2.5
WINDOW_SIZE = (1600, 900)
ENABLE_SSAO = True

# --- vídeo -------------------------------------------------------------------
# A Re ~ 100 o escoamento é estacionário: depois que a esteira se forma, os
# frames de dado são quase idênticos. O movimento do vídeo vem das partículas
# advectadas e da textura LIC, não da mudança do campo.
VIDEO_FPS = 60
VIDEO_SECONDS = 20.0
VIDEO_SIZE = (1920, 1080)

PARTICLE_COUNT = 24_000
PARTICLE_TRAIL = 7  # posições passadas desenhadas como rastro
PARTICLE_LIFETIME_SECONDS = 6.0
PARTICLE_SIZE = 3.0

SHOW_LIC = True
LIC_WIDTH = 1024
LIC_STEPS = 48
LIC_CYCLES = 3.0  # listras por janela de convolução
LIC_TURNS_PER_SECOND = 0.55
LIC_CONTRAST = 0.65  # quanto a textura modula a cor do campo
LIC_FLIP_V = False  # inverta se a textura sair espelhada na vertical

CAMERA_ORBIT_DEGREES = 16.0

BACKGROUND = "#07131F"
TEXT_COLOR = "#F1F6F9"
# Corpo sólido em cinza neutro e fosco; estrutura de vórtice em ciano
# saturado. As duas precisam ser inconfundíveis: com tons próximos elas se
# fundem numa massa só e o corpo parece deformado.
GEOMETRY_COLOR = "#9AA6AE"
WAKE_COLOR = "#37D6E8"
COLORMAP = "viridis"


# -----------------------------------------------------------------------------
# Leitura Chombo/MFSim
# -----------------------------------------------------------------------------

def decode_name(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def vector3_attribute(value, dtype=float, attribute_name="vetor"):
    """Converte atributos HDF5 vetoriais simples ou estruturados."""
    value_dtype = getattr(value, "dtype", None)
    field_names = getattr(value_dtype, "names", None)

    if field_names:
        for candidate in (("x", "y", "z"), ("i", "j", "k")):
            if all(name in field_names for name in candidate):
                return np.array(
                    [value[name] for name in candidate], dtype=dtype
                )

        raise ValueError(
            f"{attribute_name} possui campos estruturados não reconhecidos: "
            f"{field_names}"
        )

    result = np.asarray(value, dtype=dtype).reshape(-1)
    if result.size != 3:
        raise ValueError(
            f"{attribute_name} deveria conter três valores, mas contém "
            f"{result.size}: {value!r}"
        )
    return result


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


def refined_bounding_box(files: list[Path], minimum_level: int):
    """União das caixas de nível >= minimum_level sobre todos os timesteps.

    Lê apenas os metadados 'boxes', nunca os dados, então custa quase nada.
    A união precisa ser sobre todos os timesteps: o AMR muda de um passo
    para outro, e a grade de saída tem que ser a mesma em todos os frames.
    """
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    found = 0

    for path in files:
        with h5py.File(path, "r") as hdf:
            origin = vector3_attribute(
                hdf.attrs["origin"], dtype=float, attribute_name="origin"
            )
            for level_index in range(
                minimum_level, int(hdf.attrs["num_levels"])
            ):
                group = hdf[f"level_{level_index}"]
                boxes = group["boxes"][:]
                if not len(boxes):
                    continue

                spacing = vector3_attribute(
                    group.attrs["vec_dx"],
                    dtype=float,
                    attribute_name="vec_dx",
                )
                box_low = np.stack(
                    [boxes["lo_i"], boxes["lo_j"], boxes["lo_k"]], axis=1
                )
                box_high = np.stack(
                    [boxes["hi_i"], boxes["hi_j"], boxes["hi_k"]], axis=1
                )
                low = np.minimum(
                    low, (origin + box_low * spacing).min(axis=0)
                )
                high = np.maximum(
                    high, (origin + (box_high + 1) * spacing).max(axis=0)
                )
                found += len(boxes)

    if not found:
        raise ValueError(
            f"Nenhum bloco de nível >= {minimum_level} em nenhum timestep."
        )

    margin = CROP_MARGIN_CELLS * TARGET_DX
    return low - margin, high + margin


def target_geometry(hdf: h5py.File, region=None):
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

    coarse_dx = vector3_attribute(
        root.attrs["vec_dx"], dtype=float, attribute_name="vec_dx"
    )
    origin = vector3_attribute(
        hdf.attrs["origin"], dtype=float, attribute_name="origin"
    )
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

    if region is None:
        return physical_min, physical_max, dimensions

    # O recorte é feito em índices da própria malha alvo, nunca em
    # coordenadas soltas: assim os pontos do recorte coincidem exatamente
    # com os do domínio inteiro, e as duas reconstruções são comparáveis.
    region_low, region_high = region
    start = np.floor(
        (region_low - physical_min) / TARGET_DX + 1.0e-9
    ).astype(int)
    stop = np.ceil(
        (region_high - physical_min) / TARGET_DX - 1.0e-9
    ).astype(int)

    start = np.maximum(start, 0)
    stop = np.minimum(stop, dimensions - 1)
    if np.any(stop <= start):
        raise ValueError(
            f"Recorte vazio ou degenerado: início {start}, fim {stop}."
        )

    cropped_min = physical_min + start * TARGET_DX
    cropped_max = physical_min + stop * TARGET_DX
    return cropped_min, cropped_max, stop - start + 1


def decode_level(hdf: h5py.File, level_index: int, components=None):
    group = hdf[f"level_{level_index}"]
    boxes = group["boxes"][:]
    raw = group["data:datatype=0"][:]
    components = components or COMPONENTS_TO_CACHE
    spacing = vector3_attribute(
        group.attrs["vec_dx"], dtype=float, attribute_name="vec_dx"
    )
    global_origin = vector3_attribute(
        hdf.attrs["origin"], dtype=float, attribute_name="origin"
    )

    component_count = int(group.attrs["num_components"])
    component_names = [
        decode_name(group.attrs[f"component_{index}"])
        for index in range(component_count)
    ]
    component_indices = {
        name: component_names.index(name) for name in components
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


def compose_uniform_grid(hdf5_path: Path, region=None) -> pv.ImageData:
    with h5py.File(hdf5_path, "r") as hdf:
        physical_time = float(hdf.attrs["time"])
        target_origin, _, dimensions = target_geometry(hdf, region)
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
    grid.field_data["crop_level"] = np.array(
        [-1 if CROP_LEVEL is None else CROP_LEVEL]
    )

    return grid


def cache_directory() -> Path:
    dx_label = str(TARGET_DX).replace(".", "p")
    suffix = "" if CROP_LEVEL is None else f"_L{CROP_LEVEL}"
    return OUTPUT_DIR / f"{CACHE_DIR_NAME}_dx_{dx_label}{suffix}"


def cache_path_for(hdf5_path: Path) -> Path:
    return cache_directory() / f"ct.{timestep_number(hdf5_path):09d}.vti"


def preprocess(files: list[Path], force: bool = False):
    destination = cache_directory()
    destination.mkdir(parents=True, exist_ok=True)

    region = None
    if CROP_LEVEL is not None:
        # Sempre sobre TODOS os timesteps, nunca sobre a seleção de --steps
        # ou --frames: a caixa é propriedade do conjunto de dados. Derivá-la
        # de um subconjunto produziria uma grade diferente com o mesmo nome
        # de cache, e os frames deixariam de ser comparáveis.
        region = refined_bounding_box(discover_hdf5_files(), CROP_LEVEL)
        with h5py.File(files[0], "r") as hdf:
            _, _, dimensions = target_geometry(hdf, region)
        print(
            f"Recorte nível >= {CROP_LEVEL}: "
            f"{np.round(region[0], 4).tolist()} a "
            f"{np.round(region[1], 4).tolist()}"
        )
        print(
            f"  grade {dimensions[0]}x{dimensions[1]}x{dimensions[2]} = "
            f"{int(np.prod(dimensions)):,} pontos por frame"
        )

    print(f"Cache: {destination}")
    for index, path in enumerate(files, start=1):
        cached = cache_path_for(path)
        if cached.exists() and not force:
            print(f"[{index:02d}/{len(files):02d}] reutilizando {cached.name}")
            continue

        print(f"[{index:02d}/{len(files):02d}] reconstruindo {path.name}")
        start = time.perf_counter()
        grid = compose_uniform_grid(path, region)
        grid.save(cached, binary=True)
        elapsed = time.perf_counter() - start

        points = grid.n_points
        print(
            f"    {points:,} pontos • {elapsed:.1f} s • "
            f"salvo em {cached.name}"
        )


# -----------------------------------------------------------------------------
# Campos derivados
# -----------------------------------------------------------------------------

def grid_shape(grid) -> tuple[int, int, int]:
    return tuple(int(value) for value in grid.dimensions)


def as_block(values, shape):
    """point_data 1-D -> bloco 3-D, respeitando a ordem F usada na escrita."""
    return np.asarray(values).reshape(shape, order="F")


def jacobian_from_block(field, spacing):
    """jacobian[a][b] = d(u_a) / d(x_b), de um bloco (nx, ny, nz, 3)."""
    return [
        [
            derivative.astype(np.float32, copy=False)
            for derivative in np.gradient(
                field[..., axis], spacing, spacing, spacing
            )
        ]
        for axis in range(3)
    ]


def velocity_jacobian(grid):
    """jacobian[a][b] = d(u_a) / d(x_b), por diferenças centrais."""
    shape = grid_shape(grid)
    velocity = np.asarray(grid.point_data["velocity"])
    field = np.stack(
        [as_block(velocity[:, axis], shape) for axis in range(3)], axis=-1
    )
    return jacobian_from_block(field, float(grid.spacing[0]))


def q_criterion(jacobian) -> np.ndarray:
    """Q = -1/2 * tr(J²) = 1/2 (|Ω|² - |S|²)."""
    q = np.zeros_like(jacobian[0][0])
    for a in range(3):
        for b in range(3):
            q -= 0.5 * jacobian[a][b] * jacobian[b][a]
    return q


def fluid_distance(grid) -> np.ndarray | None:
    """dwall_s reorientado para que valores positivos sejam fluido.

    O sinal de dwall_s não é assumido: é determinado comparando a velocidade
    média dos dois lados. O lado quase parado é o sólido.
    """
    if "dwall_s" not in grid.point_data:
        return None

    distance = np.asarray(grid.point_data["dwall_s"], dtype=np.float32)
    speed = np.asarray(grid.point_data["velocity_magnitude"])

    valid = np.isfinite(distance)
    negative = valid & (distance < 0.0)
    positive = valid & (distance > 0.0)
    if not negative.any() or not positive.any():
        return distance

    if speed[negative].mean() < speed[positive].mean():
        return distance
    return -distance


def vorticity_magnitude(jacobian) -> np.ndarray:
    """|ω| = |∇ × u|, a partir do mesmo Jacobiano usado no Q."""
    omega_x = jacobian[2][1] - jacobian[1][2]
    omega_y = jacobian[0][2] - jacobian[2][0]
    omega_z = jacobian[1][0] - jacobian[0][1]
    return np.sqrt(
        omega_x * omega_x + omega_y * omega_y + omega_z * omega_z
    )


def recirculation_field(grid, center, radius):
    """Campo cuja isosuperfície em 0 delimita a bolha de recirculação.

    Abaixo de Re ~ 210 a esteira da esfera é estacionária e axissimétrica:
    não há vórtice destacado para o Q-criterion encontrar, e a estrutura
    característica é o toro de escoamento reverso colado atrás do corpo.

    u = 0 também vale na parede (não escorregamento) e dentro do sólido,
    então essas regiões recebem valor positivo: sem isso, a isosuperfície
    envolveria a esfera inteira em vez da bolha.

    Devolve o campo e o comprimento da bolha em diâmetros, que é uma
    medida física comparável com a literatura — logo, uma validação do
    pipeline inteiro, não só um número decorativo.
    """
    shape = grid_shape(grid)
    streamwise = as_block(
        np.asarray(grid.point_data["velocity"])[:, 0], shape
    ).copy()

    distance = fluid_distance(grid)
    if distance is not None:
        solid = np.isfinite(distance) & (distance < TARGET_DX)
        streamwise[as_block(solid, shape)] = 1.0

    if center is None or not radius:
        return streamwise, None

    origin = np.asarray(grid.origin)
    spacing = float(grid.spacing[0])
    coordinates = origin[0] + np.arange(shape[0]) * spacing

    # Só a jusante do centro: a montante não há recirculação, e o contorno
    # ali seria apenas a face frontal do corpo.
    upstream = coordinates <= center[0]
    streamwise[upstream, :, :] = 1.0

    reverse = streamwise < 0.0
    if not reverse.any():
        return streamwise, None

    columns = np.any(reverse, axis=(1, 2))
    furthest = coordinates[np.nonzero(columns)[0][-1]]
    # Medida a partir do polo traseiro da esfera, não do centro.
    length = (furthest - (center[0] + radius)) / (2.0 * radius)
    return streamwise, float(length)


def masked_q(grid, jacobian, mask_distance: float):
    """Q com a faixa de espessura mask_distance junto ao sólido zerada.

    Devolve também o máximo antes da máscara: comparar os dois valores
    distingue um campo genuinamente sem vórtices de uma máscara larga
    demais, que são falhas completamente diferentes.
    """
    q = q_criterion(jacobian)
    raw_maximum = float(q.max())

    distance = fluid_distance(grid)
    if distance is not None and mask_distance > 0.0:
        # NaN em dwall_s vem do sentinela 1e40, ou seja, longe da parede.
        solid = np.isfinite(distance) & (distance < mask_distance)
        q[as_block(solid, q.shape)] = 0.0

    return q, raw_maximum, float(q.max())


# -----------------------------------------------------------------------------
# Estágio de geometria: .vti -> .vtp
# -----------------------------------------------------------------------------

def scene_directory() -> Path:
    return cache_directory() / "scene"


def scene_metadata_path() -> Path:
    return scene_directory() / "scene.json"


def physical_time(grid) -> float:
    return float(np.asarray(grid.field_data["physical_time"])[0])


def contour_scalar(grid, name, values, level, **kwargs):
    """Isosuperfície por flying edges, o método rápido para ImageData."""
    source = pv.ImageData(
        dimensions=grid_shape(grid),
        spacing=grid.spacing,
        origin=grid.origin,
    )
    source.point_data[name] = values
    surface = source.contour(
        [level],
        scalars=name,
        method="flying_edges",
        compute_normals=True,
        **kwargs,
    )
    return surface if surface.n_points else None


def sphere_surface(grid):
    distance = fluid_distance(grid)
    if distance is None:
        return None

    # NaN significa "longe da parede": vira fluido, nunca uma falsa interface.
    filled = np.where(np.isfinite(distance), distance, 1.0e6)
    if filled.min() > 0.0 or filled.max() < 0.0:
        return None

    surface = contour_scalar(grid, "dwall_s", filled, 0.0)
    return surface.clean() if surface is not None else None


def fit_sphere(points):
    """Ajuste de esfera por mínimos quadrados.

    |p - c|² = r² é linear em (c, c·c - r²), então sai de um lstsq direto.
    O resíduo devolvido é o teste: se o contorno de dwall_s = 0 realmente
    descreve uma esfera, ele fica em fração de célula. Se não ficar, ou o
    corpo não é esférico ou dwall_s está errado — e aí nada de esférico
    deve ser desenhado.
    """
    matrix = np.hstack(
        [2.0 * points, np.ones((len(points), 1))]
    )
    target = (points ** 2).sum(axis=1)
    solution, *_ = np.linalg.lstsq(matrix, target, rcond=None)

    center = solution[:3]
    radius = float(np.sqrt(solution[3] + center @ center))
    residual = np.linalg.norm(points - center, axis=1) - radius
    return center, radius, residual


def body_surface(contour):
    """Geometria do corpo imerso, com o ajuste esférico validado.

    O corpo imerso é dado de entrada da simulação, não resultado dela: sua
    forma é conhecida exatamente. Quando o ajuste confirma que o contorno é
    uma esfera dentro de uma fração de célula, desenhar a esfera analítica
    representa o corpo melhor que o marching cubes de um campo de distância
    interpolado — que produz facetas e depressões que não existem.
    Se o ajuste não confirmar, devolve o contorno original.
    """
    if contour is None or not contour.n_points:
        return None, None, None

    center, radius, residual = fit_sphere(np.asarray(contour.points))
    rms = float(np.sqrt(np.mean(residual ** 2)))
    worst = float(np.abs(residual).max())
    print(
        f"  ajuste esférico: centro {np.round(center, 4)} • "
        f"raio {radius:.5f} • resíduo RMS {rms / TARGET_DX:.3f} célula, "
        f"máx {worst / TARGET_DX:.3f} célula"
    )

    if SPHERE_MODE == "contour" or rms > SPHERE_FIT_TOLERANCE * TARGET_DX:
        if SPHERE_MODE != "contour":
            print(
                "  AVISO: resíduo acima da tolerância; usando o contorno "
                "bruto de dwall_s = 0 em vez da esfera analítica."
            )
        return contour, center, radius

    analytic = pv.Sphere(
        radius=radius,
        center=tuple(center),
        theta_resolution=120,
        phi_resolution=120,
    )
    return analytic, center, radius


def seed_points(grid, center, radius):
    """Anéis concêntricos a montante da esfera.

    Semear num plano de entrada inteiro produz linhas paralelas que em sua
    maioria passam longe do corpo. Anéis centrados no eixo garantem que as
    linhas contornem a esfera e entrem na esteira.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = grid.bounds

    if center is None:
        center = np.array(
            [xmin, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)]
        )
        radius = 0.10 * (ymax - ymin)

    start_x = max(
        center[0] - STREAMLINE_UPSTREAM_RADII * radius,
        xmin + 2.0 * TARGET_DX,
    )

    points = [[start_x, center[1], center[2]]]
    for ring_index, ring in enumerate(STREAMLINE_RINGS):
        # Deslocamento angular por anel evita alinhamento radial artificial.
        offset = 0.5 * ring_index
        for step in range(STREAMLINE_PER_RING):
            angle = 2.0 * np.pi * step / STREAMLINE_PER_RING + offset
            points.append(
                [
                    start_x,
                    center[1] + ring * radius * np.cos(angle),
                    center[2] + ring * radius * np.sin(angle),
                ]
            )

    points = np.array(points, dtype=float)
    inside = (
        (points[:, 1] > ymin + TARGET_DX)
        & (points[:, 1] < ymax - TARGET_DX)
        & (points[:, 2] > zmin + TARGET_DX)
        & (points[:, 2] < zmax - TARGET_DX)
    )
    return points[inside]


def streamline_tubes(grid, center, radius):
    if not SHOW_STREAMLINES:
        return None

    xmin, xmax = grid.bounds[0], grid.bounds[1]
    points = seed_points(grid, center, radius)
    if not len(points):
        return None

    lines = grid.streamlines_from_source(
        pv.PointSet(points),
        vectors="velocity",
        integration_direction="forward",
        integrator_type=45,
        # step_unit padrão é 'cl' (comprimentos de célula). Passar dx aqui
        # produziria passos 250x menores que o pretendido.
        initial_step_length=0.5,
        max_step_length=1.0,
        max_steps=8000,
        max_length=3.0 * (xmax - xmin),
        terminal_speed=1.0e-8,
        compute_vorticity=False,
    )
    if not lines.n_points:
        return None

    tube_radius = STREAMLINE_TUBE_FRACTION * (xmax - xmin)
    return lines.tube(radius=tube_radius, n_sides=8, capping=True)


def build_scene(cache_files: list[Path], force: bool = False):
    destination = scene_directory()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Geometria: {destination}")

    reference = pv.read(cache_files[len(cache_files) // 2])
    contour = sphere_surface(reference)
    sphere, center, radius = body_surface(contour)
    if sphere is not None:
        sphere.save(destination / "sphere.vtp", binary=True)
        print(
            f"  corpo: {sphere.n_points:,} pontos • "
            f"{2.0 * radius / TARGET_DX:.1f} células no diâmetro"
        )
    else:
        print("  corpo: não encontrado em dwall_s = 0")

    slice_normal = (0.0, 1.0, 0.0)
    slice_origin = (
        tuple(center) if center is not None else reference.center
    )

    # (U/D)²: converte Q absoluto em Q* adimensional.
    q_scale = (
        1.0
        if not radius
        else (REFERENCE_VELOCITY / (2.0 * radius)) ** 2
    )

    # Máscara em unidades físicas, para valer o mesmo em qualquer dx.
    mask_distance = Q_MASK_DIAMETERS * 2.0 * (radius or 0.0)
    if radius:
        print(
            f"  máscara junto ao sólido: {Q_MASK_DIAMETERS:g} D = "
            f"{mask_distance:.4f} = {mask_distance / TARGET_DX:.1f} células"
        )

    speed_minimum = np.inf
    speed_maximum = -np.inf
    wake_blocks: dict[int, np.ndarray] = {}
    bubble_lengths: dict[int, float | None] = {}
    q_star_maximum = 0.0
    q_star_counts = np.zeros(len(Q_STAR_REPORT), dtype=np.int64)
    vorticity_samples: list[np.ndarray] = []
    times: dict[int, float] = {}
    generator = np.random.default_rng(0)

    for index, path in enumerate(cache_files, start=1):
        step = timestep_number(path)
        grid = pv.read(path)
        times[step] = physical_time(grid)

        speed = np.asarray(grid.point_data["velocity_magnitude"])
        finite = speed[np.isfinite(speed) & (np.abs(speed) < 1.0e20)]
        speed_minimum = min(speed_minimum, float(finite.min()))
        speed_maximum = max(speed_maximum, float(finite.max()))

        started = time.perf_counter()

        # O Jacobiano serve tanto à vorticidade do corte quanto ao Q, então
        # é calculado uma vez só, antes de fatiar.
        jacobian = velocity_jacobian(grid)
        vorticity = vorticity_magnitude(jacobian)
        grid.point_data["vorticity_magnitude"] = vorticity.ravel(order="F")

        flat = vorticity.ravel()
        sample_size = min(SLICE_SAMPLES_PER_FRAME, flat.size)
        vorticity_samples.append(
            generator.choice(flat, sample_size, replace=False)
        )

        section = grid.slice(normal=slice_normal, origin=slice_origin)
        section.save(destination / f"slice.{step:09d}.vtp", binary=True)

        tubes = streamline_tubes(grid, center, radius)
        if tubes is not None:
            tubes.save(
                destination / f"streamlines.{step:09d}.vtp", binary=True
            )

        q_report = ""
        if WAKE_MODE == "q":
            block, raw_maximum, masked_maximum = masked_q(
                grid, jacobian, mask_distance
            )
            wake_blocks[step] = block

            for position, q_star in enumerate(Q_STAR_REPORT):
                q_star_counts[position] += int(
                    (block > q_star * q_scale).sum()
                )
            q_star_maximum = max(q_star_maximum, masked_maximum / q_scale)
            q_report = (
                f" • Q* max {masked_maximum / q_scale:.3g} "
                f"(bruto {raw_maximum / q_scale:.3g}, "
                f"{int((block > 0.0).sum()):,} pts > 0)"
            )
        elif WAKE_MODE == "reverse":
            block, length = recirculation_field(grid, center, radius)
            wake_blocks[step] = block
            bubble_lengths[step] = length
            q_report = (
                " • bolha L/D "
                + ("indefinida" if length is None else f"{length:.3f}")
            )
        else:
            wake_blocks[step] = as_block(
                np.asarray(grid.point_data["velocity_magnitude"]),
                grid_shape(grid),
            ).copy()

        del jacobian, vorticity
        elapsed = time.perf_counter() - started
        tube_points = tubes.n_points if tubes is not None else 0
        print(
            f"[{index:02d}/{len(cache_files):02d}] ct.{step:09d} • "
            f"corte {section.n_points:,} • tubos {tube_points:,}"
            f"{q_report} • {elapsed:.1f} s",
            flush=True,
        )

    if WAKE_MODE == "q":
        print(
            f"D = {2.0 * (radius or 0.0):.4f} • U = {REFERENCE_VELOCITY:g} "
            f"• (U/D)² = {q_scale:.4g} • Q* máximo observado "
            f"{q_star_maximum:.3g}"
        )
        print("  pontos acima de cada Q*, somados sobre os frames:")
        for q_star, count in zip(Q_STAR_REPORT, q_star_counts):
            marker = " <- Q_STAR_LEVEL" if q_star == Q_STAR_LEVEL else ""
            print(f"    Q* >= {q_star:<5g} {count:>12,}{marker}")

        if q_star_maximum <= 0.0:
            # Campo uniforme (tipicamente a condição inicial) produz Q = 0
            # em todo o domínio. É o resultado correto, não uma falha:
            # não existe vórtice em t = 0.
            wake_level = None
            print(
                "  Nenhum Q positivo. Esperado para a condição inicial; a "
                "cena fica sem superfície de vórtice. Em timesteps "
                "desenvolvidos, compare 'Q* max' com 'bruto': se o bruto "
                "for alto e o mascarado nulo, Q_MASK_DIAMETERS está largo."
            )
        else:
            wake_level = Q_STAR_LEVEL * q_scale
            if Q_STAR_LEVEL > q_star_maximum:
                print(
                    f"  AVISO: Q_STAR_LEVEL ({Q_STAR_LEVEL:g}) excede o "
                    f"máximo observado ({q_star_maximum:.3g}). A "
                    "isosuperfície ficará vazia."
                )
            print(
                f"  limiar: Q* = {Q_STAR_LEVEL:g} → Q = {wake_level:.4g} "
                "(fixo entre todos os frames)"
            )
    elif WAKE_MODE == "reverse":
        wake_level = 0.0
        measured = [
            value for value in bubble_lengths.values() if value is not None
        ]
        if measured:
            print(
                f"Bolha de recirculação: L/D de {min(measured):.3f} a "
                f"{max(measured):.3f} sobre os frames processados"
            )
            print(
                "  para esfera a Re ~ 100 a literatura reporta L/D da ordem "
                "de 0.8 a 1.0; confira contra a sua referência antes de "
                "publicar o número"
            )
        else:
            print(
                "Nenhum escoamento reverso encontrado: sem bolha de "
                "recirculação neste intervalo."
            )
    else:
        wake_level = WAKE_SPEED
        print(f"Isosuperfície global de |u|: {wake_level:.4f}")

    for index, path in enumerate(cache_files, start=1):
        step = timestep_number(path)

        if wake_level is None:
            wake_blocks.clear()
            break

        # Todos os modos passam pelo mesmo caminho: o campo já está pronto
        # em wake_blocks e só falta cortá-lo no nível global.
        block = wake_blocks.pop(step)
        source = pv.ImageData(
            dimensions=block.shape,
            spacing=reference.spacing,
            origin=reference.origin,
        )
        source.point_data["wake"] = block.ravel(order="F")
        surface = source.contour(
            [wake_level],
            scalars="wake",
            method="flying_edges",
            compute_normals=True,
        )

        if surface.n_points:
            surface.save(destination / f"wake.{step:09d}.vtp", binary=True)
        print(
            f"[{index:02d}/{len(cache_files):02d}] ct.{step:09d} • "
            f"esteira {surface.n_points:,} pontos",
            flush=True,
        )

    pooled_vorticity = np.concatenate(vorticity_samples)
    slice_range = [
        0.0,
        float(np.percentile(pooled_vorticity, SLICE_PERCENTILE)),
    ]
    print(
        f"Escala global do corte ({SLICE_SCALAR}): 0 a "
        f"{slice_range[1]:.4g} (percentil {SLICE_PERCENTILE} sobre todos "
        "os frames)"
    )

    metadata = {
        "target_dx": TARGET_DX,
        "slice_scalar": SLICE_SCALAR,
        "slice_range": slice_range,
        "q_mask_diameters": Q_MASK_DIAMETERS if WAKE_MODE == "q" else None,
        "wake_mode": WAKE_MODE,
        "wake_level": wake_level,
        "q_star": Q_STAR_LEVEL if WAKE_MODE == "q" else None,
        "q_scale": q_scale if WAKE_MODE == "q" else None,
        "speed_range": [speed_minimum, speed_maximum],
        "sphere_center": None if center is None else center.tolist(),
        "sphere_radius": radius,
        "steps": [timestep_number(path) for path in cache_files],
        "times": {str(step): value for step, value in times.items()},
    }
    scene_metadata_path().write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        f"Escala global |u|: {speed_minimum:.6f} a {speed_maximum:.6f} • "
        f"metadados em {scene_metadata_path().name}"
    )


# -----------------------------------------------------------------------------
# Visualização
# -----------------------------------------------------------------------------

def axis_extent(mask_block, origin, spacing, axis):
    """Extensão física ocupada por uma máscara ao longo de um eixo."""
    others = tuple(other for other in range(3) if other != axis)
    present = np.any(mask_block, axis=others)
    if not present.any():
        return None
    low = int(np.argmax(present))
    high = len(present) - 1 - int(np.argmax(present[::-1]))
    return origin[axis] + low * spacing, origin[axis] + high * spacing


def report_flow_parameters(hdf5_path: Path, diameter: float):
    """Reynolds a partir da viscosidade e densidade gravadas no arquivo.

    O regime decide o que é possível visualizar. Para esfera: abaixo de
    Re ~ 210 a esteira é estacionária e axissimétrica, sem desprendimento
    nenhum; entre 210 e 270 fica estacionária com dois filamentos; só acima
    de ~270 há desprendimento periódico de vórtices em grampo. Perseguir
    uma rua de vórtices num escoamento estacionário é perseguir algo que
    não existe no dado.
    """
    with h5py.File(hdf5_path, "r") as hdf:
        blocks = decode_level(hdf, 0, components=("viscosity", "density"))

    viscosity = np.concatenate(
        [block["arrays"]["viscosity"].ravel() for block in blocks]
    )
    density = np.concatenate(
        [block["arrays"]["density"].ravel() for block in blocks]
    )

    print(
        f"  viscosity: {viscosity.min():.6g} .. {viscosity.max():.6g}"
        f" • density: {density.min():.6g} .. {density.max():.6g}"
    )

    mu = float(np.median(viscosity))
    rho = float(np.median(density))
    if mu <= 0.0:
        print("  viscosidade nula ou negativa; Reynolds indeterminado.")
        return

    # MFSim não declara aqui se 'viscosity' é dinâmica ou cinemática, e a
    # diferença é um fator rho. Reporto as duas leituras em vez de eleger
    # uma: com rho = 1 elas coincidem de qualquer forma.
    print(
        f"  Re = rho*U*D/mu  = {rho * REFERENCE_VELOCITY * diameter / mu:.1f}"
        "   (se viscosity for dinâmica)"
    )
    print(
        f"  Re = U*D/nu      = {REFERENCE_VELOCITY * diameter / mu:.1f}"
        "   (se viscosity for cinemática)"
    )
    print(
        "  referência esfera: Re < 210 estacionária axissimétrica • "
        "210-270 estacionária bifilamentar • > 270 desprendimento periódico"
    )


def inspect_cache(cache_files: list[Path]):
    """Valida a reconstrução AMR antes de confiar em qualquer imagem.

    Responde: a cobertura é completa? onde cada nível prevaleceu? qual a
    resolução efetiva dentro da esteira? Sem isso, uma cena plausível pode
    estar associando células ao nível errado.
    """
    reference = pv.read(cache_files[len(cache_files) // 2])
    _, center, radius = body_surface(sphere_surface(reference))
    if center is not None:
        print(
            f"Esfera: centro {np.round(center, 4)} • "
            f"D = {2.0 * radius:.4f} • {2.0 * radius / TARGET_DX:.1f} "
            "células no diâmetro"
        )
        source = OUTPUT_DIR / (
            f"ns_output_ct.{timestep_number(cache_files[0]):09d}.hdf5"
        )
        if source.exists():
            report_flow_parameters(source, 2.0 * radius)

    for path in cache_files:
        grid = pv.read(path)
        shape = grid_shape(grid)
        origin = np.asarray(grid.origin)
        spacing = float(grid.spacing[0])

        print(f"\n=== {path.name} • t = {physical_time(grid):.6f}")
        print(
            f"  grade {shape[0]}x{shape[1]}x{shape[2]} = "
            f"{grid.n_points:,} pontos • dx = {spacing:g}"
        )
        print(f"  limites {np.round(grid.bounds, 4).tolist()}")

        level = np.asarray(grid.point_data["amr_level"])
        uncovered = int((level < 0).sum())
        print(
            f"  cobertura: {uncovered:,} pontos SEM NÍVEL"
            if uncovered
            else "  cobertura: completa"
        )

        print("  nível        pontos   fração  caixa física ocupada")
        for value in np.unique(level):
            selection = as_block(level == value, shape)
            count = int(selection.sum())
            extents = [
                axis_extent(selection, origin, spacing, axis)
                for axis in range(3)
            ]
            box = " ".join(
                "-" if e is None else f"[{e[0]:.3f},{e[1]:.3f}]"
                for e in extents
            )
            print(
                f"  {value:>5}  {count:>12,}  {count / level.size:>6.2%}  "
                f"{box}"
            )

        velocity = np.asarray(grid.point_data["velocity"])
        for name, values in (
            ("u", velocity[:, 0]),
            ("v", velocity[:, 1]),
            ("w", velocity[:, 2]),
            ("|u|", np.asarray(grid.point_data["velocity_magnitude"])),
            ("pressure", np.asarray(grid.point_data["pressure"])),
            ("dwall_s", np.asarray(grid.point_data["dwall_s"])),
        ):
            finite = values[np.isfinite(values)]
            nan_count = int(np.isnan(values).sum())
            suffix = f" • {nan_count:,} NaN" if nan_count else ""
            print(
                f"  {name:>9}: {finite.min():>12.6g} .. "
                f"{finite.max():>12.6g}{suffix}"
            )

        if center is None:
            continue

        # Resolução efetiva onde a esteira realmente está.
        coordinates = [
            origin[axis] + np.arange(shape[axis]) * spacing
            for axis in range(3)
        ]
        downstream = coordinates[0] > center[0] + radius
        radial = np.hypot(
            (coordinates[1] - center[1])[:, None],
            (coordinates[2] - center[2])[None, :],
        )
        wake = downstream[:, None, None] & (radial < 2.0 * radius)[None]
        wake_levels = as_block(level, shape)[wake]

        if wake_levels.size:
            print(
                f"  esteira (x > {center[0] + radius:.3f}, r < "
                f"{2.0 * radius:.3f}): {wake_levels.size:,} pontos"
            )
            for value in np.unique(wake_levels):
                share = int((wake_levels == value).sum()) / wake_levels.size
                print(
                    f"    nível {value}: {share:>6.2%} • "
                    f"dx nativo {BASE_LEVEL_DX / 2 ** int(value):g}"
                )


# -----------------------------------------------------------------------------
# Advecção de partículas
# -----------------------------------------------------------------------------

def as_vector_block(grid) -> np.ndarray:
    """point_data 'velocity' -> bloco (nx, ny, nz, 3)."""
    shape = grid_shape(grid)
    velocity = np.asarray(grid.point_data["velocity"], dtype=np.float32)
    return np.stack(
        [velocity[:, axis].reshape(shape, order="F") for axis in range(3)],
        axis=-1,
    )


def sample_vectors(field, origin, spacing, positions):
    """Interpolação trilinear do campo nas posições dadas."""
    coordinates = ((positions - origin) / spacing).T
    return np.stack(
        [
            map_coordinates(
                field[..., axis],
                coordinates,
                order=1,
                mode="nearest",
                prefilter=False,
            )
            for axis in range(3)
        ],
        axis=-1,
    )


def advect_rk4(positions, sampler, dt):
    """Runge-Kutta 4. Passo fixo: o campo é suave e o dt vem do fps."""
    k1 = sampler(positions)
    k2 = sampler(positions + 0.5 * dt * k1)
    k3 = sampler(positions + 0.5 * dt * k2)
    k4 = sampler(positions + dt * k3)
    return positions + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def spawn_positions(count, bounds, generator, inlet_fraction=0.02):
    """Semeia na face de entrada, distribuído pela seção transversal."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    span = xmax - xmin
    return np.column_stack(
        [
            generator.uniform(xmin, xmin + inlet_fraction * span, count),
            generator.uniform(ymin, ymax, count),
            generator.uniform(zmin, zmax, count),
        ]
    ).astype(np.float32)


def expired(positions, ages, bounds, lifetime):
    """Partículas que saíram do domínio ou envelheceram demais.

    O limite de idade existe por causa da bolha de recirculação: uma
    partícula capturada ali nunca sai sozinha, e sem renovação a população
    inteira acabaria acumulada dentro do toro.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    outside = (
        (positions[:, 0] < xmin)
        | (positions[:, 0] > xmax)
        | (positions[:, 1] < ymin)
        | (positions[:, 1] > ymax)
        | (positions[:, 2] < zmin)
        | (positions[:, 2] > zmax)
    )
    return outside | (ages > lifetime)


def trail_connectivity(count, trail):
    """Conectividade VTK de uma polilinha por partícula, constante."""
    cells = np.empty((count, trail + 1), dtype=np.int64)
    cells[:, 0] = trail
    indices = np.arange(count, dtype=np.int64)
    for position in range(trail):
        cells[:, position + 1] = position * count + indices
    return cells.ravel()


# -----------------------------------------------------------------------------
# Line Integral Convolution
# -----------------------------------------------------------------------------

def lic_samples(vx, vy, noise, steps):
    """Amostras de ruído ao longo da linha de corrente de cada pixel.

    Devolve (2*steps, altura, largura). O caro é isto, e só depende do
    campo — não da fase. Como o campo é praticamente estacionário, isto é
    calculado uma vez por timestep de dado e reaproveitado por todos os
    frames de vídeo, que passam a custar apenas uma soma ponderada.
    """
    height, width = vx.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float32)

    magnitude = np.maximum(np.hypot(vx, vy), 1.0e-12)
    unit_x = (vx / magnitude).astype(np.float32)
    unit_y = (vy / magnitude).astype(np.float32)

    collected = np.empty((2 * steps, height, width), dtype=np.float32)

    for order, direction in enumerate((1.0, -1.0)):
        position_x = grid_x.copy()
        position_y = grid_y.copy()
        for step in range(steps):
            coordinates = np.stack([position_y.ravel(), position_x.ravel()])
            collected[order * steps + step] = map_coordinates(
                noise,
                coordinates,
                order=1,
                mode="wrap",
                prefilter=False,
            ).reshape(height, width)

            local = np.stack([position_y.ravel(), position_x.ravel()])
            position_x += direction * map_coordinates(
                unit_x, local, order=1, mode="nearest", prefilter=False
            ).reshape(height, width)
            position_y += direction * map_coordinates(
                unit_y, local, order=1, mode="nearest", prefilter=False
            ).reshape(height, width)

    return collected


def lic_weights(steps, phase, cycles):
    """Núcleo periódico deslocado pela fase.

    Deslocar a fase a cada frame faz o padrão caminhar ao longo das linhas
    de corrente: é isso que produz a sensação de escoamento, sem alterar
    nenhum valor do campo.
    """
    arc = np.arange(steps, dtype=np.float32) + 0.5
    signed = np.concatenate([arc, -arc])
    weights = 0.5 * (
        1.0 + np.cos(2.0 * np.pi * (cycles * signed / steps - phase))
    )
    return (weights / weights.sum()).astype(np.float32)


def lic_luminance(samples, phase, cycles):
    """Combina as amostras com o núcleo, normalizando para [0, 1]."""
    steps = samples.shape[0] // 2
    weights = lic_weights(steps, phase, cycles)
    image = np.tensordot(weights, samples, axes=(0, 0))

    low, high = np.percentile(image, (2.0, 98.0))
    if high <= low:
        return np.full_like(image, 0.5)
    return np.clip((image - low) / (high - low), 0.0, 1.0)


def discover_cache_files() -> list[Path]:
    files = sorted(cache_directory().glob("ct.*.vti"))
    if not files:
        raise FileNotFoundError(
            "Nenhum cache VTI encontrado. Execute primeiro com --preprocess."
        )
    return files


def load_scene():
    path = scene_metadata_path()
    if not path.exists():
        raise FileNotFoundError(
            "Geometria ausente. Execute primeiro com --geometry."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def read_optional(path: Path):
    return pv.read(path) if path.exists() else pv.PolyData()


def load_frames(metadata):
    """Toda a geometria vai para a RAM antes de abrir a janela."""
    directory = scene_directory()
    frames = []
    for step in metadata["steps"]:
        frames.append(
            {
                "step": step,
                "time": float(metadata["times"][str(step)]),
                "slice": read_optional(directory / f"slice.{step:09d}.vtp"),
                "streamlines": read_optional(
                    directory / f"streamlines.{step:09d}.vtp"
                ),
                "wake": read_optional(directory / f"wake.{step:09d}.vtp"),
            }
        )
    return frames


def bind_scalars(mapper, dataset, name, scalar_range, lookup_table):
    mapper.SetInputData(dataset)
    if name in dataset.point_data:
        mapper.SetScalarModeToUsePointFieldData()
        mapper.SelectColorArray(name)
        mapper.SetScalarRange(*scalar_range)
        mapper.SetLookupTable(lookup_table)
        mapper.ScalarVisibilityOn()
    else:
        mapper.ScalarVisibilityOff()


def view(metadata):
    speed_range = tuple(metadata["speed_range"])
    frames = load_frames(metadata)
    print(
        f"{len(frames)} frames em memória • "
        f"escala global |u|: {speed_range[0]:.6f} a {speed_range[1]:.6f}"
    )

    directory = scene_directory()
    sphere = read_optional(directory / "sphere.vtp")

    plotter = pv.Plotter(window_size=WINDOW_SIZE)
    plotter.set_background(BACKGROUND)
    plotter.enable_parallel_projection()

    lookup_table = pv.LookupTable()
    lookup_table.apply_cmap(COLORMAP, n_values=256)
    lookup_table.scalar_range = speed_range
    lookup_table.nan_opacity = 0.0

    slice_scalar = metadata["slice_scalar"]
    slice_range = tuple(metadata["slice_range"])
    slice_table = pv.LookupTable()
    slice_table.apply_cmap(SLICE_COLORMAP, n_values=256)
    slice_table.scalar_range = slice_range
    slice_table.nan_opacity = 0.0

    # Rampa de opacidade: o plano desaparece onde não há vorticidade, em vez
    # de recortar um retângulo escuro contra o fundo.
    table_values = slice_table.values
    ramp = np.clip(
        np.linspace(0.0, 1.0, len(table_values)) / SLICE_FADE_FRACTION,
        0.0,
        1.0,
    )
    table_values[:, 3] = (ramp * 255).astype(np.uint8)
    slice_table.values = table_values

    first = frames[0]

    # Campo escalar: opaco e sem iluminação, para que a cor represente
    # exclusivamente o valor medido. O extremo escuro do magma encosta na
    # cor de fundo, então o plano some onde não há vorticidade e só a
    # esteira permanece visível — sem transparência e sem custo.
    slice_actor = plotter.add_mesh(
        first["slice"],
        name="velocity_slice",
        scalars=slice_scalar,
        cmap=SLICE_COLORMAP,
        clim=slice_range,
        lighting=False,
        show_scalar_bar=False,
        reset_camera=False,
    )
    slice_actor.mapper.lookup_table = slice_table

    # Tubos recebem iluminação: é o que dá volume às linhas.
    streamline_actor = plotter.add_mesh(
        first["streamlines"],
        name="streamlines",
        scalars="velocity_magnitude",
        cmap=COLORMAP,
        clim=speed_range,
        smooth_shading=True,
        specular=0.35,
        specular_power=25,
        show_scalar_bar=False,
        reset_camera=False,
    )
    streamline_actor.mapper.lookup_table = lookup_table

    # Superfície de vórtice monocromática: a forma é lida pelo sombreamento,
    # não pela cor, e não compete com a escala de |u|.
    wake_actor = plotter.add_mesh(
        first["wake"],
        name="wake",
        color=WAKE_COLOR,
        smooth_shading=True,
        specular=0.25,
        specular_power=15,
        ambient=0.15,
        diffuse=0.85,
        show_scalar_bar=False,
        reset_camera=False,
    )

    if sphere.n_points:
        plotter.add_mesh(
            sphere,
            name="sphere",
            color=GEOMETRY_COLOR,
            smooth_shading=True,
            specular=0.30,
            specular_power=20,
            ambient=0.20,
            reset_camera=False,
        )

    plotter.enable_lightkit()
    try:
        plotter.enable_anti_aliasing("fxaa")
    except Exception as error:  # noqa: BLE001 - depende do driver
        print(f"Anti-aliasing indisponível: {error}")

    if ENABLE_SSAO:
        try:
            plotter.enable_ssao(
                radius=0.03 * first["slice"].length,
                bias=0.001,
                kernel_size=64,
                blur=True,
            )
        except Exception as error:  # noqa: BLE001 - depende do driver
            print(f"SSAO indisponível: {error}")

    plotter.add_scalar_bar(
        title="|omega| (corte)",
        mapper=slice_actor.mapper,
        n_labels=5,
        fmt="%.1f",
        color=TEXT_COLOR,
        position_x=0.035,
        position_y=0.18,
        width=0.030,
        height=0.58,
        vertical=True,
    )
    if streamline_actor.mapper.GetInput() is not None:
        plotter.add_scalar_bar(
            title="|u| (linhas)",
            mapper=streamline_actor.mapper,
            n_labels=5,
            fmt="%.3f",
            color=TEXT_COLOR,
            position_x=0.935,
            position_y=0.18,
            width=0.030,
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
        f"t = {first['time']:.6f}",
        position=(20, 20),
        font_size=12,
        color=TEXT_COLOR,
    )

    level = metadata["wake_level"]
    if level is None:
        wake_label = "esteira: sem estrutura no intervalo processado"
    elif metadata["wake_mode"] == "q":
        wake_label = (
            f"esteira: Q* = Q/(U/D)² = {metadata['q_star']:g}, "
            f"faixa de {metadata['q_mask_diameters']:g} D junto à parede "
            "excluída"
        )
    elif metadata["wake_mode"] == "reverse":
        wake_label = "bolha de recirculação: superfície u = 0"
    else:
        wake_label = f"esteira: |u| = {level:.2f}"
    plotter.add_text(
        f"dx = {metadata['target_dx']:g} • {wake_label} • projeção paralela",
        position="lower_right",
        font_size=10,
        color=TEXT_COLOR,
    )
    plotter.add_axes(
        xlabel="x", ylabel="y", zlabel="z", color=TEXT_COLOR
    )

    diameter = 2.0 * (metadata["sphere_radius"] or 0.0)
    if metadata["sphere_center"] is not None and diameter > 0.0:
        # Foco deslocado a jusante: o assunto é a esfera mais a esteira
        # próxima, não a esfera centralizada com esteira saindo do quadro.
        focus = np.asarray(metadata["sphere_center"]) + np.array(
            [CAMERA_FOCUS_OFFSET_D * diameter, 0.0, 0.0]
        )
    else:
        focus = np.asarray(first["slice"].center)
    length = first["slice"].length
    # Atribuído campo a campo no objeto Camera. A forma em lista
    # [posição, foco, viewup] não estava aplicando o viewup, e a cena vinha
    # girada 90°: o escoamento descia a tela em vez de atravessá-la.
    camera = plotter.camera
    camera.focal_point = tuple(float(value) for value in focus)
    camera.position = tuple(
        float(value)
        for value in focus + np.asarray(CAMERA_DIRECTION) * length
    )
    camera.up = (0.0, 0.0, 1.0)
    if diameter > 0.0:
        # Em projeção paralela o enquadramento é parallel_scale, e vale
        # metade da altura visível. zoom() sobre um ajuste automático não
        # é reprodutível entre execuções; isto é.
        camera.parallel_scale = 0.5 * CAMERA_FRAME_DIAMETERS * diameter
    plotter.reset_camera_clipping_range()

    state = {"index": 0, "paused": False}

    def show_frame(frame):
        bind_scalars(
            slice_actor.mapper,
            frame["slice"],
            slice_scalar,
            slice_range,
            slice_table,
        )
        bind_scalars(
            streamline_actor.mapper,
            frame["streamlines"],
            "velocity_magnitude",
            speed_range,
            lookup_table,
        )
        wake_actor.mapper.SetInputData(frame["wake"])
        wake_actor.mapper.ScalarVisibilityOff()
        time_actor.SetInput(
            f"t = {frame['time']:.6f}  •  frame "
            f"{state['index'] + 1}/{len(frames)}"
        )

    def advance(*_):
        if state["paused"]:
            return
        state["index"] = (state["index"] + 1) % len(frames)
        show_frame(frames[state["index"]])
        plotter.render()

    def toggle_pause():
        state["paused"] = not state["paused"]

    def step_forward():
        state["index"] = (state["index"] + 1) % len(frames)
        show_frame(frames[state["index"]])
        plotter.render()

    def step_backward():
        state["index"] = (state["index"] - 1) % len(frames)
        show_frame(frames[state["index"]])
        plotter.render()

    plotter.add_key_event("space", toggle_pause)
    plotter.add_key_event("Right", step_forward)
    plotter.add_key_event("Left", step_backward)

    show_frame(first)
    if len(frames) > 1:
        # O laço roda em um timer do VTK; o interactor continua livre para
        # a câmera, ao contrário de um for com time.sleep.
        plotter.add_timer_event(
            max_steps=2_000_000_000,
            duration=FRAME_DURATION_MS,
            callback=advance,
        )
        print("espaço: pausa • setas: frame a frame • q: sair")
    else:
        # Um frame só: sem timer, senão o VTK redesenha a mesma imagem
        # indefinidamente e disputa com a interação de câmera.
        print("frame único • q: sair")
    plotter.show()


# -----------------------------------------------------------------------------
# Renderização de vídeo
# -----------------------------------------------------------------------------

def video_directory() -> Path:
    return scene_directory() / "video"


def load_fields(cache_files):
    """Campos de velocidade de todos os timesteps, em memória."""
    fields, times = [], []
    origin = spacing = None
    for path in cache_files:
        grid = pv.read(path)
        fields.append(as_vector_block(grid))
        times.append(physical_time(grid))
        origin = np.asarray(grid.origin, dtype=np.float32)
        spacing = np.asarray(grid.spacing, dtype=np.float32)
    megabytes = sum(field.nbytes for field in fields) / 1024 ** 2
    print(f"  {len(fields)} campos de velocidade • {megabytes:,.0f} MB")
    return fields, np.asarray(times), origin, spacing


def plane_texture_image(luminance, vorticity, slice_range, table):
    """Combina a textura LIC com a cor do campo escalar.

    A luminância vem do LIC (direção do escoamento) e a matiz vem da
    vorticidade (intensidade). São dois canais independentes carregando
    duas grandezas distintas — nada é inventado.
    """
    normalized = np.clip(
        (vorticity - slice_range[0])
        / max(slice_range[1] - slice_range[0], 1.0e-12),
        0.0,
        1.0,
    )
    indices = (normalized * (len(table) - 1)).astype(np.int32)
    rgba = table[indices].astype(np.float32)

    shade = (1.0 - LIC_CONTRAST) + LIC_CONTRAST * luminance
    rgba[..., :3] *= shade[..., None]
    return np.clip(rgba, 0, 255).astype(np.uint8)


def resample_plane(plane, width, height):
    """Reamostra um plano (nx, nz) para a resolução da textura."""
    rows = np.linspace(0, plane.shape[1] - 1, height, dtype=np.float32)
    columns = np.linspace(0, plane.shape[0] - 1, width, dtype=np.float32)
    mesh_rows, mesh_columns = np.meshgrid(rows, columns, indexing="ij")
    return map_coordinates(
        plane,
        np.stack([mesh_columns.ravel(), mesh_rows.ravel()]),
        order=1,
        mode="nearest",
        prefilter=False,
    ).reshape(height, width)


def prepare_lic(fields, spacing, slice_index, width):
    """Amostras LIC e vorticidade do plano, por timestep de dado."""
    shape = fields[0].shape[:3]
    height = max(2, int(round(width * shape[2] / shape[0])))
    generator = np.random.default_rng(12345)
    noise = generator.random((height, width)).astype(np.float32)

    samples, vorticities = [], []
    for index, field in enumerate(fields):
        jacobian = jacobian_from_block(field, float(spacing[0]))
        vorticity = vorticity_magnitude(jacobian)[:, slice_index, :]
        del jacobian

        velocity_x = resample_plane(
            field[:, slice_index, :, 0], width, height
        )
        velocity_z = resample_plane(
            field[:, slice_index, :, 2], width, height
        )
        samples.append(lic_samples(velocity_x, velocity_z, noise, LIC_STEPS))
        vorticities.append(resample_plane(vorticity, width, height))
        print(
            f"    LIC {index + 1}/{len(fields)} • "
            f"{height}x{width}",
            flush=True,
        )

    total = sum(block.nbytes for block in samples) / 1024 ** 3
    print(f"  amostras LIC em memória: {total:.2f} GB")
    return samples, vorticities, height, width


def render_video(cache_files: list[Path], metadata, preview: int | None):
    destination = video_directory()
    destination.mkdir(parents=True, exist_ok=True)

    print("Carregando campos...")
    fields, times, origin, spacing = load_fields(cache_files)
    shape = fields[0].shape[:3]
    bounds = (
        float(origin[0]),
        float(origin[0] + (shape[0] - 1) * spacing[0]),
        float(origin[1]),
        float(origin[1] + (shape[1] - 1) * spacing[1]),
        float(origin[2]),
        float(origin[2] + (shape[2] - 1) * spacing[2]),
    )

    center = np.asarray(metadata["sphere_center"], dtype=np.float32)
    diameter = 2.0 * float(metadata["sphere_radius"])
    slice_index = int(round((center[1] - origin[1]) / spacing[1]))
    slice_range = tuple(metadata["slice_range"])

    slice_table = pv.LookupTable()
    slice_table.apply_cmap(SLICE_COLORMAP, n_values=256)
    table_values = np.asarray(slice_table.values).copy()
    ramp = np.clip(
        np.linspace(0.0, 1.0, len(table_values)) / SLICE_FADE_FRACTION,
        0.0,
        1.0,
    )
    table_values[:, 3] = (ramp * 255).astype(np.uint8)

    lic_data = None
    if SHOW_LIC:
        print("Pré-calculando LIC (uma vez por timestep de dado)...")
        lic_data = prepare_lic(fields, spacing, slice_index, LIC_WIDTH)

    total_frames = (
        1 if preview is not None else int(VIDEO_FPS * VIDEO_SECONDS)
    )
    time_span = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    dt = time_span / max(total_frames - 1, 1)

    generator = np.random.default_rng(7)
    positions = np.column_stack(
        [
            generator.uniform(bounds[0], bounds[1], PARTICLE_COUNT),
            generator.uniform(bounds[2], bounds[3], PARTICLE_COUNT),
            generator.uniform(bounds[4], bounds[5], PARTICLE_COUNT),
        ]
    ).astype(np.float32)
    ages = generator.uniform(
        0.0, PARTICLE_LIFETIME_SECONDS, PARTICLE_COUNT
    ).astype(np.float32)
    history = np.repeat(positions[None], PARTICLE_TRAIL, axis=0)
    speed_history = np.zeros(
        (PARTICLE_TRAIL, PARTICLE_COUNT), dtype=np.float32
    )
    connectivity = trail_connectivity(PARTICLE_COUNT, PARTICLE_TRAIL)

    plotter = pv.Plotter(off_screen=True, window_size=list(VIDEO_SIZE))
    plotter.set_background(BACKGROUND)
    plotter.enable_parallel_projection()

    plane = pv.PolyData(
        np.array(
            [
                [bounds[0], float(center[1]), bounds[4]],
                [bounds[1], float(center[1]), bounds[4]],
                [bounds[1], float(center[1]), bounds[5]],
                [bounds[0], float(center[1]), bounds[5]],
            ],
            dtype=float,
        ),
        np.array([4, 0, 1, 2, 3]),
    )
    plane.active_texture_coordinates = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )

    sphere = read_optional(scene_directory() / "sphere.vtp")
    if sphere.n_points:
        plotter.add_mesh(
            sphere,
            color=GEOMETRY_COLOR,
            smooth_shading=True,
            specular=0.30,
            specular_power=20,
            ambient=0.18,
            reset_camera=False,
        )

    wake_actor = plotter.add_mesh(
        read_optional(
            scene_directory() / f"wake.{timestep_number(cache_files[0]):09d}.vtp"
        ),
        color=WAKE_COLOR,
        opacity=0.45,
        smooth_shading=True,
        specular=0.25,
        show_scalar_bar=False,
        reset_camera=False,
    )

    tracers = pv.PolyData()
    tracers.points = history.reshape(-1, 3)
    tracers.lines = connectivity
    tracers.point_data["speed"] = speed_history.ravel()
    tracer_actor = plotter.add_mesh(
        tracers,
        scalars="speed",
        cmap=COLORMAP,
        clim=metadata["speed_range"],
        line_width=PARTICLE_SIZE * 0.5,
        opacity=0.85,
        lighting=False,
        show_scalar_bar=False,
        reset_camera=False,
    )

    plane_actor = None
    if SHOW_LIC:
        plane_actor = plotter.add_mesh(
            plane, lighting=False, reset_camera=False
        )

    plotter.enable_lightkit()
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception as error:  # noqa: BLE001 - depende do driver
        print(f"SSAA indisponível: {error}")

    focus = center + np.array(
        [CAMERA_FOCUS_OFFSET_D * diameter, 0.0, 0.0], dtype=np.float32
    )
    base_offset = np.asarray(CAMERA_DIRECTION, dtype=float) * (
        bounds[1] - bounds[0]
    )

    started = time.perf_counter()
    for frame in range(total_frames):
        index = preview if preview is not None else frame
        fraction = index / max(int(VIDEO_FPS * VIDEO_SECONDS) - 1, 1)
        physical = float(times[0]) + fraction * time_span

        upper = int(np.searchsorted(times, physical, side="right"))
        upper = min(max(upper, 1), len(times) - 1)
        lower = upper - 1
        gap = float(times[upper] - times[lower])
        blend = 0.0 if gap <= 0 else (physical - times[lower]) / gap

        def sampler(points, lower=lower, upper=upper, blend=blend):
            low = sample_vectors(fields[lower], origin, spacing, points)
            high = sample_vectors(fields[upper], origin, spacing, points)
            return (1.0 - blend) * low + blend * high

        positions = advect_rk4(positions, sampler, dt)
        ages += dt

        dead = expired(positions, ages, bounds, PARTICLE_LIFETIME_SECONDS)
        if dead.any():
            fresh = spawn_positions(int(dead.sum()), bounds, generator)
            positions[dead] = fresh
            ages[dead] = 0.0
            # Rastro inteiro reposicionado: senão a partícula renascida
            # apareceria ligada por uma linha à sua posição anterior.
            history[:, dead, :] = fresh

        velocity = sampler(positions)
        history = np.roll(history, 1, axis=0)
        history[0] = positions
        speed_history = np.roll(speed_history, 1, axis=0)
        speed_history[0] = np.linalg.norm(velocity, axis=1)

        tracers.points = history.reshape(-1, 3)
        tracers.point_data["speed"] = speed_history.ravel()

        nearest = int(np.argmin(np.abs(times - physical)))
        wake_actor.mapper.SetInputData(
            read_optional(
                scene_directory()
                / f"wake.{timestep_number(cache_files[nearest]):09d}.vtp"
            )
        )
        wake_actor.mapper.ScalarVisibilityOff()

        if SHOW_LIC and plane_actor is not None:
            samples, vorticities, height, width = lic_data
            phase = LIC_TURNS_PER_SECOND * index / VIDEO_FPS
            luminance = (1.0 - blend) * lic_luminance(
                samples[lower], phase, LIC_CYCLES
            ) + blend * lic_luminance(samples[upper], phase, LIC_CYCLES)
            vorticity = (1.0 - blend) * vorticities[lower] + (
                blend * vorticities[upper]
            )
            image = plane_texture_image(
                luminance, vorticity, slice_range, table_values
            )
            if LIC_FLIP_V:
                image = image[::-1]
            plane_actor.SetTexture(pv.Texture(image))

        angle = CAMERA_ORBIT_DEGREES * (fraction - 0.5)
        radians = np.radians(angle)
        cosine, sine = np.cos(radians), np.sin(radians)
        offset = np.array(
            [
                cosine * base_offset[0] - sine * base_offset[1],
                sine * base_offset[0] + cosine * base_offset[1],
                base_offset[2],
            ]
        )
        camera = plotter.camera
        camera.focal_point = tuple(float(value) for value in focus)
        camera.position = tuple(float(value) for value in focus + offset)
        camera.up = (0.0, 0.0, 1.0)
        camera.parallel_scale = 0.5 * CAMERA_FRAME_DIAMETERS * diameter
        plotter.reset_camera_clipping_range()

        output = destination / f"frame_{index:05d}.png"
        plotter.screenshot(str(output))

        if frame % 30 == 0 or frame == total_frames - 1:
            elapsed = time.perf_counter() - started
            rate = (frame + 1) / max(elapsed, 1.0e-9)
            print(
                f"  frame {frame + 1}/{total_frames} • t = {physical:.5f} • "
                f"{rate:.1f} fps • restam "
                f"{(total_frames - frame - 1) / max(rate, 1e-9):.0f} s",
                flush=True,
            )

    plotter.close()
    print(f"\nPNGs em {destination}")
    print(
        f"\nffmpeg -framerate {VIDEO_FPS} -i "
        f"'{destination}/frame_%05d.png' -c:v libx264 -pix_fmt yuv420p "
        f"-crf 16 '{destination}/mflab_esfera_re100.mp4'"
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def resolve_output_dir(explicit: str | None) -> Path:
    path = Path(explicit).expanduser() if explicit else OUTPUT_DIR
    if not path.is_dir():
        raise FileNotFoundError(
            f"{path} não é um diretório. Informe --data."
        )
    return path


def main():
    global OUTPUT_DIR, TARGET_DX, WAKE_MODE, Q_STAR_LEVEL, Q_MASK_DIAMETERS
    global CROP_LEVEL, SHOW_STREAMLINES, SLICE_SCALAR, SPHERE_MODE
    global SHOW_LIC

    parser = argparse.ArgumentParser(
        description="Reconstrói AMR do MFSim e visualiza o escoamento."
    )
    parser.add_argument("--data", help="diretório com os ns_output_ct.*.hdf5")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="HDF5 -> grade uniforme (.vti)",
    )
    parser.add_argument(
        "--geometry",
        action="store_true",
        help=".vti -> geometria da cena (.vtp)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="relatório de validação da reconstrução AMR",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="anima a geometria já calculada",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="renderiza os PNGs do vídeo (partículas + LIC)",
    )
    parser.add_argument(
        "--preview",
        type=int,
        metavar="FRAME",
        help="renderiza só um frame do vídeo, para conferir antes",
    )
    parser.add_argument(
        "--no-lic",
        action="store_true",
        help="desliga a textura LIC no vídeo",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reconstrói caches existentes",
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="processa apenas os N primeiros timesteps",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        help="processa apenas estes timesteps (ex.: --steps 1000 1200)",
    )
    parser.add_argument(
        "--wake",
        choices=("q", "reverse", "speed"),
        help=(
            "estrutura da esteira: q = Q-criterion (escoamento com "
            "desprendimento), reverse = bolha de recirculação (esteira "
            f"estacionária), speed = isosuperfície de |u| (padrão: "
            f"{WAKE_MODE})"
        ),
    )
    parser.add_argument(
        "--q-star",
        type=float,
        help=f"limiar Q/(U/D)² da esteira (padrão: {Q_STAR_LEVEL})",
    )
    parser.add_argument(
        "--mask-d",
        type=float,
        help=(
            "faixa excluída junto ao sólido, em diâmetros "
            f"(padrão: {Q_MASK_DIAMETERS})"
        ),
    )
    parser.add_argument(
        "--no-streamlines",
        action="store_true",
        help="não gera as linhas de corrente",
    )
    parser.add_argument(
        "--slice-scalar",
        choices=("vorticity_magnitude", "velocity_magnitude"),
        help=f"escalar do corte central (padrão: {SLICE_SCALAR})",
    )
    parser.add_argument(
        "--sphere",
        choices=("analytic", "contour"),
        help=f"geometria do corpo imerso (padrão: {SPHERE_MODE})",
    )
    parser.add_argument(
        "--dx",
        type=float,
        help=f"espaçamento alvo (padrão: {TARGET_DX})",
    )
    parser.add_argument(
        "--crop-level",
        type=int,
        help="recorta na caixa com refinamento de nível >= N",
    )
    args = parser.parse_args()

    OUTPUT_DIR = resolve_output_dir(args.data)
    if args.dx is not None:
        TARGET_DX = args.dx
    if args.wake is not None:
        WAKE_MODE = args.wake
    if args.q_star is not None:
        Q_STAR_LEVEL = args.q_star
    if args.mask_d is not None:
        Q_MASK_DIAMETERS = args.mask_d
    if args.crop_level is not None:
        CROP_LEVEL = args.crop_level
    if args.no_streamlines:
        SHOW_STREAMLINES = False
    if args.slice_scalar is not None:
        SLICE_SCALAR = args.slice_scalar
    if args.sphere is not None:
        SPHERE_MODE = args.sphere
    if args.no_lic:
        SHOW_LIC = False

    render_video_requested = args.render or args.preview is not None
    selected = (
        args.preprocess
        or args.geometry
        or args.view
        or args.inspect
        or render_video_requested
    )
    run_preprocess = args.preprocess or not selected
    run_geometry = args.geometry or not selected
    run_view = args.view or not selected

    print(f"Dados: {OUTPUT_DIR}")

    def select(files: list[Path]) -> list[Path]:
        if args.steps:
            wanted = set(args.steps)
            files = [f for f in files if timestep_number(f) in wanted]
            if not files:
                raise FileNotFoundError(
                    f"Nenhum timestep encontrado entre {sorted(wanted)}."
                )
        if args.frames:
            files = files[: args.frames]
        return files

    if run_preprocess:
        preprocess(select(discover_hdf5_files()), force=args.force)

    if args.inspect:
        inspect_cache(select(discover_cache_files()))

    if run_geometry:
        build_scene(select(discover_cache_files()), force=args.force)

    if render_video_requested:
        render_video(
            select(discover_cache_files()), load_scene(), args.preview
        )

    if run_view:
        view(load_scene())


if __name__ == "__main__":
    main()
