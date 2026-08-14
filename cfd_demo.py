from manim import *
import numpy as np


# =========================================================
# Tema visual provisório do MFLab
# =========================================================

BACKGROUND = "#07131F"
PANEL = "#0D2133"
BORDER = "#7894A8"
TEXT_PRIMARY = "#F1F6F9"
TEXT_SECONDARY = "#9EB2C0"

COLOR_LOW = "#2457D6"
COLOR_MID = "#27C5C3"
COLOR_HIGH = "#FFB547"


def scalar_field(x, y, t):
    """
    Campo escalar sintético.

    Em uma aplicação real, esta função seria substituída por
    valores lidos de resultados CFD usando NumPy, VTK ou PyVista.
    """
    wave = np.sin(1.25 * x - 1.8 * t)
    transverse = np.cos(1.8 * y + 0.5 * t)
    disturbance = np.exp(-((x - 0.8) ** 2 + (y + 0.2) ** 2))

    value = 0.55 * wave * transverse + 0.45 * disturbance
    return np.clip(value, -1.0, 1.0)


def field_color(value):
    """
    Converte valores no intervalo [-1, 1] para uma escala
    de cores com limites constantes durante toda a animação.
    """
    normalized = (value + 1.0) / 2.0

    if normalized <= 0.5:
        return interpolate_color(
            ManimColor(COLOR_LOW),
            ManimColor(COLOR_MID),
            normalized * 2.0,
        )

    return interpolate_color(
        ManimColor(COLOR_MID),
        ManimColor(COLOR_HIGH),
        (normalized - 0.5) * 2.0,
    )


class MFLabCFDDemo(Scene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        simulation_time = ValueTracker(0.0)

        # -------------------------------------------------
        # Cabeçalho narrativo
        # -------------------------------------------------

        title = Text(
            "Visualização de escoamento",
            font_size=36,
            weight=SEMIBOLD,
            color=TEXT_PRIMARY,
        )

        subtitle = Text(
            "Campo sintético para demonstração visual",
            font_size=18,
            color=TEXT_SECONDARY,
        )

        header = VGroup(title, subtitle)
        header.arrange(DOWN, aligned_edge=LEFT, buff=0.10)
        header.to_corner(UL, buff=0.45)

        # -------------------------------------------------
        # Domínio científico
        # -------------------------------------------------

        domain_width = 10.0
        domain_height = 4.2
        domain_center = DOWN * 0.35 + LEFT * 0.45

        domain = Rectangle(
            width=domain_width,
            height=domain_height,
            stroke_color=BORDER,
            stroke_width=2,
            fill_color=PANEL,
            fill_opacity=1,
        ).move_to(domain_center)

        # Grade usada para representar o campo.
        # Valores modestos para o primeiro teste.
        nx = 32
        ny = 14

        cell_width = domain_width / nx
        cell_height = domain_height / ny

        cells = VGroup()

        left = domain.get_left()[0]
        bottom = domain.get_bottom()[1]

        for iy in range(ny):
            for ix in range(nx):
                local_x = -domain_width / 2 + (ix + 0.5) * cell_width
                local_y = -domain_height / 2 + (iy + 0.5) * cell_height

                cell = Rectangle(
                    width=cell_width * 1.03,
                    height=cell_height * 1.03,
                    stroke_width=0,
                )

                cell.move_to(
                    [
                        left + (ix + 0.5) * cell_width,
                        bottom + (iy + 0.5) * cell_height,
                        0,
                    ]
                )

                # Guardamos a coordenada física correspondente.
                cell.field_x = local_x
                cell.field_y = local_y

                initial_value = scalar_field(local_x, local_y, 0)
                cell.set_fill(field_color(initial_value), opacity=1)

                def update_cell(square, dt):
                    value = scalar_field(
                        square.field_x,
                        square.field_y,
                        simulation_time.get_value(),
                    )
                    square.set_fill(field_color(value), opacity=1)

                cell.add_updater(update_cell)
                cells.add(cell)

        # Mantém a borda visível sobre o campo.
        border = domain.copy()
        border.set_fill(opacity=0)
        border.set_stroke(BORDER, width=2)

        inlet = Text(
            "ENTRADA",
            font_size=14,
            color=TEXT_SECONDARY,
        ).next_to(domain, LEFT, buff=0.16)

        outlet = Text(
            "SAÍDA",
            font_size=14,
            color=TEXT_SECONDARY,
        ).next_to(domain, RIGHT, buff=0.16)

        # -------------------------------------------------
        # Partículas ilustrativas
        # -------------------------------------------------

        particles = VGroup()

        particle_initial_data = [
            (-0.75, -1.35),
            (-0.20, -0.90),
            (-0.55, -0.40),
            (-0.05, 0.05),
            (-0.85, 0.50),
            (-0.35, 0.95),
            (-0.65, 1.40),
        ]

        for offset, initial_y in particle_initial_data:
            particle = Dot(
                radius=0.045,
                color=WHITE,
            )

            particle.time_offset = offset
            particle.initial_y = initial_y

            def update_particle(dot, dt):
                t = max(
                    0.0,
                    simulation_time.get_value() + dot.time_offset,
                )

                physical_x = -domain_width / 2 + 1.25 * t

                physical_y = (
                    dot.initial_y
                    + 0.16 * np.sin(1.5 * physical_x + dot.initial_y)
                )

                screen_x = domain.get_center()[0] + physical_x
                screen_y = domain.get_center()[1] + physical_y

                dot.move_to([screen_x, screen_y, 0])

                visible = (
                    -domain_width / 2 <= physical_x <= domain_width / 2
                )
                dot.set_opacity(1 if visible else 0)

            particle.add_updater(update_particle)
            particles.add(particle)

        # -------------------------------------------------
        # Escala científica fixa
        # -------------------------------------------------

        legend_steps = 28
        legend_height = 3.0
        legend_width = 0.28

        legend_colors = VGroup()

        for i in range(legend_steps):
            normalized = i / (legend_steps - 1)
            value = -1.0 + 2.0 * normalized

            segment = Rectangle(
                width=legend_width,
                height=legend_height / legend_steps * 1.05,
                stroke_width=0,
                fill_color=field_color(value),
                fill_opacity=1,
            )

            legend_colors.add(segment)

        legend_colors.arrange(UP, buff=0)
        legend_colors.next_to(domain, RIGHT, buff=0.65)

        legend_title = Text(
            "u*",
            font_size=22,
            color=TEXT_PRIMARY,
        ).next_to(legend_colors, UP, buff=0.16)

        maximum_label = Text(
            "+1.0",
            font_size=15,
            color=TEXT_SECONDARY,
        ).next_to(legend_colors, RIGHT, buff=0.10).align_to(
            legend_colors, UP
        )

        middle_label = Text(
            "0.0",
            font_size=15,
            color=TEXT_SECONDARY,
        ).next_to(legend_colors, RIGHT, buff=0.10)

        minimum_label = Text(
            "−1.0",
            font_size=15,
            color=TEXT_SECONDARY,
        ).next_to(legend_colors, RIGHT, buff=0.10).align_to(
            legend_colors, DOWN
        )

        legend = VGroup(
            legend_colors,
            legend_title,
            maximum_label,
            middle_label,
            minimum_label,
        )

        # -------------------------------------------------
        # Tempo da simulação
        # -------------------------------------------------

        # Text evita a dependência de LaTeX usada por DecimalNumber.
        time_display = always_redraw(
            lambda: Text(
                f"Tempo: {simulation_time.get_value():.2f} s",
                font_size=20,
                color=TEXT_PRIMARY,
            )
            .next_to(domain, DOWN, buff=0.28)
            .align_to(domain, LEFT)
        )

        disclaimer = Text(
            "Escala fixa em todos os frames • dados sintéticos",
            font_size=15,
            color=TEXT_SECONDARY,
        )

        disclaimer.next_to(domain, DOWN, buff=0.30)
        disclaimer.align_to(domain, RIGHT)

        # -------------------------------------------------
        # Sequência da cena
        # -------------------------------------------------

        self.play(
            FadeIn(header, shift=DOWN * 0.15),
            run_time=0.8,
        )

        self.play(
            Create(domain),
            FadeIn(inlet),
            FadeIn(outlet),
            run_time=0.9,
        )

        self.add(cells)
        self.add(border)

        self.play(
            FadeIn(cells),
            FadeIn(legend),
            FadeIn(time_display),
            FadeIn(disclaimer),
            run_time=1.2,
        )

        self.add(particles)
        self.play(
            FadeIn(particles),
            run_time=0.5,
        )

        self.play(
            simulation_time.animate.set_value(7.5),
            run_time=9,
            rate_func=linear,
        )

        self.wait(0.8)