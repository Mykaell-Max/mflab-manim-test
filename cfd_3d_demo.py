from manim import *
import numpy as np


BACKGROUND = "#07131F"
PIPE_COLOR = "#4B7896"
PARTICLE_COLOR = "#FFB547"
FLOW_COLOR = "#27C5C3"
TEXT_PRIMARY = "#F1F6F9"
TEXT_SECONDARY = "#9EB2C0"


class MFLabCFD3D(ThreeDScene):
    def construct(self):
        self.camera.background_color = BACKGROUND

        # ---------------------------------------------
        # Câmera 3D
        # ---------------------------------------------

        self.set_camera_orientation(
            phi=68 * DEGREES,
            theta=-45 * DEGREES,
            zoom=0.85,
        )

        # ---------------------------------------------
        # Títulos fixos: não giram com a câmera
        # ---------------------------------------------

        title = Text(
            "Escoamento tridimensional",
            font_size=34,
            weight=SEMIBOLD,
            color=TEXT_PRIMARY,
        ).to_corner(UL, buff=0.4)

        subtitle = Text(
            "Demonstração visual • dados sintéticos",
            font_size=17,
            color=TEXT_SECONDARY,
        ).next_to(title, DOWN, aligned_edge=LEFT, buff=0.08)

        self.add_fixed_in_frame_mobjects(title, subtitle)

        # ---------------------------------------------
        # Eixos
        # ---------------------------------------------

        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-2, 2, 1],
            z_range=[-2, 2, 1],
            x_length=10,
            y_length=4,
            z_length=4,
            axis_config={
                "color": "#547087",
                "stroke_opacity": 0.35,
            },
        )

        # ---------------------------------------------
        # Tubo transparente
        # Eixo principal orientado na direção X
        # ---------------------------------------------

        pipe = Surface(
            lambda u, v: np.array([
                u,
                1.45 * np.cos(v),
                1.45 * np.sin(v),
            ]),
            u_range=[-4.6, 4.6],
            v_range=[0, TAU],
            resolution=(32, 24),
            fill_color=PIPE_COLOR,
            fill_opacity=0.12,
            stroke_color=PIPE_COLOR,
            stroke_opacity=0.28,
            stroke_width=0.7,
        )

        inlet = Circle(
            radius=1.45,
            color=FLOW_COLOR,
            stroke_width=3,
            fill_color=FLOW_COLOR,
            fill_opacity=0.06,
        ).rotate(PI / 2, axis=UP).shift(LEFT * 4.6)

        outlet = Circle(
            radius=1.45,
            color=PARTICLE_COLOR,
            stroke_width=3,
            fill_color=PARTICLE_COLOR,
            fill_opacity=0.04,
        ).rotate(PI / 2, axis=UP).shift(RIGHT * 4.6)

        # ---------------------------------------------
        # Linhas de corrente sintéticas
        # ---------------------------------------------

        streamlines = VGroup()

        streamline_parameters = [
            (0.25, 0.0),
            (0.55, 0.8),
            (0.85, 1.6),
            (1.05, 2.4),
            (0.70, 3.2),
            (0.45, 4.0),
            (0.95, 4.8),
            (0.60, 5.6),
        ]

        for radius, phase in streamline_parameters:
            line = ParametricFunction(
                lambda t, r=radius, p=phase: np.array([
                    t,
                    r * np.cos(0.85 * t + p),
                    r * np.sin(0.85 * t + p),
                ]),
                t_range=[-4.5, 4.5],
                color=FLOW_COLOR,
                stroke_width=2,
                stroke_opacity=0.65,
            )

            streamlines.add(line)

        # ---------------------------------------------
        # Partículas animadas
        # ---------------------------------------------

        simulation_time = ValueTracker(0)
        particles = VGroup()

        particle_data = [
            (0.25, 0.0, 0.0),
            (0.55, 0.8, -0.7),
            (0.85, 1.6, -1.4),
            (1.05, 2.4, -2.1),
            (0.70, 3.2, -2.8),
            (0.45, 4.0, -3.5),
            (0.95, 4.8, -4.2),
            (0.60, 5.6, -4.9),
        ]

        for radius, phase, delay in particle_data:
            particle = Sphere(
                radius=0.09,
                resolution=(8, 8),
                color=PARTICLE_COLOR,
                fill_opacity=1,
            )

            particle.flow_radius = radius
            particle.flow_phase = phase
            particle.delay = delay

            def update_particle(sphere):
                raw_time = simulation_time.get_value() + sphere.delay

                # Faz a partícula retornar à entrada.
                progress = raw_time % 9
                x = -4.5 + progress

                y = sphere.flow_radius * np.cos(
                    0.85 * x + sphere.flow_phase
                )

                z = sphere.flow_radius * np.sin(
                    0.85 * x + sphere.flow_phase
                )

                sphere.move_to([x, y, z])

            particle.add_updater(update_particle)
            particles.add(particle)

        # ---------------------------------------------
        # Indicador temporal fixo
        # ---------------------------------------------

        time_display = always_redraw(
            lambda: Text(
                f"Tempo: {simulation_time.get_value():.2f} s",
                font_size=19,
                color=TEXT_PRIMARY,
            ).to_corner(DL, buff=0.4)
        )

        note = Text(
            "Geometria e trajetórias sintéticas",
            font_size=15,
            color=TEXT_SECONDARY,
        ).to_corner(DR, buff=0.4)

        self.add_fixed_in_frame_mobjects(time_display, note)

        # ---------------------------------------------
        # Animação
        # ---------------------------------------------

        self.play(
            Create(axes),
            Create(inlet),
            Create(outlet),
            run_time=1.2,
        )

        self.play(
            FadeIn(pipe),
            run_time=1.2,
        )

        self.play(
            LaggedStart(
                *[Create(line) for line in streamlines],
                lag_ratio=0.08,
            ),
            run_time=2,
        )

        self.add(particles)

        self.begin_ambient_camera_rotation(
            rate=0.08,
            about="theta",
        )

        self.play(
            simulation_time.animate.set_value(14),
            run_time=14,
            rate_func=linear,
        )

        self.stop_ambient_camera_rotation()
        self.wait()