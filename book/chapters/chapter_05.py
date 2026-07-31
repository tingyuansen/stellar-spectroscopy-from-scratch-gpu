"""Chapter 5: Continuous opacity and scattering."""

from __future__ import annotations

from book.cells import code, markdown, notebook, setup_cell


def build_notebook() -> dict:
    """Construct Chapter 5 from its accepted causal and exact contracts."""

    cells = [
        markdown(
            r"""
            # Continuous Opacity and Scattering

            *Stellar Spectroscopy from Scratch — from physical principles to a working code*

            The spectrum that motivated Chapter 1 contains many narrow dips. Yet even between
            those dips, photons do not pass through the photosphere as though it were empty.
            Chapter 4 has told us which particles are present at every depth, but a population
            alone does not say how strongly those particles interact with light.

            This gives us the question for the chapter:

            > **Between the narrow dips, why is the gas not transparent?**

            The answer will be a *continuum*: absorption and scattering that vary broadly with
            wavelength, sometimes with sharp ionization edges, but without the narrow profile of
            a bound-bound transition. We will build it in four movements. First, one microscopic
            interaction will become opacity per gram. Next, hydrogen, helium, metals, molecules,
            and free electrons will become a named physical budget. We will then place that
            budget on two exact numerical grids. Finally, we will close the atmosphere and
            synthesis products independently.

            We retain the assumptions already made: one-dimensional, static, plane-parallel
            layers in local thermodynamic equilibrium (LTE), with depth index 0 outermost. The
            closed atmosphere and synthesis states from Chapter 4 are supplied as integration
            fixtures here; this chapter consumes them but does not reopen ionization or molecular
            equilibrium. They are controlled six-depth stage inputs rebuilt by the accepted
            atmosphere and synthesis equilibrium routes solely to isolate continuum physics;
            they are not
            evidence that a stellar atmosphere has converged.
            Frequency is in Hz, public wavelength is in nm, and opacity is in
            cm\(^{2}\) g\(^{-1}\).

            The construction contract is:

            | role | chapter object |
            |---|---|
            | reads | one packed atmosphere-continuum state; one separate schema-v4 synthesis handoff; immutable continuum tables |
            | builds | named absorption and scattering components, an absorption-weighted thermal source, two sampling policies, and a later line-reference scale |
            | writes | atmosphere arrays on `(depth, 30000)`; a `(depth, 344)` reference threshold; synthesis arrays on `(depth, wavelength)` |
            | numerical owners | atmosphere: CPU NumPy/Numba `float64`; synthesis: Torch on the resolved device and work dtype |

            No emergent flux or spectrum is computed in this chapter. Opacity tells us how
            matter interacts with a photon; radiative transfer will later tell us which photons
            escape.

            > **Movement I — From one interaction to a light-particle background.** We begin
            > with the units. Only then will a table-driven absorber enter the calculation.
            """
        ),
        setup_cell(),
        code(
            """
            import hashlib
            import os

            import matplotlib.pyplot as plt
            import numpy as np

            from book.chapter05_runtime import (
                CONTINUUM_FIXTURE,
                HMINUS_BOUNDFREE_THRESHOLD_HZ,
                LIGHT_SPEED_NM_PER_S,
                RUNNER_OPACITY_FLAGS,
                SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM,
                atmosphere_component_budget,
                atmosphere_grid_checkpoint,
                continuum_table_preflight,
                edge_reconstruction_checkpoint,
                edge_triplet_checkpoint,
                edge_use_trace_checkpoint,
                hminus_component_checkpoint,
                hminus_edge_checkpoint,
                line_reference_checkpoint,
                metal_population_ownership_checkpoint,
                molecular_continuum_checkpoint,
                numba_timing_checkpoint,
                opacity_scaling_checkpoint,
                run_full_atmosphere_continuum,
                run_sampled_synthesis_continuum,
                run_sampled_synthesis_continuum_at_frequency,
                scattering_checkpoint,
                standard_synthesis_component_checkpoint,
                state_projection_checkpoint,
                stimulated_emission_checkpoint,
                synthesis_stored_h2_invariance_checkpoint,
            )
            from book.plot_style import PAPER_COLORS, add_quiet_grid, single_panel
            from payne_zero_atmosphere.continuum_opacity import (
                LIGHT_SPEED_CM_PER_S_EXACT,
                WAVENUMBER_PER_EV_REFERENCE,
            )
            from payne_zero_synthesis.device import resolve_runtime

            regime_names = (
                "hot_dwarf",
                "solar_dwarf",
                "low_gravity_giant",
                "cool_molecule_rich",
            )
            with np.load(CONTINUUM_FIXTURE, allow_pickle=False) as fixture:
                effective_temperature_by_regime = {
                    name: float(fixture[f"{name}__effective_temperature"])
                    for name in regime_names
                }
                hminus_edge_frequency_hz = float(
                    fixture["hminus_edge_frequency_hz"][1]
                )
                extra_edge_probe_wavelength_nm = np.asarray(
                    fixture["synthesis_edge_probe_wavelength_nm"],
                    dtype=np.float64,
                )
                solar_edge_wavelength_nm = np.asarray(
                    fixture[
                        "solar_dwarf__synthesis__continuum_edge_wavelength_nm"
                    ],
                    dtype=np.float64,
                )
                solar_edge_midpoint_wavelength_nm = np.asarray(
                    fixture[
                        "solar_dwarf__synthesis__"
                        "continuum_edge_midpoint_wavelength_nm"
                    ],
                    dtype=np.float64,
                )

            threshold_centres_hz = np.asarray(
                [
                    hminus_edge_frequency_hz,
                    LIGHT_SPEED_CM_PER_S_EXACT * 20_000.0,
                    LIGHT_SPEED_CM_PER_S_EXACT * WAVENUMBER_PER_EV_REFERENCE * 2.0,
                    LIGHT_SPEED_CM_PER_S_EXACT * WAVENUMBER_PER_EV_REFERENCE * 10.5,
                    LIGHT_SPEED_CM_PER_S_EXACT * WAVENUMBER_PER_EV_REFERENCE * 2.1,
                    LIGHT_SPEED_CM_PER_S_EXACT * WAVENUMBER_PER_EV_REFERENCE * 15.0,
                    3.28805e15,
                    2.463e15,
                    2.922e15,
                ],
                dtype=np.float64,
            )
            comparison_frequency_hz = np.concatenate(
                [
                    np.asarray(
                        [
                            np.nextafter(value, -np.inf),
                            value,
                            np.nextafter(value, np.inf),
                        ]
                    )
                    for value in threshold_centres_hz
                ]
            )
            comparison_edge_intervals = np.asarray(
                [50, 75, 100, 125, 150, 170, 200, 225, 250, 275, 300],
                dtype=np.int64,
            )
            comparison_wavelength_nm = np.concatenate(
                [
                    np.asarray(
                        [
                            np.nextafter(
                                solar_edge_wavelength_nm[index],
                                solar_edge_wavelength_nm[index + 1],
                            ),
                            solar_edge_midpoint_wavelength_nm[index],
                            np.nextafter(
                                solar_edge_wavelength_nm[index + 1],
                                solar_edge_wavelength_nm[index],
                            ),
                        ]
                    )
                    for index in comparison_edge_intervals
                ]
                + [extra_edge_probe_wavelength_nm]
            )
            reader_golden_path = (
                repository_root
                / "data"
                / "golden"
                / "payne_zero"
                / "chapter05"
                / "chapter05_continuum_reader_cpu_float64.npz"
            )
            synthesis_golden_member_names = tuple(
                [
                    f"synthesis__{route}__{name}"
                    for route in ("diagnostic", "extension")
                    for name in ("absorption", "scattering", "source")
                ]
                + [
                    "synthesis__standard__absorption",
                    "synthesis__standard__scattering",
                    "synthesis__extension__wavelength_nm",
                ]
            )

            def _load_reader_golden_members(*names):
                if not reader_golden_path.is_file():
                    raise FileNotFoundError(
                        "Chapter 5 reader golden has not been published after "
                        "independent publisher acceptance: "
                        f"{reader_golden_path}"
                    )
                with np.load(reader_golden_path, allow_pickle=False) as archive:
                    return {name: np.asarray(archive[name]).copy() for name in names}

            def _show_one_panel(
                x,
                curves,
                *,
                xlabel,
                ylabel,
                title,
                xscale=None,
                yscale=None,
                vertical=None,
                horizontal=None,
                points=(),
                legend_columns=1,
                ylim=None,
            ):
                figure, axes = single_panel()
                for label, values, colour, linestyle, width in curves:
                    displayed = np.asarray(values)
                    if yscale == "log":
                        displayed = np.where(displayed > 0.0, displayed, np.nan)
                    axes.plot(
                        x,
                        displayed,
                        color=colour,
                        linestyle=linestyle,
                        linewidth=width,
                        label=label,
                    )
                for label, point_x, point_y, colour in points:
                    axes.scatter(
                        point_x,
                        point_y,
                        color=colour,
                        edgecolor="white",
                        linewidth=0.7,
                        zorder=4,
                        label=label,
                    )
                if vertical is not None:
                    axes.axvline(
                        vertical[0],
                        color=PAPER_COLORS["grey"],
                        linestyle=":",
                        label=vertical[1],
                    )
                if horizontal is not None:
                    axes.axhline(
                        horizontal[0],
                        color=PAPER_COLORS["slate"],
                        linestyle=":",
                        label=horizontal[1],
                    )
                if xscale is not None:
                    axes.set_xscale(xscale)
                if yscale is not None:
                    axes.set_yscale(yscale)
                if ylim is not None:
                    axes.set_ylim(*ylim)
                axes.set_xlabel(xlabel)
                axes.set_ylabel(ylabel)
                axes.set_title(title)
                add_quiet_grid(axes, axis="y" if yscale is None else "both")
                axes.legend(ncol=legend_columns)
                plt.show()
                plt.close(figure)
            """,
            tags=("book-setup", "hide-input"),
        ),
        markdown(
            r"""
            ## 5.1 From one particle to one gram

            A **cross section** \(\sigma_\nu\) is an effective interaction area for one
            particle at frequency \(\nu\). It has units cm\(^{2}\). If there are \(n\)
            absorbers per cm\(^{3}\), then a very thin path \(ds\) has interaction probability

            \[
            n\sigma_\nu ds .
            \]

            The product is dimensionless: cm\(^{-3}\) \(\times\) cm\(^{2}\) \(\times\) cm.
            Removing the path length gives the absorption coefficient per distance,
            \(\alpha_\nu=n\sigma_\nu\), in cm\(^{-1}\). Dividing by the local mass density
            \(\rho\) then describes the interaction strength of one gram of gas:
            """
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch05-cross-section-to-opacity-v2.png"
                   alt="Conceptual hand-sketched chain from a microscopic cross section through absorber number density to absorption per length, followed by division by mass density to obtain mass opacity.">
              <figcaption><strong>Conceptual schematic: microscopic area to mass
              opacity.</strong> The thin-slice probability is the separate dimensionless
              product \(n\sigma_\nu ds\). Removing \(ds\) gives
              \(n\,[\mathrm{cm}^{-3}]\times\sigma_\nu\,[\mathrm{cm}^{2}]
              =\alpha_\nu\,[\mathrm{cm}^{-1}]\); only then does division by
              \(\rho\,[\mathrm{g\,cm}^{-3}]\) give
              \(\kappa_\nu\,[\mathrm{cm}^{2}\,\mathrm{g}^{-1}]\). The drawn particle sizes
              are explanatory, not measured.</figcaption>
            </figure>

            \[
            \boxed{\kappa_\nu=\frac{\alpha_\nu}{\rho}
                         =\frac{n\sigma_\nu}{\rho}}
            \qquad [\mathrm{cm}^{2}\,\mathrm{g}^{-1}].
            \]

            This equation makes two predictions before any detailed physics is known:
            doubling the absorber population doubles its opacity, while doubling the mass
            density halves it. The following cell reads two positive number densities,
            cross sections, and mass densities, all as CPU NumPy `float64` arrays. It writes
            two mass-opacity values and the two predicted ratios.
            """
        ),
        code(
            """
            scaling = opacity_scaling_checkpoint()

            print("baseline opacity [cm^2 g^-1]:",
                  np.array2string(scaling.baseline_cm2_per_g, precision=6))
            print("population doubled / baseline:",
                  scaling.population_ratio)
            print("mass density doubled / baseline:",
                  scaling.density_ratio)
            print(
                "first value from n sigma / rho:",
                f"{scaling.baseline_cm2_per_g[0]:.6e} cm^2 g^-1",
            )
            """,
        ),
        markdown(
            r"""
            The live ratios are 2 and \(1/2\), exactly as the units predicted. This check
            establishes the conversion law, not the cross section of any particular atom.
            We now need to ask how a photon exchanges energy with matter.

            In **bound-free absorption**, a photon supplies enough energy to free a bound
            electron. The contribution is zero on the inactive side of its threshold. In
            **free-free absorption**, an already free electron gains energy while being
            accelerated by an ion; there is no one bound-state threshold. A free-free formula
            commonly contains a **Gaunt factor**, a dimensionless quantum correction to the
            classical calculation. It changes the cross section, not the particle population.

            ## 5.2 Net absorption is not the same as redirection

            Gross absorption is not yet the LTE answer. The same radiation field can stimulate
            the reverse emission process. Detailed balance leaves the net factor

            \[
            x=\frac{h\nu}{kT},
            \qquad
            s_\nu(T)=1-\exp(-x).
            \]

            At low photon energy, \(x\ll1\), the factor approaches \(x\); at high energy it
            approaches one. Directly subtracting two nearly equal floating-point numbers is
            inaccurate when \(x\) is small, so we evaluate
            \(s_\nu=-\operatorname{expm1}(-x)\). This cell writes one dimensionless curve and
            compares it with both limits.
            """
        ),
        code(
            """
            energy_ratio = np.geomspace(1.0e-6, 30.0, 400)
            stimulated = stimulated_emission_checkpoint(energy_ratio)
            low = energy_ratio <= 0.08
            low_limit = np.full_like(energy_ratio, np.nan)
            low_limit[low] = stimulated.low_energy_approximation[low]
            _show_one_panel(
                energy_ratio,
                (
                    (r"$1-e^{-x}$", stimulated.factor,
                     PAPER_COLORS["blue"], "-", 1.8),
                    (r"low-energy limit $x$", low_limit,
                     PAPER_COLORS["orange"], "--", 1.8),
                ),
                xlabel=r"Photon energy relative to thermal energy, $x=h\\nu/kT$",
                ylabel=r"Net LTE absorption factor, $s_\\nu$",
                title="One stable expression reaches both physical limits",
                xscale="log",
                horizontal=(1.0, "high-energy limit"),
                ylim=(-0.02, 1.07),
            )

            print("small-x ratio s/x:", stimulated.factor[0] / energy_ratio[0])
            print("large-x distance from one:", 1.0 - stimulated.factor[-1])
            """,
        ),
        markdown(
            r"""
            The curve follows \(s_\nu\simeq x\) at the left and saturates at one on the
            right, while remaining between zero and one. This does **not** mean that every
            fitted opacity should be multiplied by a new copy of \(s_\nu\). Some tabulated
            coefficients already encode an equivalent convention; each named component must
            follow its own exact formula once.

            A second distinction is equally important. Absorption transfers photon energy to
            matter. Scattering removes a photon from one ray but redirects it rather than
            thermalizing it.
            """
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch05-absorption-vs-scattering-v3.png"
                   alt="Conceptual hand-sketched fork in which an incoming photon is either absorbed and transfers energy to matter or is scattered into another direction.">
              <figcaption><strong>Conceptual schematic: two bookkeeping
              consequences.</strong> Absorption both contributes to extinction and carries a
              thermal source contribution. Scattering contributes to extinction by changing
              direction, but not to the LTE thermal numerator. The source numerator records
              what absorptive opacity emits; it does not create or cause the opacity shown in
              the other branch.</figcaption>
            </figure>

            We therefore keep three arrays distinct:

            \[
            \kappa_\nu^{\rm abs},\qquad
            \kappa_\nu^{\rm sca},\qquad
            N_\nu=\sum_j\kappa_{\nu,j}^{\rm abs}S_{\nu,j}.
            \]

            Chapter 1 introduced \(B_\nu(T)\), the LTE intensity scale per frequency, in
            erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\). Most absorptive continuum
            terms have \(S_{\nu,j}=B_\nu\); hydrogen and H-minus may carry
            departure-coefficient corrections. Because
            \(\kappa_{\nu,j}^{\rm abs}\) is in cm\(^{2}\) g\(^{-1}\), the numerator
            \(N_\nu\) has units
            \((\mathrm{cm}^{2}\,\mathrm{g}^{-1})
            (\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,
            \mathrm{sr}^{-1}\,\mathrm{Hz}^{-1})\). The quotient

            \[
            S_\nu^{\rm cont}=
            \frac{N_\nu}{\kappa_\nu^{\rm abs}}
            \]

            restores the \(B_\nu\) unit, so `continuum_source` is also in
            erg s\(^{-1}\) cm\(^{-2}\) sr\(^{-1}\) Hz\(^{-1}\). The exact atmosphere
            composite uses \(B_\nu\) if absorption is zero. Scattering remains outside
            \(N_\nu\). Only now have we earned the aggregate names `continuum_absorption`,
            `continuum_scattering`, and, for the atmosphere product, `continuum_source`.

            ## 5.3 H-minus: the first complete absorber

            Neutral hydrogen can weakly bind one extra electron to form H\(^{-}\). Although
            H\(^{-}\) is fragile, neutral hydrogen is abundant and photospheric electrons are
            available, so its continuum can be important. The closed equilibrium state did not
            store an H-minus population. The continuum consumer instead forms a local LTE
            factor from
            `temperature`, `electron_density`, and
            `hydrogen_partition_normalized_ion_stage_populations[:, 0]`, which means
            \(n_{\rm H\,I}/U_{\rm H\,I}\), not the actual neutral-hydrogen density.

            In the exact cgs convention, the locally inferred population is

            \[
            n_{\rm H^-}^{\rm local}
            =\frac{b_-b_1}{2(2.4148\times10^{15})T^{3/2}}
            \exp\!\left(\frac{0.754209\ {\rm eV}}{kT}\right)
            n_e\frac{n_{\rm H\,I}}{U_{\rm H\,I}} .
            \]

            The numerical prefactor carries the cgs conversion, so the result is in
            cm\(^{-3}\). Both \(b_-\), for H-minus, and \(b_1\), for the H I ground state,
            are dimensionless departure coefficients; each is one in LTE. The same
            0.754209-eV binding energy sets the strict bound-free threshold through
            \(h\nu_0=0.754209\) eV.

            A **departure coefficient** multiplies the LTE reference population or level
            relation; a value of one is the LTE case. This local construction is a consumer
            policy, not a second ionization solve.

            Before the first table enters the calculation, we verify the immutable archives
            against the data manifest. The cell reads file bytes and metadata but writes no
            scientific array. It also demonstrates in memory that one altered byte would fail
            the identity check.
            """
        ),
        code(
            """
            table_identity = continuum_table_preflight()
            first_path = repository_root / table_identity.relative_paths[0]
            altered_digest = hashlib.sha256(
                first_path.read_bytes() + b"changed"
            ).hexdigest()

            print("role                         SHA-256 prefix   verified")
            for role, digest, verified in zip(
                table_identity.roles,
                table_identity.sha256,
                table_identity.manifest_verified,
            ):
                print(f"{role:28s} {digest[:12]}        {verified}")
            print("H- bound-free table shapes:",
                  table_identity.hminus_boundfree_wavelength_shape,
                  table_identity.hminus_boundfree_cross_section_shape)
            print("H- free-free inverse-wavelength / theta axes:",
                  table_identity.hminus_freefree_wavelength_shape,
                  table_identity.hminus_freefree_temperature_shape)
            print("stored bound-free convention:",
                  table_identity.hminus_stored_unit)
            print("one altered byte accepted:", altered_digest == table_identity.sha256[0])
            """,
        ),
        markdown(
            r"""
            All five archives match their declared hashes, while the in-memory altered digest
            does not. The first bound-free array contains 85 stored numbers. Despite the field
            spelling `hminus_boundfree_cross_section_cm2`, those numbers are in units of
            \(10^{-18}\) cm\(^{2}\), so the physical cross section requires an explicit
            `1e-18` conversion. The free-free table has a 22-point inverse-wavelength
            coordinate and an 11-point \(\theta=5040/T\) coordinate. Neither stored axis is
            simply wavelength or temperature.

            The exact kernels intentionally retain two constant tiers. CODATA-exact
            \(c\), \(h\), and \(k\) literals own Planck evaluation and grid conversion;
            inherited rounded reference literals own the H\(_2\), Si II, and table-fit
            formulae that were calibrated with them. Substituting one tier for the other can
            move a boundary or a last bit, so constant choice is part of the algorithm.

            The H-minus bound-free term is active only above
            \(\nu_0=1.82365\times10^{14}\) Hz, or to the blue of the corresponding wavelength.
            Its free-free companion has no such edge. The next cell reads one six-depth
            solar-like fixture and writes separate `(depth, wavelength)` bound-free,
            free-free, and ordered-sum arrays in cm\(^{2}\) g\(^{-1}\). Depth index 3 is
            declared before plotting; no comparison output chooses it.
            """
        ),
        code(
            """
            hminus_edge = hminus_edge_checkpoint("solar_dwarf")
            hminus_wavelength_nm = np.geomspace(800.0, 2_200.0, 280)
            hminus = hminus_component_checkpoint("solar_dwarf", hminus_wavelength_nm)
            hminus_depth = 3
            edge_wavelength_nm = LIGHT_SPEED_NM_PER_S / HMINUS_BOUNDFREE_THRESHOLD_HZ
            _show_one_panel(
                hminus_wavelength_nm,
                (
                    (r"H$^-$ bound-free", hminus.boundfree_absorption_cm2_per_g[
                        hminus_depth], PAPER_COLORS["blue"], "-", 1.8),
                    (r"H$^-$ free-free", hminus.freefree_absorption_cm2_per_g[
                        hminus_depth], PAPER_COLORS["orange"], "-", 1.8),
                    ("ordered sum", hminus.total_absorption_cm2_per_g[
                        hminus_depth], PAPER_COLORS["black"], "-", 2.2),
                ),
                xlabel="Wavelength [nm]",
                ylabel=r"H$^-$ absorption [cm$^2$ g$^{-1}$]",
                title="A thresholded branch sits above a smooth companion",
                yscale="log",
                vertical=(edge_wavelength_nm, "bound-free edge"),
            )

            print("bound-free maxima below / at / above frequency edge:",
                  np.max(hminus_edge.boundfree_absorption_cm2_per_g, axis=0))
            print("explicit 1e-18 reconstruction matches the LTE exact branch:",
                  np.allclose(
                      hminus.transparent_unit_boundfree_absorption_cm2_per_g,
                      hminus.exact_unit_boundfree_absorption_cm2_per_g,
                      rtol=2.0e-12, atol=1.0e-20,
                  ))
            """,
        ),
        markdown(
            r"""
            The bound-free contribution is zero below and exactly at the strict frequency
            threshold, then becomes active just above it. In wavelength order that means it is
            present on the blue, shorter-wavelength side of the vertical line. The free-free
            curve remains smooth through the same location. Small structure within the active
            branch comes from interpolation in the measured/fitted table; the discontinuous
            change of activity comes from the physical threshold.

            Bound-free and free-free retain their own stored-coefficient conventions. The
            verified ordered sum is therefore not formed by applying an extra generic
            stimulated-emission factor to both completed arrays.

            ## 5.4 Hydrogen and helium change with thermal regime

            H-minus is not the whole light-particle background. A bound-state sum reads a
            population divided by its partition function,

            \[
            \widetilde n_r=\frac{n_r}{U_r(T)},\qquad
            n_i=\widetilde n_r g_i e^{-E_i/kT},
            \]

            while a free-free term reads actual electrons and actual charged ions. The
            partition function \(U_r\) is dimensionless, so both \(n_r\) and
            \(\widetilde n_r\) retain units cm\(^{-3}\), but they are not interchangeable.

            The physical budget grows for a reason. Neutral H supplies explicit bound-free
            levels and a high-level tail. Ionized H supplies a charge-one free-free partner.
            H\(_2^+\) adds an ionic absorption law. He I and He II supply bound-free terms and
            high-level tails. Their free-free partners are the next charge stages, He II and
            He III, respectively; bare He III has no bound levels. He\(^{-}\) is a fitted
            free-free-like interaction involving neutral helium and electrons. H\(_2^+\) is
            not the H\(_2\) collision-induced absorption introduced later.

            We compare the same grouped light-particle budget in one solar-like and one hot
            fixture. Each exact component has shape `(6, 220)`; the plot uses depth index 3
            and absolute mass opacity, with colour identifying the physical group and line
            style identifying the regime.
            """
        ),
        code(
            """
            light_wavelength_nm = np.geomspace(120.0, 2_500.0, 220)
            light_frequency_hz = LIGHT_SPEED_NM_PER_S / light_wavelength_nm
            light_budgets = {
                regime: atmosphere_component_budget(regime, light_frequency_hz)
                for regime in ("solar_dwarf", "hot_dwarf")}
            light_depth = 3
            other_light = ("h2plus", "helium_neutral", "helium_ionized", "heminus")
            groups = (
                ("H-minus", ("hminus",), PAPER_COLORS["blue"]),
                ("H I / H II", ("hydrogen",), PAPER_COLORS["orange"]),
                (r"$\\mathrm{H_2^+}$ and helium", other_light, PAPER_COLORS["green"]),
            )
            light_curves, grouped_opacity = [], {}
            for regime, linestyle in (("solar_dwarf", "-"), ("hot_dwarf", "--")):
                budget = light_budgets[regime]
                for group, members, colour in groups:
                    indices = [budget.component_names.index(name) for name in members]
                    components = budget.component_absorption_cm2_per_g
                    values = np.sum(components[indices, light_depth], axis=0)
                    grouped_opacity[regime, group] = values
                    light_curves.append(
                        (f"{group}; {regime.replace('_', ' ')}", values, colour,
                         linestyle, 1.8))
            _show_one_panel(
                light_wavelength_nm, light_curves,
                xlabel="Wavelength [nm]",
                ylabel=r"Grouped absorption [cm$^2$ g$^{-1}$]",
                title="Ionization state changes the light-particle background",
                xscale="log", yscale="log", legend_columns=2)
            sample_index = int(np.argmin(np.abs(light_wavelength_nm - 500.0)))
            for regime in light_budgets:
                values = [grouped_opacity[regime, group][sample_index]
                          for group, _, _ in groups]
                print(regime, "largest 500-nm group:", groups[int(np.argmax(values))][0])
            """,
        ),
        markdown(
            r"""
            The solid and dashed families do not preserve the same ordering: changing the
            thermal and ionization state changes which particles are available to own the
            continuum. That is the physical point of the comparison. It does not claim that
            every layer of every hot or cool atmosphere has the same dominant process, and it
            still omits metals and molecular continua.

            > **Movement II — Complete the physical budget without hiding population
            > ownership.** The light-particle sum leaves broad frequency regions unexplained.
            > We next add bound electrons in metals, molecular photodissociation and
            > collision-induced absorption, and redirection by free and neutral particles.

            ## 5.5 Metals fill a different part of the background

            “Metal continuum” is not one constant. Neutral C, Mg, Al, Si, and Fe have
            photoionization thresholds. Warmer ions add further bound-free terms, while
            charged metal ions also act as Coulomb partners for free-free absorption.

            These two roles read different closed-state arrays:

            \[
            \kappa_\nu^{\rm metal,bf}\ \hbox{reads}\ \widetilde n_{Z,r},
            \qquad
            \kappa_\nu^{\rm metal,ff}\ \hbox{reads}\
            Q_Z=\sum_r q_r^2 n_{Z,r}.
            \]

            Here \(q_r\) is the ion charge. The first expression needs a bound-state base
            divided by its partition function; the second needs actual ionic densities.
            The next calculation makes controlled copies of one fixture and doubles one view
            at a time. Those copies are sensitivity diagnostics, not chemically
            self-consistent atmospheres.
            """
        ),
        code(
            """
            metal_ownership = metal_population_ownership_checkpoint("hot_dwarf")
            solar_budget = light_budgets["solar_dwarf"]
            metal_names = (
                "carbon_neutral",
                "magnesium_neutral",
                "aluminum_neutral",
                "silicon_neutral",
                "iron_neutral",
                "lukewarm_metals",
                "hot_metals",
            )

            normalized_error = np.nanmax(
                np.abs(metal_ownership.normalized_ratio - 2.0)
            )
            actual_error = np.nanmax(
                np.abs(metal_ownership.actual_ratio - 2.0)
            )
            print("normalized-only bound-free doubling error:", normalized_error)
            print("actual-only free-free doubling error:", actual_error)
            print("normalized perturbation preserves charge-square sum:",
                  metal_ownership.normalized_perturbation_preserves_charge_square)
            print("actual perturbation preserves normalized hot-metal array:",
                  metal_ownership.actual_perturbation_preserves_hot_metal_populations)
            print("named atmosphere metal maxima at the selected depth")
            for name in metal_names:
                index = solar_budget.component_names.index(name)
                maximum = np.max(
                    solar_budget.component_absorption_cm2_per_g[index, light_depth]
                )
                print(f"  {name:20s} {maximum:.6e} cm^2 g^-1")
            """,
        ),
        markdown(
            r"""
            Each owned contribution doubles to floating-point accuracy, while the opposite
            population view remains unchanged. This is why
            `partition_normalized_populations` and `ion_stage_populations` cannot be replaced
            by one generic “population” array. At the synthesis boundary, `build_pops`
            materializes 21 normalized hot-metal columns and five charge-square sums from
            those two owners.

            The exact consumers share the physics but not one blended approximation:

            | physical family | atmosphere product | standard synthesis product |
            |---|---|---|
            | neutral C/Mg/Al/Si | full level families | compact branches |
            | neutral Fe | separate full family | no separate standard term |
            | N I, O I, C II, Mg II, Si II, Ca II | atmosphere “lukewarm” group | only the standard Si II Peach branch from this group |
            | hot bound-free and ionic free-free | ordered full group | ordered hot-metal group |

            The sampled diagnostic already evaluates the full C I, Mg I, Al I, Si I, and Fe I
            fallback formulae one frequency at a time with Coulomb layout `True`. The
            precomputed extension materializes those neutral-metal helpers as `(depth,
            frequency)` grids and additionally activates explicit N I, O I, Mg II, and Ca II
            grids. Neither alternative is inserted into the standard synthesis product.
            Si II is a useful warning: the atmosphere generates its own cached table, whereas
            synthesis reads a packaged Peach table. The 60-record hot bound-free sum also
            retains its source order and running one-percent acceptance rule. Reassociating
            that sum would change the algorithm, even though algebra calls addition
            commutative.

            ## 5.6 Molecular continua have exact consumer boundaries

            A molecule can absorb without producing a narrow rotational-vibrational line.
            A continuum photon can photodissociate CH or OH. A collision can also give an
            H\(_2\)-H\(_2\) or H\(_2\)-He pair a temporary dipole, producing
            **collision-induced absorption** (CIA).

            We do not repeat the equilibrium solve. The atmosphere adapter supplies
            `ch_population` and `oh_population`, but those exact names are aliases for
            \(N_{\rm CH}/U_{\rm CH}\) and \(N_{\rm OH}/U_{\rm OH}\) in packed slots 845 and
            847. Their opacity helpers restore the corresponding partition factors.

            H\(_2\) follows a different policy. This consumer does not read the handed-off
            `hydrogen_partition_normalized_ion_stage_populations[:, 0]` used by H-minus.
            Instead it starts from the actual `hydrogen_neutral_population` and recomputes a
            local base with the continuum's six-level hydrogen partition function,

            \[
            \widetilde n_{\rm H\,I}^{\,{\rm local}}
            =\frac{n_{\rm H\,I}}{U_{\rm H}^{(6)}(T)}.
            \]

            The ground-state statistical weight 2 and the dimensionless H-ground departure
            coefficient \(b_1\) then produce the actual local molecular density:

            \[
            n_{\rm H_2}
            =\left(2b_1\widetilde n_{\rm H\,I}^{\,{\rm local}}\right)^2
            K_{\rm H_2}(T).
            \]

            Here \(K_{\rm H_2}(T)\) is the atmosphere table-derived association factor in
            cm\(^{3}\). Thus \((\mathrm{cm}^{-3})^2\mathrm{cm}^{3}\) returns the actual
            \(n_{\rm H_2}\) in cm\(^{-3}\).

            That value feeds H\(_2\)-H\(_2\) and H\(_2\)-He CIA and atmosphere H\(_2\)
            Rayleigh scattering. It is not the stored schema field
            `molecular_hydrogen_population`.

            This cell reads a cool six-depth fixture and writes four separate molecular
            contributions on a declared 180-point wavenumber grid. The one-panel comparison
            uses depth index 4. It shows the 8999/9000 K CH/OH activity boundary and separately
            perturbs the stored synthesis H\(_2\) field by \(10^6\).
            """
        ),
        code(
            """
            molecular_wavenumber_cm1 = np.geomspace(2_500.0, 30_000.0, 180)
            molecular = molecular_continuum_checkpoint(
                "cool_molecule_rich", molecular_wavenumber_cm1)
            stored_h2 = synthesis_stored_h2_invariance_checkpoint("cool_molecule_rich")
            wavelength_order = np.argsort(molecular.wavelength_nm)
            components = (
                ("CH photodissociation", molecular.ch_absorption_cm2_per_g,
                 PAPER_COLORS["blue"]),
                ("OH photodissociation", molecular.oh_absorption_cm2_per_g,
                 PAPER_COLORS["orange"]),
                (r"$\\mathrm{H_2}$ collision-induced",
                 molecular.collision_induced_absorption_cm2_per_g,
                 PAPER_COLORS["green"]),
            )
            molecular_curves = tuple(
                (label, values[4, wavelength_order], colour, "-", 1.8)
                for label, values, colour in components
            )
            _show_one_panel(
                molecular.wavelength_nm[wavelength_order],
                molecular_curves,
                xlabel="Wavelength [nm]",
                ylabel=r"Molecular absorption [cm$^2$ g$^{-1}$]",
                title="A cool atmosphere contains genuine molecular continua",
                xscale="log",
                yscale="log")
            print("CH zero at 9000 K:",
                  molecular.ch_at_temperature_gate_cm2_per_g[1] == 0.0)
            print("OH zero at 9000 K:",
                  molecular.oh_at_temperature_gate_cm2_per_g[1] == 0.0)
            print("local H2 retained at 20000 K and zero above:",
                  molecular.h2_rayleigh_at_cutoff_cm2_per_g[1] > 0.0,
                  molecular.h2_rayleigh_at_cutoff_cm2_per_g[2] == 0.0)
            print("stored synthesis H2 x 1e6 leaves standard continuum unchanged:",
                  stored_h2.bitwise_equal)
            """,
        ),
        markdown(
            r"""
            The three curves show that “molecular opacity” is not synonymous with a line
            list: photodissociation and collision-induced absorption create broad components.
            The CH and OH terms switch off per layer at 9000 K. The local H\(_2\) consumer
            remains active at exactly 20000 K and becomes zero only above it. The underlying
            H\(_2\) equilibrium-table interpolation clips at 100 and 19900 K; CIA is active
            through 20000 cm\(^{-1}\), and its two temperature-table weights retain their
            pinned lower/upper ordering.

            The atmosphere enters its molecular composite only when
            `min(temperature) < 9000 K`. In a mixed column, CH and OH then become zero
            layer-by-layer at \(T\geq9000\) K, while continuum-local H\(_2\) CIA can remain
            through 20000 K. An all-warm column skips the whole composite.
            `enable_molecules=False` normally zeros the packed CH/OH inputs; it does not
            disable the continuum-local H\(_2\) reconstruction, so H\(_2\) CIA and Rayleigh
            scattering may remain. Focused tests cover both column gates and their adjacent
            boundary nodes.

            The stored synthesis H\(_2\) perturbation changes neither standard absorption nor
            standard scattering. Synthesis reconstructs its own H\(_2\) Rayleigh contribution
            from neutral H. The route boundary is therefore:

            | process | atmosphere product | standard synthesis product |
            |---|---:|---:|
            | CH and OH continuum | yes | no |
            | H\(_2\)-H\(_2\) / H\(_2\)-He CIA | yes | no |
            | H\(_2^+\) absorption | yes | yes |
            | H\(_2\) Rayleigh | yes, atmosphere-local H\(_2\) | yes, synthesis-local H\(_2\) |

            This asymmetry is a declared model boundary, not a missing term to be repaired for
            visual symmetry.

            ## 5.7 Scattering remains a separate slab

            A free electron redirects light through Thomson scattering:

            \[
            \kappa_{\rm e}^{\rm sca}
            =\sigma_{\rm T}\frac{n_e}{\rho},
            \qquad
            \sigma_{\rm T}=0.6653\times10^{-24}\ {\rm cm}^{2}.
            \]

            At fixed \(n_e/\rho\), this opacity is independent of wavelength. Neutral H,
            neutral He, and H\(_2\) undergo Rayleigh scattering: an oscillating electric
            field induces a dipole, so the redirection generally grows rapidly toward shorter
            wavelength. The exact atmosphere fits cap their input frequencies at
            \(2.463\times10^{15}\), \(5.15\times10^{15}\), and
            \(2.922\times10^{15}\) Hz, respectively.

            The following cell writes the four named scattering components on a
            `(6, 240)` grid and uses depth index 3. It also probes each cap directly.
            """
        ),
        code(
            """
            scattering_wavelength_nm = np.geomspace(80.0, 2_500.0, 240)
            scattering = scattering_checkpoint("solar_dwarf", scattering_wavelength_nm)
            components = (
                ("free electrons", scattering.electron_scattering_cm2_per_g,
                 PAPER_COLORS["black"], "-", 1.8),
                ("H I Rayleigh", scattering.hydrogen_rayleigh_cm2_per_g,
                 PAPER_COLORS["blue"], "-", 1.8),
                ("He I Rayleigh", scattering.helium_rayleigh_cm2_per_g,
                 PAPER_COLORS["orange"], "-", 1.8),
                (r"$\\mathrm{H_2}$ Rayleigh",
                 scattering.molecular_hydrogen_rayleigh_cm2_per_g,
                 PAPER_COLORS["green"], "-", 1.8),
            )
            scattering_curves = tuple(
                (label, values[3], colour, style, width)
                for label, values, colour, style, width in components
            )
            _show_one_panel(
                scattering_wavelength_nm,
                scattering_curves,
                xlabel="Wavelength [nm]",
                ylabel=r"Scattering opacity [cm$^2$ g$^{-1}$]",
                title="Electron scattering is grey; Rayleigh scattering is coloured",
                xscale="log",
                yscale="log",
            )

            electron_span = np.ptp(scattering.electron_scattering_cm2_per_g[3])
            print("Thomson span across the plotted grid:", electron_span)
            print("H I / He I / H2 values unchanged above their input caps:",
                  np.array_equal(
                      scattering.cap_component_value_cm2_per_g,
                      scattering.above_cap_component_value_cm2_per_g,
                  ))
            print("runner-default IFOP vector:", list(RUNNER_OPACITY_FLAGS))
            """,
        ),
        markdown(
            r"""
            The electron curve is flat to the printed precision, while the neutral-particle
            curves grow blueward and then flatten once their capped input frequency is
            reached. The atmosphere and synthesis H I Rayleigh laws are not numerically
            identical—the former uses an inverse-wavelength fit and the latter tabulated
            polarizability information—so each product is checked against its own consumer.

            Most importantly, the total scattering array remains separate from absorption.
            It contributes to extinction but never enters the LTE thermal numerator as though
            redirection had deposited heat.

            The printed 20-entry vector is the atmosphere runner default. A branch is enabled
            only when its flag is exactly integer `1`. Passing `None` to the lower-level
            composite means twenty ones and is therefore **not** this default. IFOP 9 owns
            CH, OH, CIA, and five neutral-metal groups together. H\(_2\) Rayleigh under IFOP
            13 depends on the H I ground population built under IFOP 4. IFOP 19 is an
            optional, default-off Rosseland surrogate with a bolometric-like
            temperature-fourth source convention; it is retained in branch tests but is not
            mixed into the per-frequency source budget above.

            > **Movement III — One physical budget, two exact grids.** At one frequency, every
            > component now has a physical owner. We next decide where those formulae must be
            > evaluated for an atmosphere and for a requested synthesis window.

            ## 5.8 The atmosphere samples its continuum directly

            A fixed 30,000-point budget should not spend the same fraction of samples in the
            far ultraviolet for a 3200 K star and a 30000 K star. The atmosphere therefore
            chooses one of five logarithmic wavelength grids:

            \[
            \lambda_i[\mathrm{nm}]
            =10^{1+10^{-4}(i+s-1)},\qquad i=1,\ldots,30000,
            \]

            The start index is fully determined by the effective-temperature interval:

            | effective-temperature interval [K] | \(s\) |
            |---|---:|
            | \(T_{\rm eff}<4500\) | 11601 |
            | \(4500\leq T_{\rm eff}<7250\) | 9599 |
            | \(7250\leq T_{\rm eff}<13000\) | 7027 |
            | \(13000\leq T_{\rm eff}<30000\) | 3577 |
            | \(T_{\rm eff}\geq30000\) | 1 |

            Thus equality belongs to the hotter interval at every boundary. The grid is
            uniform in \(\log_{10}\lambda\), not in \(\lambda\). Every enabled continuum
            process is evaluated directly at all 30,000 frequencies.

            A smaller, second atmosphere calculation builds a reference scale for later line
            selection. It contains 343 physical wavelengths, a duplicated 344th wavelength,
            and a packed `2**30` sentinel. Its threshold is

            \[
            q_\nu=
            \frac{10^{-3}(\kappa_\nu^{\rm abs}+\kappa_\nu^{\rm sca})}
            {\max[1-\exp(-h\nu/kT),10^{-300}]}.
            \]

            Dividing out the stimulated-emission factor prepares a later comparison; it does
            not redefine continuum opacity. The cell below writes a five-row grid table and
            the solar-like `(6, 344)` threshold.
            """
        ),
        code(
            """
            atmosphere_grids = atmosphere_grid_checkpoint()
            solar_reference = line_reference_checkpoint(
                "solar_dwarf",
                effective_temperature_by_regime["solar_dwarf"],
            )

            print("T_eff [K] | first wavelength [nm] | last wavelength [nm] | "
                  "samples | active reference")
            for row in range(atmosphere_grids.effective_temperature_k.size):
                print(
                    f"{atmosphere_grids.effective_temperature_k[row]:9.0f} | "
                    f"{atmosphere_grids.first_wavelength_nm[row]:21.9f} | "
                    f"{atmosphere_grids.last_wavelength_nm[row]:20.6f} | "
                    f"{atmosphere_grids.sample_count[row]:7d} | "
                    f"{atmosphere_grids.active_reference_count[row]:16d}"
                )
            print("representative weights [first, interior, last] Hz:",
                  np.column_stack([
                      atmosphere_grids.first_frequency_weight_hz,
                      atmosphere_grids.interior_frequency_weight_hz,
                      atmosphere_grids.last_frequency_weight_hz,
                  ]))
            print("solar reference shape / dtype:",
                  solar_reference.threshold_cm2_per_g.shape,
                  solar_reference.dtype)
            print("active count / duplicate / sentinel:",
                  solar_reference.active_count,
                  solar_reference.duplicated_last_column,
                  solar_reference.packed_sentinel)
            print("inactive placeholder follows the exact scaling:",
                  solar_reference.inactive_placeholder_matches)
            """,
        ),
        markdown(
            r"""
            Every branch contains 30,000 samples, but its first wavelength moves blueward as
            effective temperature rises. The corresponding active reference counts are 226,
            240, 263, 299, and 338. A reference wavelength equal to the first direct-grid
            wavelength is inactive because the comparison is strict. Inactive short-wavelength
            entries begin from the exact `1e10` placeholder before receiving the same
            \(10^{-3}/s_\nu\) scaling.

            The output threshold is deliberately `float32`, its last physical column is
            duplicated, and its final packed coordinate is the printed sentinel. We have built
            the background scale that a later line stage will consume, but we have not
            introduced an oscillator strength or a line-selection inequality.

            ## 5.9 Why frequency columns may run in parallel

            Evaluating 30,000 frequencies would be slow in ordinary Python. Chapter 2
            introduced `njit`, compilation caches, and `prange`; here we can finally apply
            them to their real physical ownership pattern.
            """
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch05-prange-columns-v4.png"
                   alt="Conceptual hand-sketched array in which several horizontal frequency workers read one immutable state bus and each writes one complete distinct vertical depth column.">
              <figcaption><strong>Conceptual schematic: disjoint column
              ownership.</strong> A worker for frequency \(f\) reads the shared immutable
              depth state and tables, then writes the complete matching depth column
              \(K_{:,f}\). Different workers own different columns, so their write sets do
              not overlap. Ordered component assembly and interpolation remain outside the
              worker boxes.</figcaption>
            </figure>

            For the transparent law

            \[
            K_{df}=\frac{n_d\sigma_f}{\rho_d},
            \]

            one frequency worker writes only \(K_{:,f}\). The workers share read-only arrays
            but no output cell. That disjoint write ownership—not the mere presence of a
            loop—is what makes `prange` safe. The next cell compares plain Python, serial
            `njit`, and parallel `prange` on an `(80, 30000)` workload. First-call
            compilation, warm execution, and a fresh process reusing the external cache are
            reported separately.
            """
        ),
        code(
            """
            maximum_threads = min(4, max(1, os.cpu_count() or 1))
            timing = numba_timing_checkpoint(maximum_threads=maximum_threads)
            first = timing.first_process
            cached = timing.cached_process

            timing_rows = (
                ("plain Python", first["python_seconds"], 1),
                ("serial njit, first call", first["serial_first_seconds"], 1),
                ("serial njit, warm call", first["serial_warm_seconds"], 1),
                ("prange, one thread", first["parallel_one_thread_seconds"], 1),
                ("prange, warm many threads",
                 first["parallel_many_thread_seconds"], first["used_threads"]),
                ("serial njit, cached fresh process",
                 cached["serial_first_seconds"], 1),
            )
            print("condition                           seconds    threads")
            for label, seconds, threads in timing_rows:
                print(f"{label:35s} {seconds:9.5f} {threads:10d}")
            print("shape / dtype:", tuple(first["shape"]), first["dtype"])
            print("all outputs bitwise equal:", timing.all_outputs_bitwise_equal)
            print("external cache files:", timing.cache_file_count)
            """,
        ),
        markdown(
            r"""
            Parity comes before speed: every implementation wrote the same `float64`
            opacity columns bit for bit in this measured run. The first compiled calls include
            compilation; the warm and fresh-process cached rows answer different timing
            questions. The numbers are properties of this machine, thread count, shape, and
            cache condition—not a universal speedup promise.

            The atmosphere continuum remains a CPU NumPy/Numba calculation. Ordered flag
            assembly, component addition, and interpolation stay outside the parallel
            frequency loop. A separate production dispatch uses serial interpolation below
            roughly 8192 queries because worker overhead can exceed the useful arithmetic;
            that is an overhead policy, not different physics.

            ## 5.10 Synthesis evaluates only used edge intervals

            A requested spectrum can contain many pixels while crossing only a limited number
            of continuum thresholds. Synthesis therefore uses a 341-edge table, or 340
            intervals, rather than evaluating every component at every requested pixel.
            """
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch05-two-grids-edge-triplet-v4.png"
                   alt="Conceptual hand-sketched fork from two declared consumer views and their owned immutable table bundles into a direct 30000-point atmosphere grid and a synthesis edge interval sampled just inside its two sides plus its midpoint.">
              <figcaption><strong>Conceptual schematic: two process-distinct
              consumers.</strong> Two declared consumer views and their separately owned
              immutable table bundles feed two exact algorithms: the atmosphere evaluates all
              30,000 samples directly, while synthesis evaluates only the intervals used by
              the requested window. The fork is conceptual; it is not one shared mapping or
              table object. In one enlarged interval, samples lie at
              \(\lambda_L+\epsilon\), \(\lambda_M\), and
              \(\lambda_R-\epsilon\). An exact internal boundary
              \(\lambda=\lambda_R\) belongs to the next, redder interval
              \([\lambda_R,\lambda_{R+1})\). The drawing is conceptual; distances are not to
              scale.</figcaption>
            </figure>

            Red means larger wavelength and lower frequency. For one interval
            \(\lambda_L<\lambda_M<\lambda_R\), the exact one-sided sample frequencies are

            \[
            \nu_L=\frac{|\nu(\lambda_L)|}{1.0000001},\quad
            \nu_M=\frac{c}{(\lambda_L+\lambda_R)/2},\quad
            \nu_R=|\nu(\lambda_R)|\,1.0000001 .
            \]

            `searchsorted(..., side="right") - 1` assigns an exact internal edge to its
            red-wavelength interval. The stored edge frequencies remain signed provenance,
            but this sampler uses their magnitudes at both ends.

            Three basis functions reconstruct a value inside the interval:

            \[
            \begin{aligned}
            L_L&=\frac{(\lambda-\lambda_M)(\lambda-\lambda_R)}{d_2},\\
            L_M&=\frac{2(\lambda_L-\lambda)(\lambda-\lambda_R)}{d_2},\\
            L_R&=\frac{(\lambda-\lambda_L)(\lambda-\lambda_M)}{d_2},
            \qquad d_2=\frac{(\lambda_R-\lambda_L)^2}{2}.
            \end{aligned}
            \]

            They sum to one and reproduce their three nodes. Absorption and scattering are
            floored independently at `1e-30`, interpolated in \(\log_{10}\kappa\), and
            exponentiated. A physically inactive component may still be exactly zero before
            this final interpolation floor. The one-panel reconstruction below uses depth
            index 3.
            """
        ),
        code(
            """
            reconstruction = edge_reconstruction_checkpoint("solar_dwarf", 500.0, 240)
            edge_probe = edge_triplet_checkpoint((
                np.nextafter(reconstruction.left_wavelength_nm,
                             reconstruction.right_wavelength_nm),
                reconstruction.midpoint_wavelength_nm,
                reconstruction.right_wavelength_nm,
            ))
            edge_trace = edge_use_trace_checkpoint()
            edge_curves = (
                ("standard continuum", reconstruction.exact_absorption_cm2_per_g[3],
                 PAPER_COLORS["black"], "-", 2.2),
                ("three-point reconstruction",
                 reconstruction.reconstructed_absorption_cm2_per_g[3],
                 PAPER_COLORS["blue"], "--", 1.8),
            )
            _show_one_panel(
                reconstruction.target_wavelength_nm, edge_curves,
                xlabel="Wavelength [nm]",
                ylabel=r"Continuum absorption [cm$^2$ g$^{-1}$]",
                title="Three one-sided samples reconstruct one positive interval",
                yscale="log",
                vertical=(reconstruction.right_wavelength_nm, "right edge"),
                points=(("one-sided samples", reconstruction.sample_wavelength_nm,
                         reconstruction.absorption_samples_cm2_per_g[3],
                         PAPER_COLORS["orange"]),),
            )
            print("packaged samples bitwise / sign-flip invariant:",
                  edge_probe.packaged_samples_bitwise_equal,
                  edge_probe.sign_flip_invariant)
            print("largest basis-sum error:",
                  np.max(np.abs(reconstruction.basis_sum - 1.0)))
            print("used / total intervals / opacity calls:",
                  edge_trace.used_interval_index.size,
                  edge_trace.total_interval_count, edge_trace.called_frequency_count)
            print("exact edge uses red interval:", edge_trace.exact_internal_edge_assigned_interval)
            """,
        ),
        markdown(
            r"""
            The reconstructed and standard curves coincide at the plotted scale, while the
            printed basis sum closes at floating-point accuracy. The complete 1020-sample
            vector is bitwise identical to the packaged field, and changing only the stored
            frequency signs does not change the present sampler. Exactly three opacity columns
            are evaluated for each used interval; unused intervals perform no opacity work.

            This figure establishes one positive interval and its exact indexing policy. It
            does not claim that an edge-smoothed reconstruction is the atmosphere algorithm:
            the atmosphere evaluates its own direct grid.

            > **Movement IV — Bind the exact consumers and close them independently.** The
            > physical terms and sampling rules are now visible. We can introduce the complete
            > state projections without turning the chapter into an API tour.

            ## 5.11 The two 18-field consumers are not one schema

            The molecular-equilibrium handoff released a complete 27-field schema-v4 synthesis
            mapping. The standard continuum call projects exactly 18 fields from it. That
            trimmed synthesis view is
            **not** the packed 18-field `ContinuumAtmosphereState`: the two consumers merely
            happen to have the same field count.

            | physical role | packed atmosphere adapter | trimmed synthesis projection |
            |---|---|---|
            | thermodynamic vectors | temperature, density, electrons, gas pressure, microturbulence | temperature, density, electrons |
            | H/He bases | normalized H stages plus actual and normalized named H/He vectors | normalized H stages plus actual neutral H, neutral He, and singly ionized He vectors |
            | broad populations | actual and normalized `(D,1006)` packed arrays | actual and normalized `(D,6,139)` cubes |
            | molecules | normalized CH/OH aliases and departure coefficients | no CH/OH/CIA input; no stored H\(_2\) input |
            | edge geometry | none | signed frequencies, wavelengths, midpoints, interval widths |

            The synthesis projection also omits the optional actual
            `hydrogen_ionized_population`; `build_pops` reconstructs the required H II fallback
            from the second normalized hydrogen-stage column. Neither 18-field view is
            reconstructed from the other.

            Host `float64` owns edge searches, unique-interval selection, and table brackets.
            The large population and opacity arrays remain on the resolved Torch device in its
            work dtype. The next cell checks field counts and reports the actual runtime pair;
            it does not move an opacity slab to the host.
            """
        ),
        code(
            """
            projections = state_projection_checkpoint("solar_dwarf")
            runtime_device, runtime_dtype = resolve_runtime(None, None)
            cpu_device, cpu_dtype = resolve_runtime("cpu", None)
            from payne_zero_synthesis.continuum import ContinuumTables

            standalone_path = (
                repository_root / "data/static/synthesis_tables/continuum_tables.npz"
            )
            standalone_tables = ContinuumTables.from_npz(
                standalone_path, device="cpu", dtype=None
            )

            synthesis_fields = set(projections.synthesis_continuum_field_names)
            atmosphere_fields = set(projections.atmosphere_continuum_field_names)
            print("schema-v4 synthesis handoff fields:",
                  len(projections.synthesis_handoff_field_names))
            print("trimmed synthesis continuum fields:",
                  len(projections.synthesis_continuum_field_names))
            print("packed atmosphere continuum fields:",
                  len(projections.atmosphere_continuum_field_names))
            print("shared exact spellings:",
                  len(projections.shared_field_names),
                  projections.shared_field_names)
            print("stored H II consumed by trimmed synthesis:",
                  "hydrogen_ionized_population" in synthesis_fields)
            print("stored H2 consumed by trimmed synthesis:",
                  "molecular_hydrogen_population" in synthesis_fields)
            print("CH/OH aliases owned by atmosphere:",
                  {"ch_population", "oh_population"} <= atmosphere_fields)
            print("resolved synthesis device / dtype:",
                  runtime_device, runtime_dtype)
            print("CPU parity device / dtype:", cpu_device, cpu_dtype)
            print("standalone CPU table dtype when dtype=None:",
                  standalone_tables.dtype)
            """,
        ),
        markdown(
            r"""
            The live counts reconcile the apparent contradiction: 27 fields remain available
            upstream, while the standard synthesis continuum consumes only its ordered 18-field
            projection. The separate atmosphere adapter owns packed populations and molecular
            aliases. The shared spellings identify genuinely shared physical vectors; they do
            not make either structure a view of the other.

            The resolved product policy is CUDA then MPS then CPU, with `float64` on CUDA/CPU
            and `float32` on MPS. Product calls pass that resolved dtype explicitly. The
            standalone `ContinuumTables.from_npz(device="cpu", dtype=None)` call is a deliberate
            exception: it does not invoke the resolver and falls through to the module default,
            `float32`. The comparisons below pass the printed CPU `float64` reference dtype;
            large synthesis arrays remain device-resident, and only these compact checkpoints
            return host arrays.

            ## 5.12 Close the atmosphere product

            The atmosphere product now has no missing ingredient. For each of the four declared
            regimes, we will:

            1. retain all 14 named absorption components and their source contributions on the
               exact 27-frequency diagnostic grid;
            2. preserve their ordered partial sums;
            3. pass the explicit runner-default IFOP vector;
            4. evaluate the direct 30,000-point product;
            5. build the `(depth, 344)` line-reference threshold;
            6. only then open the compact comparison golden.

            The full output arrays are CPU NumPy `float64`
            `continuum_absorption`, `continuum_scattering`, and `continuum_source`, each
            `(6, 30000)`. Valid positive inputs enter the exact kernels; no final
            non-negativity “repair” is applied.
            """
        ),
        code(
            """
            atmosphere_budgets, atmosphere_products, line_references = [], [], []
            for regime in regime_names:
                atmosphere_budgets.append(
                    atmosphere_component_budget(regime, comparison_frequency_hz))
                atmosphere_products.append(run_full_atmosphere_continuum(
                    regime, effective_temperature_by_regime[regime]))
                line_references.append(line_reference_checkpoint(
                    regime, effective_temperature_by_regime[regime]))
            local_atmosphere = {
                name: np.stack([getattr(item, attribute) for item in atmosphere_budgets])
                for name, attribute in (
                    ("absorption", "exact_absorption_cm2_per_g"),
                    ("scattering", "exact_scattering_cm2_per_g"),
                    ("source", "exact_source"),
                )
            }
            local_threshold = np.stack(
                [item.threshold_cm2_per_g for item in line_references])
            opacity_arrays = [
                values for item in atmosphere_products
                for values in (item.absorption_cm2_per_g, item.scattering_cm2_per_g)
            ]
            print("full shapes:", {item.absorption_cm2_per_g.shape
                                   for item in atmosphere_products})
            print("finite/nonnegative opacity:", all(
                np.all(np.isfinite(values)) and np.all(values >= 0.0)
                for values in opacity_arrays))
            """,
        ),
        markdown(
            r"""
            The direct products have now been calculated and checked without consulting a
            target. Only at this point do we open the compact reader golden. That order
            prevents the reference data from becoming an input to the calculation it is
            supposed to test. The comparison keeps absorption, scattering, and source
            separate, then checks the regime axis, the 14-component order, and the
            `float32` line-reference slab.
            """
        ),
        code(
            """
            atmosphere_golden = _load_reader_golden_members(
                "axis__regime_name",
                "atmosphere__compact__absorption",
                "atmosphere__compact__scattering",
                "atmosphere__compact__source",
                "atmosphere__component__absorption_name",
                "line_reference__threshold",
            )
            compact_parity = all(
                np.allclose(values, atmosphere_golden[f"atmosphere__compact__{name}"],
                            rtol=3.0e-15, atol=1.0e-30)
                for name, values in local_atmosphere.items()
            )
            metadata_exact = (
                tuple(atmosphere_golden["axis__regime_name"].tolist()) == regime_names
                and tuple(atmosphere_golden[
                    "atmosphere__component__absorption_name"].tolist())
                == atmosphere_budgets[0].component_names
            )
            closures = tuple(max(np.max(np.abs(getattr(item, attribute)))
                                 for item in atmosphere_budgets)
                             for attribute in ("absorption_residual_cm2_per_g",
                                               "source_numerator_residual"))
            print("compact / metadata / line-reference parity:", compact_parity,
                  metadata_exact, np.array_equal(local_threshold,
                                                 atmosphere_golden[
                                                     "line_reference__threshold"]))
            print("absorption / source-numerator closure to roundoff:", closures)
            """,
        ),
        markdown(
            r"""
            Each full product has the promised `(6, 30000)` shape and contains finite,
            nonnegative absorption and scattering without a repair clamp. The compact
            component route agrees with the pinned Payne Zero comparison golden for all four
            regimes, and the line-reference arrays agree with their exact `float32` targets.

            The 14-component reconstruction closes to the printed floating-point roundoff; it
            is **not** claimed to be bitwise identical after regrouping. Retaining the
            elementwise components and source numerators is what makes that statement
            meaningful—a small term cannot hide behind one global total. The exhaustive
            `(6, 30000)` oracle products remain integration-test evidence and are deliberately
            not loaded by this notebook.

            IFOP 19 is checked separately for implementation coverage. Its
            temperature-fourth, bolometric-like source is not ordinary per-frequency
            \(B_\nu\), so it cannot enter the standard source comparison above.

            ## 5.13 Close standard synthesis, then label the alternatives

            Standard synthesis uses the trimmed state, calls `build_pops`, evaluates three
            columns for each used edge interval, and interpolates absorption and scattering
            separately. Its exact sample-column addition order is:

            1. H-minus;
            2. H I;
            3. H\(_2^+\), He-minus, and compact neutral metals;
            4. He I;
            5. He II;
            6. hot metals;
            7. Si II Peach.

            Scattering adds H I Rayleigh, Thomson, and the He I/H\(_2\) Rayleigh terms carried
            by the minor component. The standard call makes that convention explicit as
            `coulomb_table_energy_first=False`, and no `FrequencyInvariants` object is passed.

            The comparison wavelength vector is fixed before the golden is opened: three
            one-sided points in each of eleven declared edge intervals plus three fixture
            boundary probes, for 36 wavelengths. After the four standard products are built,
            one solar-like diagnostic call evaluates the exact 27 threshold frequencies.
            One solar-like precomputed extension call evaluates the exact validated
            **12-point vector**; the interval from 100 to 2500 nm is not claimed as continuous
            support.
            """
        ),
        code(
            """
            standard_products = [
                standard_synthesis_component_checkpoint(regime, comparison_wavelength_nm)
                for regime in regime_names]
            standard = {
                "absorption": np.stack([x.final_absorption_cm2_per_g for x in standard_products]),
                "scattering": np.stack([x.final_scattering_cm2_per_g for x in standard_products])}
            first_standard = standard_products[0]
            standard_residuals = tuple(
                max(np.max(np.abs(getattr(x, field))) for x in standard_products)
                for field in ("absorption_residual_cm2_per_g",
                              "scattering_residual_cm2_per_g"))
            print("standard component order:", first_standard.absorption_component_names,
                  first_standard.scattering_component_names)
            print("maximum ordered-sum residuals:", standard_residuals)
            diagnostic = run_sampled_synthesis_continuum_at_frequency(
                "solar_dwarf", comparison_frequency_hz)
            extension_support_nm = np.asarray(SAMPLED_EXTENSION_SUPPORTED_WAVELENGTH_NM, np.float64)
            extension = run_sampled_synthesis_continuum(
                "solar_dwarf", extension_support_nm, precomputed=True)
            continuum_opacity = standard["absorption"] + standard["scattering"]
            standard_layout = all(not item.coulomb_table_energy_first and
                                  item.frequency_invariants_was_none
                                  for item in standard_products)
            print("local route shapes:", continuum_opacity.shape,
                  diagnostic.absorption_cm2_per_g.shape,
                  extension.absorption_cm2_per_g.shape)
            print("coulomb_table_energy_first=False:", standard_layout)
            print("diagnostic / extension source equals direct B_nu:",
                  diagnostic.source_matches_direct, extension.source_matches_direct)
            print("diagnostic frequencies preserve all 27 probe bit patterns:",
                  np.array_equal(diagnostic.frequency_hz.view(np.uint64),
                                 comparison_frequency_hz.view(np.uint64)))
            print("extension is the exact validated 12-point vector:",
                  extension_support_nm.size == 12 and
                  np.array_equal(extension.supported_wavelength_nm, extension_support_nm))
            """,
        ),
        markdown(
            r"""
            All three synthesis routes have now been evaluated locally. The standard route
            has also exposed its exact call convention, while both sampled routes have checked
            their Planck sources directly. We now open the reader golden and compare like with
            like: standard absorption and scattering across four regimes, then the solar-like
            diagnostic and extension arrays. The golden never supplies a frequency, wavelength,
            population, or opacity to the calculations above.
            """
        ),
        code(
            """
            synthesis_golden = _load_reader_golden_members(
                *synthesis_golden_member_names)
            standard_match = all(
                np.allclose(values, synthesis_golden[f"synthesis__standard__{name}"],
                            rtol=3.0e-15, atol=1.0e-30)
                for name, values in standard.items()
            )
            sampled_match = {}
            for route, item in (("diagnostic", diagnostic), ("extension", extension)):
                sampled_match[route] = all(
                    np.allclose(getattr(item, attribute),
                                synthesis_golden[f"synthesis__{route}__{name}"][1],
                                rtol=3.0e-15, atol=1.0e-30)
                    for name, attribute in (("absorption", "absorption_cm2_per_g"),
                                            ("scattering", "scattering_cm2_per_g"),
                                            ("source", "source_bnu"))
                )
            print("route | shape | Coulomb layout | invariants | pinned parity")
            print("standard", continuum_opacity.shape, False, "none", standard_match)
            print("diagnostic", diagnostic.absorption_cm2_per_g.shape,
                  True, "none", sampled_match["diagnostic"])
            print("extension", extension.absorption_cm2_per_g.shape,
                  extension.coulomb_table_energy_first, "explicit",
                  sampled_match["extension"])
            print("golden preserves the exact validated 12-point vector:",
                  np.array_equal(
                      synthesis_golden["synthesis__extension__wavelength_nm"],
                      extension_support_nm))
            """,
        ),
        markdown(
            r"""
            The standard four-regime absorption and scattering slabs reproduce their pinned
            CPU-`float64` targets. Their sum,
            `continuum_opacity = continuum_absorption + continuum_scattering`, is a derived
            extinction scale for later line selection and transfer—not a third independently
            calculated process.

            The two sampled calls are deliberately labelled alternatives:

            | route | sampling and process identity | output | non-claim |
            |---|---|---|---|
            | atmosphere product | direct 30,000-point CPU grid, including CH/OH/CIA | absorption, scattering, absorption-weighted source | not edge interpolation |
            | standard synthesis product | used edge triplets, compact metals, no CH/OH/CIA | absorption and scattering | no source; no `FrequencyInvariants` |
            | sampled diagnostic | caller frequencies; scalar full-neutral fallback formulae; Coulomb layout `True` | absorption, scattering, direct \(B_\nu\) | no materialized grids; not the standard route |
            | sampled extension | exact fixed frequencies; materialized full-neutral grids; additional N I/O I/Mg II/Ca II grids; explicit `FrequencyInvariants` | independently checked absorption, scattering, direct \(B_\nu\) | validated only on the printed 12-point vector |

            The diagnostic and extension source arrays match a direct Planck calculation. That
            identity does not turn either sampled lane into the atmosphere's
            absorption-weighted `continuum_source`, and it does not promote the extension to
            the standard pipeline. The standard call trace contains no
            `FrequencyInvariants`.

            ## 5.14 Chapter summary

            We can now answer the opening question. Gas remains opaque between narrow spectral
            features because several broad interactions survive there:

            - a microscopic cross section becomes mass opacity through
              \(\kappa_\nu=n\sigma_\nu/\rho\);
            - bound-free, free-free, photodissociation, and collision-induced absorption
              transfer photon energy to matter, while stimulated emission reduces the net LTE
              absorption;
            - Thomson and Rayleigh scattering redirect photons and remain outside the thermal
              source numerator;
            - bound-state terms read partition-normalized population bases, while ionic
              free-free terms read actual charge-square populations;
            - the atmosphere evaluates a direct 30,000-point CPU grid, whereas synthesis
              evaluates only used edge triplets and reconstructs positive opacity in logarithmic
              space;
            - exactness means matching each consumer's state, process set, grid, order, dtype,
              and device policy—not forcing distinct routes to equal one another.

            The exact outputs now available are:

            - atmosphere `continuum_absorption`, `continuum_scattering`, and
              `continuum_source`, each CPU NumPy `float64` with shape `(D, 30000)`;
            - the atmosphere line-selection threshold, NumPy `float32` with shape `(D, 344)`;
            - synthesis `continuum_absorption` and `continuum_scattering`, each `(D, W)` in the
              resolved Torch work dtype;
            - derived synthesis
              `continuum_opacity = continuum_absorption + continuum_scattering`.

            No line opacity, emergent flux, or spectrum has yet been computed. A smooth
            continuum cannot create the narrow dips in the opening observation.

            ### Next: give one transition a strength and shape

            [Chapter 6: One Spectral Line](/reader.html?ch=6) begins with that missing
            bound-bound interaction and asks how one rest wavelength acquires a strength,
            thermal and microturbulent width, damping wings, and a normalized profile.
            """
        ),
    ]
    return notebook(cells)
