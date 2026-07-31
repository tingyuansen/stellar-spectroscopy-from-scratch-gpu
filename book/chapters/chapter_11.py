"""Canonical Chapter 11 notebook: starting and blanketing an atmosphere."""

from __future__ import annotations

from book.cells import code, markdown, notebook, setup_cell


TITLE = "Starting and Blanketing an Atmosphere"


def build_notebook() -> dict:
    """Return the deterministic Chapter 11 notebook document."""

    cells = [
        markdown(
            r"""
            # Chapter 11 — Starting and Blanketing an Atmosphere

            Chapter 10 deliberately accepted a supplied, schema-valid
            atmosphere so that we could finish the synthesis pipeline first.
            It did not claim that the supplied structure was converged. We now
            begin building the physical atmosphere that synthesis ultimately
            needs. This chapter asks one question:

            > How does a supplied fixed-column seed become the 80-layer,
            > 30,000-frequency opacity state needed by one atmosphere pass?

            The endpoint is deliberately `OpacityState`, before transfer.
            Radiation moments, thermodynamics, and convection belong to
            the next chapter; applying a correction, remapping the coupled
            state, and repeating passes come later. A live seed, population
            state, or opacity state is not a converged schema-v4 product.

            We use one checksum-bound solar-like integration fixture. Its
            depth structure is a supplied starting state, not a convergence
            claim. Every production boundary below calls the canonical staged
            implementation whose source identity is verified outside the
            reader.

            > **Movement I — Which seed is safe to start from?**
            """
        ),
        setup_cell(),
        code(
            """
            from dataclasses import fields

            import matplotlib.pyplot as plt
            import numpy as np

            from book.chapter11_runtime import (
                blanketing_checkpoint, configure_chapter11_runtime,
                continuum_checkpoint, full_opacity_state,
                hydrostatic_checkpoint, load_seed_atmosphere,
                opacity_pass_checkpoint, population_checkpoint,
                quantization_checkpoint, remap_checkpoint,
                reuse_checkpoint, sampling_grid_checkpoint,
                seed_checkpoint, selection_checkpoint, setup_checkpoint,
            )
            from book.chapter11_teaching import opacity_memory_ledger
            from book.plot_style import add_quiet_grid, single_panel

            configure_chapter11_runtime()
            """,
            tags=("book-setup", "hide-input"),
        ),
        markdown(
            r"""
            ## 11.1 A seed is nine aligned columns plus explicit metadata

            An iteration needs a complete starting structure, not only
            \(T_{\rm eff}\) and \(\log g\). At each depth it needs temperature,
            gas pressure, electron density, column mass, Rosseland opacity,
            radiative acceleration, microturbulence, convective flux, and
            convective velocity.

            These are the nine exact `ModelAtmosphere` columns. Mass density is
            not stored in the seed: the equation of state recomputes it from
            the current thermodynamic state before opacity is formed. Neither
            the current nor the standard Rosseland optical depth is a seed
            column. The standard coordinate is constructed from the run setup
            in Section 11.2; the current optical depth is obtained only after
            frequency-dependent opacity has been accumulated.

            Column mass is especially useful in a plane-parallel atmosphere:

            \[
            m(z)=\int_z^\infty \rho(z')\,dz'.
            \]

            It is the mass above one square centimeter of surface. The outer
            layer has the smallest \(m\); index and column mass increase
            inward. This turns hydrostatic balance into a particularly simple
            pressure equation later in the chapter.

            Every array is depth-major and uses cgs units: column mass in
            g cm\(^{-2}\), temperature in K, gas pressure in dyn cm\(^{-2}\),
            number densities in cm\(^{-3}\), opacity in cm\(^2\) g\(^{-1}\),
            acceleration in cm s\(^{-2}\), microturbulence and convective
            velocity in cm s\(^{-1}\), and convective flux in
            erg cm\(^{-2}\) s\(^{-1}\).
            The abundance block has 99 entries. Hydrogen and helium are linear
            number fractions; metals are stored as base-10 logarithmic number
            abundances. Mixing those conventions would change the stellar
            composition by orders of magnitude.
            """
        ),
        code(
            """
            seed = seed_checkpoint()
            print("layers:", seed.layers)
            print("abundance entries:", seed.abundance_count)
            print("column mass increases inward:", seed.column_mass_strictly_increasing)
            print("field                  shape")
            for name, shape in zip(seed.field_names, seed.field_shapes):
                print(f"{name:23s} {shape}")
            """,
        ),
        markdown(
            r"""
            Validation answers “is this object safe to interpret?” rather than
            “is this atmosphere correct?” It requires finite aligned columns,
            positivity for five structural fields, increasing column mass, and
            nonnegative microturbulence. It does not sort layers, floor values,
            require 80 layers, or repair an incomplete abundance block. Those
            would be scientific changes hidden inside an input check.

            Nor does validation certify hydrostatic or radiative equilibrium.
            The production 80-layer requirement appears only when the
            line-opacity kernel needs that exact grid. Physical acceptance
            comes after repeated passes, not at the seed boundary.
            """
        ),
        code(
            """
            print("one-change-at-a-time failure gallery")
            for case, message in zip(seed.error_cases, seed.error_messages):
                print(f"{case:27s} -> {message}")
            """,
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch11-seed-gates-to-pass-state-v1.png"
                   alt="Conceptual diagram in which the nine aligned ModelAtmosphere columns—column mass, temperature, gas pressure, electron density, Rosseland opacity, radiative acceleration, microturbulence, convective flux, and convective velocity—pass shape, sign, and ordering validation, optionally cross a declared fixed-column quantization gate, and form RunSetup for a copied first pass, while previous radiation support enters hydrostatic gas pressure only on pass two and later.">
              <figcaption><strong>Conceptual seed-boundary schematic.</strong>
              Validation checks a supplied <code>ModelAtmosphere</code>;
              fixed-column quantization is crossed only at a declared numerical
              boundary. Pass 1 copies the seed. Hydrostatic gas pressure first
              consumes remapped radiation support on pass 2.</figcaption>
            </figure>
            """
        ),
        markdown(
            r"""
            ## 11.2 The standard Rosseland array is a coordinate

            The production depth coordinate is

            \[
            \tau_{{\rm std},d}=10^{-6.875+0.125d},\qquad d=0,\ldots,79.
            \]

            The logarithmic spacing places many layers near the optically thin
            surface while still reaching deep, nearly diffusive material. It
            labels the standard 80-layer route, controls the microturbulence
            prescription, and provides a common target when a corrected state
            is remapped.

            The word “standard” is essential. This array is a coordinate
            template, not the current physical Rosseland optical depth.
            Computing the latter requires the frequency-dependent opacity,
            its Rosseland mean, and integration through column mass. Chapters
            12 and 13 perform those steps.
            """
        ),
        code(
            """
            setup = setup_checkpoint()
            tau_std = setup.standard_rosseland_optical_depth
            print("shape:", tau_std.shape)
            print("endpoints:", tau_std[0], tau_std[-1])
            print("constant log10 spacing:", np.unique(np.round(np.diff(np.log10(tau_std)), 12)))
            """,
        ),
        markdown(
            r"""
            ## 11.3 Resolve controls once, visibly

            A physical pass should not reinterpret configuration halfway
            through. `resolve_run_setup` therefore turns user input and seed
            metadata into one explicit `RunSetup`: it clamps iteration
            counters, obtains surface gravity and the 20 opacity switches,
            rejects or disables unsupported branches, and records whether
            pressure iteration, molecular chemistry, and convection are
            active.

            Surface gravity follows the usual spectroscopic convention

            \[
            g=10^{\log g}\ {\rm cm\,s^{-2}}.
            \]

            Chapter 11 enables molecular chemistry and the pressure-iteration
            machinery, disables convection so that we can stop before that
            physics, and requests one pass. The printed setup makes every one
            of those choices inspectable.
            """
        ),
        code(
            """
            print("RunSetup fields:", len(setup.setup_field_names))
            print(", ".join(setup.setup_field_names))
            print()
            print("iterations:", setup.iterations)
            print("Teff, log g:", setup.effective_temperature, setup.log_surface_gravity)
            print("g [cm s^-2]:", setup.surface_gravity_cgs)
            print("molecules:", setup.molecules_enabled)
            print("pressure iteration:", setup.pressure_iteration_enabled)
            print("IFOP:", setup.opacity_flags)
            """,
        ),
        markdown(
            r"""
            Microturbulence represents unresolved velocity broadening below
            the resolved atmospheric scale. It enters the Doppler width and
            therefore changes both line selection and line opacity.

            Its initialization has a sharper rule than “fill missing values.”
            If no layer is positive, the exact standard profile overwrites the
            supplied array in place. If even one layer is positive, the whole
            array is left unchanged—including any remaining zeros.
            """
        ),
        code(
            """
            print("all-zero input after resolution")
            print("  positive layers:", np.count_nonzero(setup.all_zero_profile_after > 0.0))
            print("  range [km/s]:", setup.all_zero_profile_after.min() / 1e5,
                  setup.all_zero_profile_after.max() / 1e5)
            print("partly-positive input unchanged:", setup.partly_positive_unchanged)
            print("  positive layers:",
                  np.count_nonzero(setup.partly_positive_profile_after > 0.0))
            """,
        ),
        markdown(
            r"""
            ## 11.4 Pass 1 keeps the supplied pressure

            In plane-parallel geometry, \(dm=-\rho\,dz\) changes the familiar
            hydrostatic equation \(dP/dz=-\rho g\) into \(dP/dm=g\).
            Radiation and turbulent motions can carry part of the weight, so
            the gas alone need not supply the full pressure. On pass 2 and
            later, the production runner uses

            \[
            P_{\rm gas}=g\,m-P_{\rm rad,int}-P_{\rm turb}.
            \]

            The missing surface constant has not been forgotten. The absolute
            radiation pressure is
            \(P_{\rm rad,abs}=P_{\rm rad,int}+P_0\), while the external-deck
            convention writes total support as

            \[
            P_{\rm gas}+P_{\rm rad,abs}+P_{\rm turb}=g\,m+P_0.
            \]

            Substitution cancels \(P_0\) from the gas-pressure update. The
            reusable hydrostatic helper has a generic `pressure_constant`
            argument, but `run_atmosphere_model` leaves it at its production
            default of zero and passes the integrated radiation pressure.

            Pass 1 is different: the runner copies the seed pressure because
            it has not yet computed a self-consistent radiation field. The
            support columns below are controlled inputs used to expose the
            later-pass boundary. Evolving turbulent pressure is outside the
            supported model. If excessive support makes the formula
            nonpositive, the implementation warns and applies a positive
            floor; it does not quietly return an invalid pressure.
            """
        ),
        code(
            """
            hydro = hydrostatic_checkpoint()
            print("maximum balance residual:",
                  np.max(np.abs(hydro.balance_residual)))
            print("radiation support / gravity pressure:",
                  np.unique(hydro.integrated_radiation_pressure / hydro.gravity_pressure))
            print("turbulent support all zero:", np.all(hydro.turbulent_pressure == 0.0))
            print("production pressure_constant:", hydro.pressure_constant)
            print("forced bad-support warning:", hydro.warning_message)
            print("warning branch remains positive:", hydro.warning_floor_positive)
            """,
        ),
        markdown(
            r"""
            ## 11.5 Fixed-column text is a numerical operator

            The historical fixed-column atmosphere format stores only a finite
            number of decimal digits. Formatting and parsing are therefore not
            a transparent file operation; together they define a quantization
            map

            \[
            Q(\mathbf{x})=\operatorname{parse}
              [\operatorname{format}(\mathbf{x})].
            \]

            The returned arrays are float64, but the digits discarded by the
            formatter do not reappear. This small change matters because an
            iterative nonlinear solver can follow a different last-bit
            trajectory from a slightly different starting point. Once a deck
            is on the representable lattice, a second application satisfies
            \(Q[Q(\mathbf{x})]=Q(\mathbf{x})\).
            """
        ),
        code(
            """
            quantized = quantization_checkpoint()
            print("field                    max |delta|       max relative")
            for name, absolute, relative in zip(
                quantized.field_names,
                quantized.maximum_absolute_delta,
                quantized.maximum_relative_delta,
            ):
                print(f"{name:23s} {absolute:13.5e} {relative:13.5e}")
            print("deck lines:", quantized.deck_line_count)
            print("second application bitwise equal:",
                  quantized.second_application_bitwise_equal)
            """,
        ),
        markdown(
            r"""
            Later, a correction changes both the physical columns and their
            Rosseland-depth locations. To compare successive passes, every
            carried field must be placed back on the standard grid. The exact
            scalar remapper from Chapter 9 is the primitive for that task.

            Constant data stay constant inside the source domain. The helper's
            second return is the final source-interval index reached (reported
            as `previous_source_index - 1`), not a count of target points
            outside the source grid. Chapter 11 checks this easily
            misunderstood contract but does not remap a corrected atmosphere;
            Chapter 13 owns that complete, coupled operation.
            """
        ),
        code(
            """
            remapped = remap_checkpoint()
            interior = (
                (remapped.target_grid >= remapped.source_grid[0])
                & (remapped.target_grid <= remapped.source_grid[-1])
            )
            print("constant interior exact:",
                  np.all(remapped.constant_remap[interior] == 3.5))
            print("constant final source interval:",
                  remapped.constant_final_source_interval_index)
            print("monotone final source interval:",
                  remapped.monotone_final_source_interval_index)
            print("monotone remap:", remapped.monotone_remap)
            """,
        ),
        markdown(
            r"""
            > **Movement II — How does the mixture become blanketing opacity?**

            ## 11.6 Recompute the current population state

            Opacity depends on how many absorbers occupy each ionization,
            excitation, and molecular state. A seed may contain pressure and
            electron density, but its populations cannot simply be trusted
            after temperature, pressure, or composition changes. The equation
            of state therefore recomputes atomic, ionic, electronic, and
            molecular populations at every depth.

            The two population arrays retain their Chapter 3 meanings:
            physical ion-stage counts and counts divided by their partition
            functions are separate state, not interchangeable views. The
            latter multiplies Boltzmann factors in line opacity. The state also
            supplies fractional Doppler widths—thermal plus microturbulent
            broadening divided by \(c\)—and the packed population-strength
            factor used during line selection. Convection has not yet been
            evaluated, so its thermodynamic samples are deliberately absent.
            """
        ),
        code(
            """
            population = population_checkpoint()
            print("population fields:", population.population_field_names)
            print("runtime fields:", len(population.runtime_field_names))
            print("layers, packed slots:", population.layers, population.packed_slot_count)
            print("molecular state:", population.molecular_state_present)
            print("active molecules:", population.molecule_count)
            print("mass density range [g cm^-3]:",
                  population.mass_density.min(), population.mass_density.max())
            print("Doppler table shape:", population.fractional_doppler_widths.shape)
            print("line-strength factor shape:",
                  population.line_strength_population_factors.shape)
            """,
        ),
        markdown(
            r"""
            ## 11.7 The atmosphere sampling grid is not a synthesis grid

            Atmosphere iteration needs an economical estimate of how the whole
            radiation field transports energy. It does not need a
            publication-resolution spectrum at every pass. The solver
            therefore uses 30,000 logarithmically spaced opacity samples across
            a broad wavelength range.

            Four strict `<` tests move the blue starting point beyond expensive
            ionization-edge regions that contribute negligibly in cooler
            models. Exact inequality matters at the threshold temperatures.
            Wavelength increases and frequency decreases. There are no
            resolving-power samples, continuum-edge triplets, or requested
            detector-window crop; those belong to the synthesis grid built in
            Chapter 10.
            """
        ),
        code(
            """
            grid = sampling_grid_checkpoint()
            print("samples:", grid.wavelength_nm.size)
            print("wavelength endpoints [nm]:",
                  grid.wavelength_nm[0], grid.wavelength_nm[-1])
            print("wavelength increasing:", grid.wavelength_increasing)
            print("frequency decreasing:", grid.frequency_decreasing)
            print("positive total frequency weight:", grid.frequency_weights.sum())
            print()
            for temperature, start in zip(
                grid.strict_threshold_temperatures,
                grid.strict_threshold_start_indices,
            ):
                print(f"{temperature:22.16g} K -> start index {start}")
            """,
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch11-two-atmosphere-grids-v1.png"
                   alt="Conceptual comparison of the fixed outer-to-inner 80-layer standard Rosseland coordinate and the separate direct 30,000-sample atmosphere wavelength grid, whose starting coverage changes across five effective-temperature branches while frequency runs opposite to wavelength.">
              <figcaption><strong>Conceptual coordinate schematic.</strong>
              The standard 80-layer coordinate organizes seed
              microturbulence and later remapping. The direct 30,000-point
              wavelength grid organizes atmosphere opacity sampling; its
              coverage changes with effective temperature, while its sample
              count does not.</figcaption>
            </figure>
            """
        ),
        markdown(
            r"""
            ## 11.8 Adapt populations to the continuum interface

            The continuum kernels do not need the entire 1006-slot runtime
            state. `ContinuumAtmosphereState` is a typed adapter containing the
            18 columns they actually consume. This makes the interface smaller
            without recomputing physics.

            CH and OH illustrate why aliases must be explicit. They occupy
            fixed packed molecular slots, and their named fields are views of
            the partition-normalized population table—not independent chemical
            solutions that could drift out of agreement.
            """
        ),
        code(
            """
            continuum = continuum_checkpoint()
            print("adapter fields:", len(continuum.adapter_field_names))
            print(", ".join(continuum.adapter_field_names))
            print("active continuum references:",
                  continuum.active_reference_indices.size)
            """,
        ),
        markdown(
            r"""
            At each depth and frequency, extinction has two physical pieces:
            true absorption \(\kappa_\nu\), which exchanges energy with matter,
            and scattering \(\sigma_\nu\), which redirects photons. The
            continuum call allocates complete `(80, 30000)` float64 slabs for
            absorption, scattering, and the thermal source. This is the first
            chapter calculation whose natural working set is the full
            atmosphere-frequency rectangle, so the memory ledger is part of
            the scientific design rather than an afterthought.
            """
        ),
        code(
            """
            ledger = opacity_memory_ledger()
            print("absorption:", continuum.absorption.shape, continuum.absorption.dtype)
            print("scattering:", continuum.scattering.shape, continuum.scattering.dtype)
            print("source:    ", continuum.source.shape, continuum.source.dtype)
            print("all finite:",
                  all(np.all(np.isfinite(values)) for values in (
                      continuum.absorption, continuum.scattering, continuum.source
                  )))
            print("opacity nonnegative:",
                  np.all(continuum.absorption >= 0.0)
                  and np.all(continuum.scattering >= 0.0))
            print("three continuum slabs [MiB]:",
                  ledger.three_float64_continuum_slabs_bytes / 2**20)
            """,
        ),
        markdown(
            r"""
            ## 11.9 Selection uses 344 packed reference coordinates

            Millions of source lines cannot all be deposited on every pass.
            Selection asks whether a line could rise above a small fraction of
            the local continuum anywhere in the atmosphere. Its reference
            threshold is
            \(10^{-3}(\kappa_{\nu,c}+\sigma_{\nu,c})\) divided by the
            stimulated-emission factor. A line below this level at all
            reference points cannot materially change the opacity-sampling
            solution.

            Two design choices are hiding in that sentence. The first is the
            number 344. Testing every line against all 30,000 frequencies at
            every depth would cost about as much as the deposit we are trying to
            avoid, so the filter instead evaluates a sparse set of reference
            coordinates spanning the range. The second is the direction of the
            error. Selection is a *conservative* filter: keeping a line that
            turns out to be negligible costs a little arithmetic, while dropping
            one that mattered silently removes opacity that nothing downstream
            will flag. The test therefore asks whether a line could rise above
            the threshold *anywhere* in the atmosphere, not whether it typically
            does.

            The threshold is float32 because the selected-line route consumes
            it in that dtype. The source field
            `wavelength_bin_edges` is historically named: it actually stores
            packed logarithmic wavelength indices. Entry 343 duplicates
            reference wavelength 342 and carries the sentinel \(2^{30}\).

            That duplicate-plus-sentinel pair is a performance construction of
            the same family as the `prange` work we met in Chapter 2. A scan that
            walks the reference coordinates comparing each entry against the next
            one would normally need a bounds test on every iteration to avoid
            running off the end. Padding the array with a repeated coordinate
            whose packed index exceeds any real value means the comparison itself
            terminates the walk, so the innermost loop carries no branch. It is a
            small thing that reappears constantly in kernels that run millions of
            times.
            """
        ),
        code(
            """
            print("threshold:", continuum.threshold.shape, continuum.threshold.dtype)
            print("reference:", continuum.reference_wavelength_nm.shape)
            print("packed coordinates:", continuum.packed_reference_indices.shape,
                  continuum.packed_reference_indices.dtype)
            print("duplicated final wavelength:",
                  continuum.reference_wavelength_nm[343]
                  == continuum.reference_wavelength_nm[342])
            print("terminal packed sentinel:",
                  continuum.packed_reference_indices[343])
            """,
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch11-select-once-recompute-opacity-v1.png"
                   alt="Conceptual lifecycle in which atomic, diatomic, TiO, and H2O sources create one selected catalog object, detailed transitions load into a second object, and two atmosphere states reuse both objects while producing different line-opacity slabs and separate OpacityState outputs; H3+ is an explicit optional path.">
              <figcaption><strong>Conceptual lifecycle schematic.</strong>
              Catalog membership is a discrete first-pass decision. The exact
              objects can be reused by a changed atmosphere, but populations
              and widths still require a newly accumulated line-opacity slab
              before each <code>OpacityState</code> enters Chapter 12.</figcaption>
            </figure>
            """
        ),
        markdown(
            r"""
            ## 11.10 Select membership; recompute opacity

            Selection and deposition have different lifetimes. Source
            wavelength, species, excitation energy, and oscillator strength
            determine whether a record belongs to the selected catalog for a
            stellar model. The current temperature, density, population, and
            Doppler width determine how much opacity that selected record
            contributes on this particular pass.

            The compact source view covers observed atomic, diatomic, TiO, and
            water records in source order. H3+ is absent unless a caller
            supplies its optional path. A separate detailed-transition subset
            validates the XLINOP record interface, while IFOP(17) is off in the
            normal compact run because cold compilation of that larger
            parallel specialization is not a practical reader requirement.
            Ordinary selected-line opacity is nevertheless evaluated on the
            full 80-by-30,000 grid.
            """
        ),
        code(
            """
            selected = selection_checkpoint()
            print("SelectedLineCatalog fields:",
                  selected.selected_catalog_field_names)
            print("LineTransitionCatalog fields:",
                  selected.transition_catalog_field_names)
            print("selected membership:", selected.selected_line_count)
            print("validated detailed records:", selected.detailed_line_count)
            print("detailed accumulation enabled:",
                  selected.detailed_accumulation_enabled)
            print("contributing records:", selected.contributing_line_count)
            """,
        ),
        markdown(
            r"""
            Each line touches only a bounded wavelength neighborhood around its
            center. The kernel therefore deposits directly into one shared
            depth-by-frequency slab rather than allocating an impossible
            `(depth, line, frequency)` cube. When enabled, ordinary selected
            and detailed-transition routes add into that same slab. The
            compiled selected-line path returns the source-defined float32
            array; the no-line allocation branch begins as float64, so the
            observed dtype is a branch contract, not a universal assumption.
            """
        ),
        code(
            """
            line = selected.line_mass_absorption_coefficient
            print("line slab:", line.shape, line.dtype)
            print("finite:", np.all(np.isfinite(line)))
            print("nonnegative:", np.all(line >= 0.0))
            print("nonzero cells:", np.count_nonzero(line))
            print("peak [cm^2 g^-1]:", np.max(line))
            print("line slab [MiB]:", line.nbytes / 2**20)
            """,
        ),
        markdown(
            r"""
            ## 11.11 One pass ends at `OpacityState`

            `OpacityState` is the first object that contains all inputs to the
            monochromatic transfer reduction: grids and weights, continuum
            absorption and scattering, the source function, line-selection
            references, reusable catalog objects, the line-opacity slab, the
            current population state, and the Rosseland lookup history.

            On pass 1 that lookup is empty because no new Rosseland mean has
            yet been finalized. The object is large, mutable working state. It
            is not the compact schema-v4 product that synthesis is allowed to
            retain.
            """
        ),
        code(
            """
            opacity = opacity_pass_checkpoint()
            print("OpacityState fields:", len(opacity.opacity_state_field_names))
            print(", ".join(opacity.opacity_state_field_names))
            print("depth x frequency:", opacity.continuum_shape)
            print("line shape:", opacity.line_shape)
            print("continuum / line dtype:",
                  opacity.continuum_dtype, opacity.line_dtype)
            print("continuum + line [MiB]:",
                  (opacity.continuum_bytes + opacity.line_bytes) / 2**20)
            print("first-pass Rosseland entries:", opacity.rosseland_entry_count)
            print("schema-v4 product:", opacity.schema_v4_product)
            """,
        ),
        markdown(
            r"""
            Catalog membership is label- and source-dependent, so the caller
            may carry catalog objects into a later pass. The changed
            atmosphere must still rebuild populations, continuum, thresholds,
            and line opacity. Reuse is explicit object threading—not a hidden
            cache that could accidentally pair old opacity with a new
            structure.
            """
        ),
        code(
            """
            reused = reuse_checkpoint()
            print("temperature changed:", reused.temperature_changed)
            print("selected catalog object reused:",
                  reused.selected_catalog_same_object)
            print("detailed branch inactive:",
                  reused.detailed_branch_inactive)
            print("line opacity changed:", reused.line_opacity_changed)
            print("maximum line-opacity change:",
                  reused.maximum_line_opacity_change)
            """,
        ),
        markdown(
            r"""
            ## 11.12 Why blocking photons heats the interior

            Adding lines to the opacity does not change what the atmosphere is
            required to do. Every layer must still carry the same total flux
            outward,

            \[
            F=\sigma_{\rm SB}T_{\rm eff}^4 ,
            \]

            because energy is neither created nor destroyed on the way out.
            What lines change is how hard that requirement is to satisfy — and
            following that through predicts the temperature structure.

            Deep down, transport is diffusive and the flux obeys

            \[
            F=-\frac{16\sigma_{\rm SB}T^3}{3\kappa_R\rho}
              \frac{dT}{dr},
            \]

            with \(\kappa_R\) the harmonic Rosseland mean we will assemble in
            §12.10. Harmonic weighting means the *transparent* frequencies
            dominate: radiation leaks out through whatever gaps the continuum
            leaves open. Lines close those gaps. A forest of bound-bound peaks
            raises \(\kappa_\nu\) across a large share of the weighting
            function \(\partial B_\nu/\partial T\), and \(\kappa_R\) rises with
            it.

            Now hold \(F\) fixed and read the diffusion relation backwards. A
            larger \(\kappa_R\) at the same flux forces a steeper
            \(|dT/dr|\). Integrating that steeper gradient inward from a
            surface temperature that barely moves, every deep layer ends up
            *hotter* than it would be with the continuum alone. This is
            **backwarming**, and it is the reason Chapter 1's grey atmosphere
            was only ever a scaffold: a real line-blanketed star runs hotter
            inside than its grey twin of the same \(T_{\rm eff}\).

            The outer layers do the opposite. Where the atmosphere is
            optically thin, line cores are efficient radiators that let photons
            escape directly from high, cool material. Flux blocked in the
            interior emerges through those cores from further out, so the
            outermost layers run *cooler* than grey — surface cooling, the
            companion effect to backwarming.

            This also explains why Chapter 13 needs to exist. The blanketed
            opacity we are about to hand off changes the flux the next pass
            actually measures. The gap between that flux and
            \(\sigma_{\rm SB}T_{\rm eff}^4\) is precisely the residual
            \(\delta_H\) that drives the temperature correction.

            ## 11.13 Lines turn a smooth continuum into a blanketing field

            Bound-bound transitions create a dense forest of narrow and broad
            opacity peaks on top of the smoother continuum.

            At one representative depth, the plot compares continuum
            extinction with continuum plus line absorption. The calculation
            retains all 30,000 points; plotting merely strides them for
            legibility. The next stage will reduce this blanketed field into
            the flux, radiation pressure, and heating columns that can reshape
            the atmosphere.
            """
        ),
        code(
            """
            blanketing = blanketing_checkpoint()
            fig, ax = single_panel()
            ax.loglog(
                blanketing.wavelength_nm,
                blanketing.continuum_extinction,
                label="continuum extinction",
            )
            ax.loglog(
                blanketing.wavelength_nm,
                blanketing.blanketed_extinction,
                label="continuum + lines",
                alpha=0.85,
            )
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel(r"Extinction [cm$^2$ g$^{-1}$]")
            ax.set_title(f"Blanketing at depth index {blanketing.depth_index}")
            add_quiet_grid(ax)
            ax.legend()
            plt.show()

            print("peak blanketed / continuum ratio:",
                  blanketing.line_to_continuum_peak_ratio)
            print("handoff: OpacityState -> transfer reductions")
            """,
        ),
        markdown(
            r"""
            The peak ratio above is large — the strongest line centres sit tens of
            times above the continuum. It is worth asking what that does to the
            harmonic Rosseland mean of §11.12, since that is the quantity the
            temperature structure actually responds to.
            """
        ),
        code(
            """
            ratio = (
                blanketing.rosseland_blanketed
                / blanketing.rosseland_continuum_only
            )
            covered = np.mean(
                blanketing.blanketed_extinction
                > 1.1 * blanketing.continuum_extinction
            )

            fig, ax = single_panel()
            ax.plot(blanketing.temperature_k, ratio, marker=".", lw=1.2)
            ax.axhline(1.0, color="#4d5966", lw=1.0, ls="--",
                       label="continuum only")
            ax.set_xlabel("Layer temperature [K]")
            ax.set_ylabel(r"$\\kappa_R$ blanketed / continuum only")
            ax.set_title("Strong lines, small effect on the harmonic mean")
            add_quiet_grid(ax)
            ax.legend()
            plt.show()

            print(f"largest kappa_R increase: {100 * (ratio.max() - 1):.3f}%")
            print(f"sampled frequencies where lines exceed 10% of continuum:"
                  f" {100 * covered:.3f}%")
            """,
        ),
        markdown(
            r"""
            The ratio never drops below one — adding opacity cannot lower a
            harmonic mean whose weight is positive everywhere, so the direction of
            §11.12's argument is confirmed. But the size is only a fraction of a
            percent, and the second printed number explains why: these lines
            exceed the continuum across a very small share of the sampled
            frequencies. A harmonic mean is dominated by its most transparent
            windows, so opacity that is tall but narrow barely moves it.

            That is a genuine property of the physics, not a defect of the
            fixture, and it carries two lessons. Real line blanketing becomes
            structurally important through *coverage* rather than peak height: it
            takes a dense forest of millions of transitions, of the kind Chapters
            7 and 8 build, to close enough windows to reshape the temperature
            profile. And the same insensitivity is what makes opacity sampling
            viable at all — a mean that shrugged at narrow features would be
            hostage to exactly which 30,000 frequencies we happened to sample.

            This compact state carries a selected subset, so read the figure as
            the mechanism confirmed and bounded, not as the full blanketing of a
            real stellar atmosphere.
            """
        ),
        markdown(
            r"""
            ## 11.14 Chapter summary

            A trustworthy atmosphere pass begins with boundaries, not with a
            correction formula:

            - the seed validator rejects malformed physics but does not
              sanitize or impose the later 80-layer line-opacity gate;
            - run setup owns strict controls and the all-zero-only,
              in-place microturbulence fill;
            - pass 1 keeps the supplied pressure; hydrostatic support updates
              begin on later passes;
            - fixed-column format/parse is a measurable, idempotent
              quantization operator;
            - the current EOS and molecular state feed an exact, direct
              30,000-frequency continuum calculation;
            - packed 344-column thresholds choose line membership, while the
              current atmosphere determines the bounded line-opacity slab;
            - the chapter stops at `OpacityState`, before radiation,
              thermodynamics, convection, correction, convergence, or
              schema-v4 promotion.

            ### Next: let radiation reshape the atmosphere

            `OpacityState` says how strongly each layer interacts with each
            sampled frequency, but it does not yet say how much energy or
            momentum radiation carries through that layer. [Chapter 12 turns
            this exact state into integrated radiation, thermodynamic, and
            convective columns.](/reader.html?ch=12)
            """
        ),
    ]
    return notebook(cells)
