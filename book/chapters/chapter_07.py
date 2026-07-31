"""Chapter 7: atomic line forests and special profiles."""

from __future__ import annotations

from book.cells import code, markdown, notebook, setup_cell


def build_notebook() -> dict:
    """Construct the causal atomic-forest chapter."""

    cells = [
        markdown(
            r"""
            # Atomic Line Forests and Special Profiles

            *Stellar Spectroscopy from Scratch — from physical principles to a working code*

            Chapter 6 followed one ordinary Fe I transition from two energy levels
            into a checked opacity profile. A real catalog is not merely that
            calculation repeated in a loop. The source contains nearly two million
            atomic records; broad lines can matter even when their centers lie outside
            the requested wavelength interval; many centers land on the same pixel;
            and hydrogen, helium, and autoionizing resonances require different
            profile physics.

            A dense array with one axis for depth, one for line, and one for
            wavelength would make this concrete but impossible. With 80 depths,
            \(2\times10^6\) records, and \(2.5\times10^4\) wavelength samples,

            \[
            80(2\times10^6)(2.5\times10^4)(4\ {\rm bytes})
            \approx 16\ {\rm TB}.
            \]

            We need a representation that never creates that line axis. This chapter
            asks one question:

            > **How do we turn a heterogeneous source catalog into one correct
            > atomic-opacity slab without discarding broad wings or losing
            > overlapping contributions?**

            The construction has two movements. First we compile ordinary records,
            conservatively select them, and deposit their centers and wings sparsely.
            Then we route hydrogen, deuterium, helium, and autoionizing records through
            the physics they actually need. Throughout the chapter,
            \(K^{\rm line}_{d,w}\) means line mass absorption coefficient in
            cm\(^{2}\) g\(^{-1}\), with depth index \(d\) and wavelength index \(w\).
            It is still opacity—not intensity or flux.

            <figure>
              <img src="assets/schematics/textbook/ch07-catalog-to-grid-v1.png"
                   alt="A raw atomic ledger passes through energy, isotope, damping, and conservative wavelength-window decisions before records are arranged by route type and grid center. One broad line centered outside the window is retained because its wing reaches inside.">
              <figcaption><strong>From source rows to usable records.</strong>
              A raw row is not deposited directly. Its physical meanings are decoded,
              missing damping information is completed, a conservative window keeps
              reachable wings, and only then are records organized for accumulation.
              The rails at right represent data locality, not a spectrum.</figcaption>
            </figure>

            > **Movement I — Build the ordinary atomic forest.** We begin with the
            catalog decisions that must be identical before any CPU or device kernel
            evaluates a profile.
            """
        ),
        setup_cell(),
        code(
            """
            from time import perf_counter

            import matplotlib.pyplot as plt
            import numpy as np

            from book.chapter06_runtime import (
                configure_local_data_paths,
                run_synthesis_one_line,
            )
            configure_local_data_paths()
            from book.chapter07_runtime import (
                atomic_source_checkpoint,
                autoionizing_checkpoint,
                catalog_layout_checkpoint,
                compare_forest_wing_modes,
                hydrogen_checkpoint,
                record_transform_checkpoint,
                route_ledger,
                run_atomic_forest,
                scatter_add_checkpoint,
                window_selection_checkpoint,
            )
            from book.chapter07_teaching import (
                njit_keep_mask,
                prange_keep_mask,
                python_keep_mask,
            )
            from book.plot_style import PAPER_COLORS, add_quiet_grid, single_panel
            from payne_zero_synthesis.atomic_lines import Grid

            regime_names = (
                "hot_dwarf",
                "solar_dwarf",
                "low_gravity_giant",
                "cool_molecule_rich",
            )
            depth_index = 2

            def finish_plot(axes, *, xlabel, ylabel, title, ylog=False):
                axes.set_xlabel(xlabel)
                axes.set_ylabel(ylabel)
                axes.set_title(title)
                if ylog:
                    axes.set_yscale("log")
                add_quiet_grid(axes, axis="both" if ylog else "y")
                plt.show()
                plt.close(axes.figure)
            """,
            tags=("book-setup", "hide-input"),
        ),
        markdown(
            r"""
            ## 7.1 A catalog row is a claim, not yet a kernel input

            A source row combines measurements, theoretical calculations, and compact
            routing fields. The important raw names are explicit:
            `stored_wavelength_nm`, `species_code`, two term energies,
            `raw_log_oscillator_strength`, three damping logs, isotope corrections,
            principal quantum numbers, `line_size`, and `line_category_tag`.

            The chapter uses a self-contained 133-row teaching subset. It contains
            128 strong ordinary records near 499 nm, including the Fe I transition
            checked in Chapter 6, plus one example each of H I, D I, He I, He II,
            and an `AUT` autoionizing record. The rows are copied field-for-field
            from a checksum-pinned 1,939,975-row catalog; the compact file is a
            teaching slice, not a claim that the real source is small.

            The first calculation checks both identities and counts. Route code 0 is
            ordinary. Negative codes and code 1 will earn their meanings later.
            """
        ),
        code(
            """
            source = atomic_source_checkpoint()

            print(f"teaching records:       {source.record_count:,}")
            print(f"full catalog records:   {sum(source.full_catalog_route_counts.values()):,}")
            print(f"source catalog SHA-256: {source.source_catalog_sha256[:16]}...")
            print("teaching route counts:")
            for route_code, count in sorted(source.route_counts.items()):
                print(f"  type {route_code:2d}: {count:3d}")
            """,
        ),
        markdown(
            r"""
            ## 7.2 Decode physical meanings before selecting wavelengths

            The record compiler applies a causal sequence.

            First, isotope corrections modify the logarithmic strength:

            \[
            \log(gf)=\log(gf)_{\rm raw}
            +\Delta_{\rm iso,1}+\Delta_{\rm iso,2}.
            \]

            Second, absolute term energies identify the lower and upper levels.
            Energy-shift subfields are applied before their difference determines the
            wavelength,

            \[
            \lambda_l=\frac{10^7}
            {\left|(\widetilde E_2+\delta\widetilde E_2)
                   -(\widetilde E_1+\delta\widetilde E_1)\right|}
            +10^{-4}\delta_{\rm isotope}
            \quad [{\rm nm}].
            \]

            If the corrected energy difference is zero, the stored wavelength is the
            fallback. A nonzero difference uses the level-derived value, so the
            rounded source wavelength is not automatically the indexing authority.

            Finally, nonzero damping logs become linear coefficients. A blank
            radiative, Stark, or van der Waals field means “use the declared physical
            default,” not “this line has no damping.” Ordinary coefficients are
            normalized by the implementation's \(4\pi\nu\) convention, while their
            raw logs remain available because a special route may interpret the
            source columns differently.

            The deuterium example makes the isotope operation visible:
            \(\Delta_{\rm iso}=-5\) changes its strength by exactly \(10^{-5}\).
            It does not change the hydrogen abundance solver; it changes this
            catalog transition's weight.
            """
        ),
        code(
            """
            transformed = record_transform_checkpoint()

            print(f"records with isotope correction: {transformed.isotope_changed_count}")
            print(f"radiative defaults filled:      {transformed.default_radiative_count}")
            print(f"Stark defaults filled:          {transformed.default_stark_count}")
            print(f"van der Waals defaults filled:  {transformed.default_van_der_waals_count}")
            print(f"D I source row:                 {transformed.deuterium_source_row}")
            print(
                "D I log(gf): "
                f"{transformed.deuterium_log_gf_without_correction:.2f} -> "
                f"{transformed.deuterium_log_gf_with_correction:.2f}"
            )
            print(f"D I strength ratio:             {transformed.deuterium_strength_ratio:.1e}")
            print(
                "largest |stored-derived lambda|:  "
                f"{transformed.maximum_stored_minus_derived_wavelength_nm:.3e} nm"
            )
            """,
        ),
        markdown(
            r"""
            A maximum wavelength adjustment of only a few \(10^{-5}\) nm may appear
            unimportant. At a native grid resolution of \(R_{\rm grid}=300{,}000\),
            however, a pixel is only about \(1.7\times10^{-3}\) nm wide near 499 nm.
            Discrete indexing should therefore use one precise host calculation, not
            whatever rounding happens on each device.

            ## 7.3 Build the requested grid once, then grow context

            The synthesis grid is geometric:

            \[
            \lambda_j=\lambda_0 r^j,
            \qquad r=1+\frac{1}{R_{\rm grid}}.
            \]

            Constant \(R_{\rm grid}\) is approximately constant velocity spacing.
            Wavelengths and the integer line-center decisions are built in host
            float64. The public requested samples are constructed exactly once.
            Sixteen context samples are grown outward by repeated division and
            multiplication by \(r\); they are removed only after every opacity source
            has been accumulated.

            Why not rebuild one wider grid? A slightly different first wavelength can
            move later float64 samples by a few ulps. Growing from the exact endpoints
            keeps the requested interior bitwise unchanged.
            """
        ),
        code(
            """
            requested_grid = Grid(498.95, 499.15, 300_000.0)
            requested_wavelength = requested_grid.build()
            ratio = requested_grid.ratio

            blue = requested_wavelength[0] / ratio ** np.arange(16, 0, -1)
            red = requested_wavelength[-1] * ratio ** np.arange(1, 17)
            synthesis_wavelength = np.concatenate(
                (blue, requested_wavelength, red)
            )
            output_slice = slice(16, 16 + requested_wavelength.size)

            assert np.array_equal(
                synthesis_wavelength[output_slice],
                requested_wavelength,
            )
            print(f"requested samples: {requested_wavelength.size}")
            print(f"internal samples:  {synthesis_wavelength.size}")
            print("requested interior bitwise unchanged: True")
            """,
        ),
        markdown(
            r"""
            ## 7.4 A wavelength window must keep reachable wings

            A line is not a delta function at its center. The raw `line_size` code
            selects one margin from

            \[
            [100,\ 30,\ 10,\ 3,\ 1,\ 0.3,\ 0.1]\ {\rm nm}.
            \]

            Hydrogen and deuterium always receive the 100 nm class. For an ordinary
            record, the code maps `line_size` into a clipped class. Above 500 nm the
            margins scale by \(\lambda_{\rm start}/500\); the same scaled margin is
            used on both sides. The inclusive keep rule is

            \[
            \lambda_l\geq\lambda_{\rm start}-m_l
            \quad\land\quad
            \lambda_l\leq\lambda_{\rm end}+m_l.
            \]

            This is a catalog-window decision. It says that a profile *could* reach
            the output interval; it does not yet say that the line is strong in the
            current atmosphere.
            """
        ),
        code(
            """
            selected = window_selection_checkpoint()
            outside = selected.selected_with_margin & ~selected.center_inside

            print(f"centers inside requested window: {selected.center_inside_count}")
            print(f"records kept with margins:       {selected.selected_count}")
            print(f"outside centers still retained:  {selected.outside_center_but_selected_count}")

            distance_to_window = np.where(
                selected.wavelength_nm < selected.start_wavelength_nm,
                selected.start_wavelength_nm - selected.wavelength_nm,
                selected.wavelength_nm - selected.end_wavelength_nm,
            )
            figure, axes = single_panel(height=3.8)
            axes.scatter(
                distance_to_window[outside],
                selected.margin_nm[outside],
                s=34,
                facecolors="none",
                edgecolors=PAPER_COLORS["orange"],
            )
            limit = 0.105
            axes.plot(
                (0.0, limit),
                (0.0, limit),
                color=PAPER_COLORS["slate"],
                linestyle="--",
                label="distance = allowed margin",
            )
            axes.fill_between(
                (0.0, limit),
                (0.0, limit),
                limit,
                color=PAPER_COLORS["blue"],
                alpha=0.07,
                label="retained",
            )
            axes.set_xlim(0.0, limit)
            axes.set_ylim(0.0, limit)
            axes.legend(loc="lower right")
            finish_plot(
                axes,
                xlabel="center distance beyond nearest window edge (nm)",
                ylabel="allowed selection margin (nm)",
                title="Every retained outside center lies within its permitted margin",
            )
            """,
        ),
        markdown(
            r"""
            Only 45 centers lie inside the 0.20 nm requested interval, but 100
            records survive the conservative test. The 55 open markers are not
            mistaken inclusions: every center lies in the shaded region where its
            distance beyond the nearest edge is no larger than its allowed margin.

            ## 7.5 A route code chooses physics

            The record compiler classifies each transition before the profile kernels
            run. A source tag and an implemented method are not the same thing.

            - Type 0 is the ordinary LTE/Harris route from Chapter 6.
            - Type 1 is an autoionizing resonance.
            - Type 3 carries a `PRD` source tag but currently uses the ordinary LTE
              route. No partial-frequency-redistribution transfer is solved.
            - Type 2 carries a `COR` tag but is explicitly skipped in both current
              opacity lanes.
            - Negative codes route H, D, and helium families.

            The full pinned source contains no type-2 `COR`, type-3 `PRD`, or
            type-\(-4\) He-3 record, but the compiler and route ledger still state what
            would happen if those codes were present. “Available in a parser” must
            never be reported as “contributes opacity.”
            """
        ),
        code(
            """
            print(
                f"{'type':>5s}  {'family':18s}  "
                f"{'synthesis route':32s}  {'teaching rows':>13s}"
            )
            for code_value, family, synthesis_route, atmosphere_route, count in route_ledger():
                print(
                    f"{code_value:5d}  {family:18s}  "
                    f"{synthesis_route:32s}  {count:13d}"
                )

            print("\\nImportant coupling:")
            print("do_metal=True  -> ordinary type 0/3 AND autoionizing type 1")
            print("do_metal=False -> neither branch")
            """,
        ),
        markdown(
            r"""
            The name `do_metal` is historical implementation language, not a literal
            statement that every routed record is chemically a metal. Its true public
            behavior includes autoionizing opacity.

            ## 7.6 Compile a structure of arrays, not a list of objects

            Once a line passes the window test, its invariant fields are stored in
            parallel one-dimensional arrays: wavelength, \(gf\), lower excitation,
            damping coefficients, species, ion stage, route, and principal quantum
            numbers. This **structure of arrays** makes one operation over many lines
            contiguous.

            The geometric grid and host-float64 wavelengths set two integer targets:
            the nearest center column and a raw wing anchor. Population columns follow
            the exact public layout from Chapter 3,

            \[
            i_{\rm element}=Z-1,\qquad
            i_{\rm ion}=r-1.
            \]

            A catalog may preserve source order (`sort="catalog"`) or use a stable
            `(line_type, center)` ordering for locality. The order is part of the
            numerical contract because float32 sums are not associative. Our teaching
            forest uses `type_center` so the segment is visible; the standard complete
            pipeline records its selected sort in cache metadata.
            """
        ),
        code(
            """
            layout = catalog_layout_checkpoint()

            print(f"geometric-grid samples:       {layout.wavelength_nm.size}")
            print(f"selected catalog records:     {layout.line_count}")
            print(f"type segments:                {layout.type_segments}")
            print(f"centers actually on grid:     {layout.on_grid_center_count}")
            print(f"unique on-grid center pixels: {layout.unique_on_grid_center_count}")
            print(f"extra colliding records:      {layout.colliding_on_grid_record_count}")

            invariant_fields = (
                "wavelength_nm", "oscillator_strength", "lower_excitation_cm",
                "radiative_damping", "stark_damping",
                "van_der_waals_damping", "atomic_number", "ion_stage",
                "line_type",
            )
            print("\\nwindow invariants:", ", ".join(invariant_fields))
            print("per-atmosphere state: populations, T, rho, ne, perturbers, continuum")
            """,
        ),
        markdown(
            r"""
            Fifteen of the 45 on-grid records share a center pixel with another
            record. That is not an edge case; it is already common in this tiny
            high-resolution slice.

            ## 7.7 Selection happens at three different physical stages

            We have completed the first selection: *can this profile reach the
            requested wavelength interval?* Two later decisions prevent unnecessary
            work without changing that conservative answer.

            The CPU atmosphere first compresses an enormous packed source stream.
            For each species and frequency bin it forms the maximum, over depth, of
            a partition-normalized population/width factor divided by the local
            continuum selection threshold. A standard record is kept only if

            \[
            R_{s,b}>R_{\min},\qquad
            Q_l\geq1,\qquad
            Q_l B_l\geq1.
            \]

            Here \(Q_l\) is the float32 center ratio and \(B_l\) is the deepest-layer
            Boltzmann lookup. The first comparison is strict; the other two are
            inclusive. This source-level maximum is conservative across depth.

            After a record is loaded for synthesis, each depth-line pair faces a
            local decision. Its population, width, and density must be positive, and
            both its pre-excitation and post-excitation strengths must reach
            \(10^{-3}\) of the continuum at the clamped center. The three stages
            answer different questions and must not share one ambiguous word
            “cutoff.”
            """
        ),
        code(
            """
            minimum_ratio = np.float32(0.25)
            population_ratio = np.array(
                [minimum_ratio, np.nextafter(minimum_ratio, np.float32(np.inf)), 1.0],
                dtype=np.float32,
            )
            center_ratio = np.array([2.0, 1.0, np.nextafter(np.float32(1.0), 0.0)])
            boltzmann = np.array([1.0, 1.0, 2.0], dtype=np.float32)

            keep = (
                (population_ratio > minimum_ratio)
                & (center_ratio >= 1.0)
                & (center_ratio * boltzmann >= 1.0)
            )
            print("record                  population > min   center >= 1   product >= 1   keep")
            labels = ("strict-boundary", "inclusive-boundary", "center-below-one")
            for index, label in enumerate(labels):
                clauses = (
                    population_ratio[index] > minimum_ratio,
                    center_ratio[index] >= 1.0,
                    center_ratio[index] * boltzmann[index] >= 1.0,
                )
                print(f"{label:23s} {str(clauses[0]):>16s} {str(clauses[1]):>13s} "
                      f"{str(clauses[2]):>14s} {str(bool(keep[index])):>6s}")
            """,
        ),
        markdown(
            r"""
            The boundary table is executable documentation: equality fails the
            population-envelope clause but passes the two strength clauses.

            ## 7.8 `njit` and `prange` accelerate independent decisions

            The full atmosphere selector can examine roughly \(10^8\) packed
            records. A Python loop would repeatedly interpret the same operations.
            Numba's `@njit` compiles an array-specialized machine-code loop. With
            `nogil=True`, the compiled body does not hold Python's global interpreter
            lock. With `parallel=True`, `prange` distributes iterations across CPU
            threads.

            Parallelization is safe only when iterations do not race. In this mask
            kernel, line \(l\) reads its own inputs and writes only `keep[l]`.
            There is no shared sum. `fastmath` remains off because reassociation could
            flip a record at one of the boundaries above.

            The three small functions below have identical clauses. The production
            selector additionally decodes packed words and gathers lookup tables, but
            the parallel reasoning is the same.
            """
        ),
        code(
            """
            from numba import njit, prange

            def python_keep_mask(population, center, boltzmann, minimum):
                keep = np.zeros(population.size, dtype=np.bool_)
                for line in range(population.size):
                    keep[line] = (
                        population[line] > minimum
                        and center[line] >= 1.0
                        and center[line] * boltzmann[line] >= 1.0
                    )
                return keep

            @njit(nogil=True)
            def njit_keep_mask(population, center, boltzmann, minimum):
                keep = np.zeros(population.size, dtype=np.bool_)
                for line in range(population.size):
                    keep[line] = (
                        population[line] > minimum
                        and center[line] >= 1.0
                        and center[line] * boltzmann[line] >= 1.0
                    )
                return keep

            @njit(nogil=True, parallel=True)
            def prange_keep_mask(population, center, boltzmann, minimum):
                keep = np.zeros(population.size, dtype=np.bool_)
                for line in prange(population.size):
                    keep[line] = (
                        population[line] > minimum
                        and center[line] >= 1.0
                        and center[line] * boltzmann[line] >= 1.0
                    )
                return keep
            """,
            tags=("textbook-teaching-code",),
        ),
        code(
            """
            generator = np.random.default_rng(7)
            sample_count = 250_000
            population = generator.lognormal(0.0, 1.0, sample_count).astype(np.float32)
            center = generator.lognormal(0.0, 1.2, sample_count).astype(np.float32)
            boltz = generator.uniform(0.02, 1.0, sample_count).astype(np.float32)

            started = perf_counter()
            serial_cold = njit_keep_mask(population, center, boltz, 0.25)
            serial_cold_seconds = perf_counter() - started
            started = perf_counter()
            parallel_cold = prange_keep_mask(population, center, boltz, 0.25)
            parallel_cold_seconds = perf_counter() - started

            timings = {}
            for name, function in (
                ("Python", python_keep_mask),
                ("warm njit", njit_keep_mask),
                ("warm prange", prange_keep_mask),
            ):
                started = perf_counter()
                result = function(population, center, boltz, 0.25)
                timings[name] = perf_counter() - started
                assert np.array_equal(result, serial_cold)
            assert np.array_equal(parallel_cold, serial_cold)

            print(f"cold njit compile + run:   {serial_cold_seconds:.4f} s")
            print(f"cold prange compile + run: {parallel_cold_seconds:.4f} s")
            for name, seconds in timings.items():
                print(f"{name:12s}: {seconds:.5f} s")
            print("all masks byte-identical: True")
            """,
        ),
        markdown(
            r"""
            Cold timing includes compilation; warm timing measures reuse. On a small
            array, thread setup can make `prange` slower than serial `njit`. The
            correct lesson is not “parallel is always faster.” It is that a sufficiently
            large independent line axis can be distributed after equality has been
            proved, and the crossover must be measured on the actual machine.

            The atmosphere deposit kernel has a second shortcut because its public
            boundary always contains exactly 80 layers. It checks depths 8, 16,
            ..., 80 first. An interior seven-layer block is visited only when a
            bounding gate is active. This is a work gate, not a new observable:
            `selected_line_count` means a line touched at least one sampled depth
            gate, not that every depth deposited and not a number of nonzero pixels.
            """
        ),
        code(
            """
            sampled_depth = np.arange(7, 80, 8)       # zero-based 8, 16, ..., 80
            toy_center_ratio = np.zeros(80)
            toy_center_ratio[15] = 1.4               # one active sampled gate
            toy_center_ratio[16:23] = 2.0            # its neighboring interior block
            gate_active = toy_center_ratio[sampled_depth] >= 1.0

            visited = np.zeros(80, dtype=bool)
            visited[sampled_depth] = True
            for block in range(9):
                left_gate = gate_active[block]
                right_gate = gate_active[min(block + 1, gate_active.size - 1)]
                start = 8 * block + 8
                stop = min(start + 7, 80)
                if left_gate or right_gate:
                    visited[start:stop] = True

            print(f"sampled gate depths:          {sampled_depth + 1}")
            print(f"active sampled gates:         {sampled_depth[gate_active] + 1}")
            print(f"depths visited in toy line:   {np.flatnonzero(visited) + 1}")
            print("count meaning: line touched a sampled gate; it is not a pixel count")
            """,
        ),
        markdown(
            r"""
            ## 7.9 Overlapping lines require scatter-add

            After selection, a live line center is only a triplet:

            \[
            (d,\ j_l,\ k_{d,l}),
            \]

            where \(j_l\) is the target wavelength column. The desired slab is

            \[
            K^{\rm line}_{d,w}
            =\sum_{l:j_l=w} k_{d,l}.
            \]

            Assignment is wrong when two records share a column: the later value
            replaces the earlier one. `np.add.at`, Torch `index_put_` with
            `accumulate=True`, and `index_add_` express the segmented sum.
            The small collision below makes the difference visible before we trust a
            million-line kernel.
            """
        ),
        code(
            """
            scatter = scatter_add_checkpoint()

            print("target columns: ", scatter.columns)
            print("contributions:  ", scatter.contributions)
            print("assignment:     ", scatter.overwritten)
            print("scatter-add:    ", scatter.scatter_added)
            print(f"shared pixel 4: {scatter.overwritten[4]:.1f} -> "
                  f"{scatter.scatter_added[4]:.1f}")
            assert np.array_equal(scatter.scatter_added, scatter.reduced)
            """,
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch07-scatter-add-reduction-v1.png"
                   alt="Several narrow and broad line profiles point to shared wavelength pixels where their values are added. Separate private CPU opacity ribbons are then summed in a fixed reduction into one positive line-opacity forest.">
              <figcaption><strong>No lost overlaps.</strong> Several records may
              target one wavelength cell, so their values add. On the CPU atmosphere
              route, parallel chunks instead write private slabs and a later ordered
              reduction combines them. The schematic promises no lost update; it does
              not promise identical floating-point association on every backend.</figcaption>
            </figure>

            ## 7.10 Parallel deposits need ownership, not hopeful scheduling

            A mask is easy to parallelize because each iteration owns one output
            element. An opacity deposit is different: many lines write the same
            \((d,w)\) cell.

            The CPU atmosphere route gives each Numba line chunk a private float32
            \((D,W)\) buffer. `prange` may fill those buffers concurrently because no
            two threads own the same buffer. A serial loop then adds buffers in chunk
            order. The synthesis route uses scatter/index-add for narrow deposits and
            explicit walks for wide wings; its complete pipeline processes metal
            invariant chunks of 40,000 records and adds returned slabs sequentially.

            Private buffers remove data races. Fixed chunk order makes the parallel
            CPU regrouping deterministic for a fixed thread configuration, but it does
            not reproduce serial line-by-line association exactly. That last-bit
            distinction belongs in a numerical tolerance, not under the word
            “identical.”
            """
        ),
        code(
            """
            print("private buffer 0:", scatter.private_buffers[0])
            print("private buffer 1:", scatter.private_buffers[1])

            reduced_in_fixed_order = np.zeros(scatter.reduced.size)
            for chunk_index in range(scatter.private_buffers.shape[0]):
                reduced_in_fixed_order += scatter.private_buffers[chunk_index]

            assert np.array_equal(reduced_in_fixed_order, scatter.scatter_added)
            dense_bytes = 80 * 2_000_000 * 25_000 * 4
            sparse_slab_bytes = 80 * 25_000 * 4
            print(f"dense (D,L,W) estimate: {dense_bytes / 1e12:.1f} TB")
            print(f"one float32 (D,W) slab: {sparse_slab_bytes / 1e6:.1f} MB")
            print("fixed private-buffer reduction preserves every contribution: True")
            """,
        ),
        markdown(
            r"""
            ## 7.11 The exact compact ordinary forest

            We now compose the steps on one supplied solar atmosphere. The compiler
            keeps 100 ordinary records for the conservative 498.95–499.15 nm window.
            The exact synthesis accumulator maps them to population columns, applies
            the Chapter 6 excitation, cutoff, center-profile, wing, and once-only
            stimulated-emission policies, and returns a float32 \((6,121)\) slab.

            The plot shows one depth only. It is intentionally a single panel: the
            claim is that the checked Fe I line is now embedded in a sum of overlapping
            records. Continuum is not added to the ordinate, so every visible feature
            is atomic line absorption.
            """
        ),
        code(
            """
            forest = run_atomic_forest("solar_dwarf", runtime_device="cpu")
            opacity = forest.net_line_mass_absorption_coefficient[depth_index]
            positive_floor = max(float(np.max(opacity)) * 1.0e-7, 1.0e-12)

            figure, axes = single_panel()
            axes.plot(
                forest.wavelength_nm,
                np.maximum(opacity, positive_floor),
                color=PAPER_COLORS["blue"],
            )
            fe_wavelength = 499.034119461782
            fe_column = int(np.argmin(np.abs(forest.wavelength_nm - fe_wavelength)))
            axes.annotate(
                "Chapter 6 Fe I",
                xy=(forest.wavelength_nm[fe_column], max(opacity[fe_column], positive_floor)),
                xytext=(22, -44),
                textcoords="offset points",
                arrowprops={"arrowstyle": "-", "color": PAPER_COLORS["slate"]},
                color=PAPER_COLORS["slate"],
            )
            finish_plot(
                axes,
                xlabel="wavelength (nm)",
                ylabel=r"net line mass absorption (cm$^2$ g$^{-1}$)",
                title=f"One checked line inside a {forest.metal_line_count}-record atomic forest",
                ylog=True,
            )

            print(f"center collisions: {forest.center_collision_count}")
            print(f"nonzero samples by depth: {forest.nonzero_count_per_depth}")
            print(f"work -> accumulation dtype: {forest.work_dtype} -> {forest.accumulation_dtype}")
            """,
        ),
        markdown(
            r"""
            A single pixel may contain several centers plus neighboring wings. The
            result therefore cannot be reconstructed from a list of isolated peak
            heights; the additive slab is the physical object needed by transfer.

            ## 7.12 Wings divide into narrow and wide work

            An ordinary line that passes its center cutoff walks outward until its
            opacity drops below a continuum-relative threshold or reaches a bounded
            grid limit. The current narrow/wide split is:

            \[
            {\rm maximum\ reach}\leq128\ {\rm pixels}
            \quad\Rightarrow\quad {\rm batched\ narrow\ walker}.
            \]

            Narrow work is grouped into reach tiers 1, 8, 32, and 128. Wider lines
            use an explicit walk. The distinction changes temporary shape and launch
            strategy, not the profile equation.

            We can see both cases without inventing a new line. The same Fe I record
            from Chapter 6 reaches different distances in different atmospheric
            regimes because populations, damping perturbers, Doppler width, and the
            continuum cutoff change with depth.
            """
        ),
        code(
            """
            reach_by_regime = {}
            for name in regime_names:
                one_line = run_synthesis_one_line(name, runtime_device="cpu")
                reach_by_regime[name] = one_line.wing_reach

            figure, axes = single_panel()
            depth = np.arange(6)
            for name, reaches in reach_by_regime.items():
                axes.plot(depth, reaches, marker="o", label=name.replace("_", " "))
            axes.axhline(
                128,
                color=PAPER_COLORS["orange"],
                linestyle="--",
                linewidth=1.3,
                label="narrow/wide boundary",
            )
            axes.set_xticks(depth)
            axes.legend(ncol=2)
            finish_plot(
                axes,
                xlabel="depth index (0 = outermost)",
                ylabel="maximum deposited reach (grid pixels)",
                title="One line can cross the narrow–wide boundary as its state changes",
            )
            """,
        ),
        markdown(
            r"""
            The cool molecule-rich atmosphere crosses 128 pixels at one depth; the
            hot dwarf is absent in its three deepest supplied layers after the local
            cutoff. The route is driven by state, not by a permanent “wide line” label.

            ## 7.13 Batched and explicit wing walks must agree

            Optimization is accepted only after the scalable batched walk is compared
            with the explicit loop on the same records and state. Because both routes
            accumulate float32 values in different groupings, we declare the
            comparison contract before looking at the result:

            \[
            |\Delta K|\leq5\times10^{-7},
            \qquad
            \frac{|\Delta K|}{\max(|K|,10^{-30})}\leq2\times10^{-6}.
            \]

            These are compact CPU-fixture tolerances, not universal promises for a
            different device or catalog.
            """
        ),
        code(
            """
            absolute_tolerance = 5.0e-7
            relative_tolerance = 2.0e-6
            wing_comparison = compare_forest_wing_modes("solar_dwarf")

            print(
                "maximum absolute difference: "
                f"{wing_comparison.maximum_absolute_difference:.3e}"
            )
            print(
                "maximum relative difference: "
                f"{wing_comparison.maximum_relative_difference:.3e}"
            )
            assert wing_comparison.maximum_absolute_difference <= absolute_tolerance
            assert wing_comparison.maximum_relative_difference <= relative_tolerance
            print("declared float32 wing-walk contract passed: True")
            """,
        ),
        markdown(
            r"""
            The ordinary forest is now scalable and checked. We have not forced all
            records through it.

            > **Movement II — Route the profiles that are not ordinary.** A route
            code is useful only if it leads to the physics that justified the code.

            <figure>
              <img src="assets/schematics/textbook/ch07-profile-routing-v1.png"
                   alt="One atomic source catalog fans into ordinary, hydrogen and deuterium, helium, and autoionizing profile routes, whose positive opacity contributions rejoin one shared atomic-opacity budget.">
              <figcaption><strong>One catalog, several profile families.</strong>
              Ordinary lines use the checked core-and-wing route. H and D require
              fine structure, strong plasma broadening, and series logic. Helium has
              its own wing and optional continuum-merge treatment. Autoionizing
              states couple to a continuum and produce an asymmetric profile. The
              routes reunite only through addition to the opacity slab.</figcaption>
            </figure>

            ## 7.14 Hydrogen and deuterium are line series

            Hydrogen profiles cannot be summarized by one ordinary damping ratio.
            The principal quantum numbers `n_lower` and `n_upper` identify a series
            transition. Fine-structure components share its strength but are shifted
            in frequency and combined with tabulated weights. Electrons create strong
            impact and quasi-static Stark wings; neutral H, He, and H2 add
            perturbations; temperature and microturbulence set the Doppler core; and
            the continuum controls the stopping threshold.

            The teaching subset contains H-alpha and its deuterium counterpart. Their
            level labels are both \(2\rightarrow3\), but their source energies place
            them 0.1716 nm apart. The seven H-alpha fine-structure weights sum to one,
            so they redistribute the transition strength rather than create it.
            """
        ),
        code(
            """
            hydrogen = hydrogen_checkpoint()
            frequency_offset_mhz = hydrogen.component_offset_hz / 1.0e6

            figure, axes = single_panel(height=3.8)
            markerline, stemlines, baseline = axes.stem(
                frequency_offset_mhz,
                hydrogen.component_weight,
                basefmt=" ",
            )
            plt.setp(markerline, color=PAPER_COLORS["blue"], markersize=6)
            plt.setp(stemlines, color=PAPER_COLORS["blue"], linewidth=1.6)
            finish_plot(
                axes,
                xlabel="fine-structure frequency offset (MHz)",
                ylabel="component weight",
                title="H-alpha strength is distributed over seven components",
            )

            print(f"H-alpha wavelength:  {hydrogen.hydrogen_wavelength_nm:.6f} nm")
            print(f"D-alpha wavelength:  {hydrogen.deuterium_wavelength_nm:.6f} nm")
            print(f"isotope separation:  {hydrogen.isotope_separation_nm:.6f} nm")
            print(f"sum of weights:      {np.sum(hydrogen.component_weight):.12f}")
            print(f"f(2 -> 3):           {hydrogen.oscillator_strength_2_to_3:.9f}")
            """,
        ),
        markdown(
            r"""
            The support boundary is lane-specific. The synthesis H/D engine accepts
            resolved neutral records with `n_lower >= 2`; a Lyman record with
            `n_lower < 2` raises `NotImplementedError`. The CPU atmosphere transition
            route accepts principal levels 1 through 100 and its dedicated evaluator
            includes Lyman-alpha resonance and quasi-molecular cutoff terms.

            Therefore the honest statement is:

            > Lyman lines are guarded out of the synthesis H/D engine; they are not
            > absent from the entire atmosphere implementation.

            ## 7.15 Dense plasma dissolves the top of a series

            At high \(n_{\rm upper}\), hydrogen levels crowd toward a continuum edge.
            Plasma electric fields broaden and dissolve the highest distinguishable
            members. Treating every member as forever isolated would leave an
            artificial comb.

            The exponent in the formula below looks arbitrary, but it is not — it
            follows from comparing two lengths that both depend on \(n\).

            Consecutive hydrogen levels crowd together as \(n\) grows: the term
            values go as \(1/n^2\), so the spacing between neighbours falls as

            \[
            \Delta\widetilde\nu_{\rm spacing}\propto n^{-3}.
            \]

            Meanwhile the levels themselves become easier to perturb. A hydrogen
            orbital of principal quantum number \(n\) has radius \(\propto n^2\), and
            hydrogen's degeneracy gives it a *linear* Stark effect, so a level splits
            in an ambient field \(E\) by

            \[
            \Delta\widetilde\nu_{\rm Stark}\propto n^2 E .
            \]

            The field comes from the neighbouring charges. With mean separation
            \(d\sim n_e^{-1/3}\), the typical microfield is
            \(E\sim e/d^2\propto n_e^{2/3}\).

            A line stops being a distinguishable member of the series once its Stark
            splitting exceeds the distance to its neighbour. Setting the two equal,

            \[
            n^2 n_e^{2/3}\sim n^{-3}
            \qquad\Longrightarrow\qquad
            n^5\sim n_e^{-2/3}
            \qquad\Longrightarrow\qquad
            n\propto n_e^{-2/15},
            \]

            which is exactly the density dependence used below. The \(2/15\) is not a
            fitted constant; it is \(2/3\) divided by \(5\).

            The synthesis catalog marks merged-continuum records with
            `n_upper == 99`. Their one atmosphere-dependent invariant follows the
            Inglis–Teller construction

            \[
            n_{\rm IT}=\frac{1600}{n_e^{2/15}},\qquad
            n_{\rm merge}=\max(n_{\rm IT}-1.5,1),
            \]

            \[
            \widetilde\nu_{\rm merge}
            =\frac{109737.312}{n_{\rm merge}^2}\quad [{\rm cm}^{-1}].
            \]

            Every other catalog invariant can be reused for another star, but this
            array must be recomputed from that star's electron-density profile.
            """
        ),
        code(
            """
            inglis_teller_level = 1600.0 / hydrogen.electron_density_cm3 ** (2.0 / 15.0)
            merge_level = np.maximum(inglis_teller_level - 1.5, 1.0)

            figure, axes = single_panel()
            axes.plot(
                hydrogen.electron_density_cm3,
                merge_level,
                color=PAPER_COLORS["blue"],
            )
            axes.set_xscale("log")
            axes.set_yscale("log")
            finish_plot(
                axes,
                xlabel=r"electron density $n_e$ (cm$^{-3}$)",
                ylabel=r"last distinguishable level $n_{\\rm merge}$",
                title="Increasing electron density dissolves lower series members",
                ylog=True,
            )

            print(
                "merge wavenumber range: "
                f"{hydrogen.merge_wavenumber_cm[0]:.2f} -> "
                f"{hydrogen.merge_wavenumber_cm[-1]:.2f} cm^-1"
            )
            print(
                "synthesis lower-level support: "
                f"n_lower >= {hydrogen.synthesis_minimum_supported_lower_level}"
            )
            print(
                "atmosphere level validation: "
                f"{hydrogen.atmosphere_supported_lower_level_range}"
            )
            """,
        ),
        markdown(
            r"""
            The CPU atmosphere represents dissolved-series records differently: a
            nonnormal negative transition type carries its last-level value, and the
            kernel deposits the associated pseudo-continuum over a bounded interval.
            The two lanes share the physical idea but not one universal encoding.

            ## 7.16 Helium and autoionizing resonances preserve their source meanings

            The synthesis helium route recognizes He I (type \(-3\)), He-3 I
            (\(-4\)), and He II (\(-6\)). It reuses the population, excitation,
            damping, and cutoff logic already earned, then uses a helium-specific
            wing walk. The implemented He-3 contract is exactly

            \[
            A_{\rm eff}=A/1.155,\qquad
            \Delta\lambda_{D,\rm eff}=1.155\,\Delta\lambda_D,\qquad
            a_{\rm eff}=a/1.155.
            \]

            This is a source contract, not a general mass-scaling derivation.
            Per-depth `helium_core_weight_grid` and `helium_tail_weight_grid` can
            taper a line into a continuum edge. The standard packaged optical route
            currently supplies neither array, so zero grids select the untapered
            behavior.

            Type-1 autoionizing records describe a discrete state coupled to a
            continuum. Their preserved raw damping columns acquire special meanings:
            a raw radiative log sets the frequency scale, the raw Stark field supplies
            an asymmetry parameter \(q\), and the raw van der Waals field supplies a
            baseline \(w\). The implemented ratio is

            \[
            \Phi(x)=\frac{qx+w}{x^2+1}\frac{1}{w},
            \qquad
            x=\frac{2(\nu-\nu_l)}{\gamma_{\rm rad}}.
            \]

            The synthesis walker keeps only positive, above-cutoff, contiguous
            samples. The actual teaching row has extremely small \(q/w\) and is
            nearly symmetric. To expose the role of asymmetry, the plot holds the
            center scale fixed and varies the dimensionless ratio \(q/w\); these
            controlled curves are explanatory, while the printed parameters are the
            exact source record.
            """
        ),
        code(
            """
            auto = autoionizing_checkpoint()
            reduced_offset = auto.reduced_frequency_offset

            figure, axes = single_panel()
            for ratio, color in zip(
                (0.0, 0.15, 0.40),
                (PAPER_COLORS["slate"], PAPER_COLORS["blue"], PAPER_COLORS["orange"]),
            ):
                controlled = (ratio * reduced_offset + 1.0) / (
                    reduced_offset**2 + 1.0
                )
                axes.plot(
                    reduced_offset,
                    np.maximum(controlled, 0.0),
                    color=color,
                    label=rf"$q/w={ratio:.2f}$",
                )
            axes.legend()
            finish_plot(
                axes,
                xlabel=r"reduced frequency offset $x$",
                ylabel=r"positive profile ratio $\\max[\\Phi(x),0]$",
                title="Continuum coupling makes the two wings unequal",
            )

            print(f"AUT source row:       {auto.source_row}")
            print(f"wavelength:           {auto.wavelength_nm:.6f} nm")
            print(f"radiative width:      {auto.radiative_width:.6e}")
            print(f"source asymmetry q:   {auto.shore_asymmetry:.6e}")
            print(f"source baseline w:    {auto.shore_baseline:.6e}")
            print("COR type 2: parsed, skipped")
            print("PRD type 3: ordinary LTE opacity, not redistribution")
            """,
        ),
        markdown(
            r"""
            ## 7.17 One atomic opacity slab

            The routes now have a precise additive meaning:

            \[
            \begin{aligned}
            K^{\rm atomic}
            ={}&K^{\rm ordinary\ type\ 0/3}
              +K^{\rm auto\ type\ 1}\\
             &+K^{\rm He\ type\ -3/-4/-6}
              +K^{\rm H/D\ type\ -1/-2}
              +K^{\rm merged\ series}.
            \end{aligned}
            \]

            Each branch owns its profile, stopping rule, and once-only stimulated
            emission. They share a float32 wavelength slab because opacity sources
            add. Type-2 `COR` contributes zero; type-3 `PRD` contributes through the
            ordinary LTE profile and does not alter the later transfer equation.

            The CPU atmosphere and device-synthesis lanes implement this sum with
            different representations. The atmosphere first selects packed words,
            then uses compiled Numba line chunks with private slabs and fixed-order
            reduction. Synthesis compiles a structure-of-arrays catalog, fixes
            discrete indices on the host, and uses Torch scatter/index-add plus
            bounded wing walks. We require each lane to reproduce its own declared
            authority; we do not pretend their intermediate slabs are bitwise
            interchangeable.

            ## 7.18 Chapter summary

            1. A raw line becomes usable only after isotope and energy corrections,
               wavelength construction, damping completion, and route classification
               have explicit meanings.
            2. A conservative wavelength margin retains profiles whose centers lie
               outside the requested interval but whose wings can enter it.
            3. Atmosphere source selection and synthesis depth-line selection remove
               different kinds of impossible work; neither replaces window selection.
            4. A sparse \((D,W)\) slab replaces an impossible \((D,L,W)\) tensor.
               Scatter-add or private-buffer reduction preserves overlapping lines.
            5. Narrow and wide wing paths change work shape, not line physics, and
               their float32 agreement must be measured under a declared tolerance.
            6. Hydrogen/deuterium, helium, and autoionizing records require their own
               profiles and source meanings. `COR` remains unwired, while a `PRD`
               label currently follows ordinary LTE opacity.
            7. The result is one checked atomic line-absorption slab. It is not yet
               emergent intensity or flux.

            ### Next: Molecular sources do not share one encoding

            Atomic records now share a trustworthy route into one opacity slab. Cool
            stellar spectra add molecular bands, but their sources do not share one
            encoding or one correction rule. [Chapter 8 compiles those heterogeneous
            molecular sources into the same kind of sparse, checked
            contribution.](/reader.html?ch=8)
            """
        ),
    ]
    return notebook(cells)


if __name__ == "__main__":
    build_notebook()
