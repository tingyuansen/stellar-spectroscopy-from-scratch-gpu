"""Canonical Chapter 10 notebook: complete structured-atmosphere synthesis."""

from __future__ import annotations

from book.cells import code, markdown, notebook, setup_cell


TITLE = "GPU Synthesis from a Structured Atmosphere"


def build_notebook() -> dict:
    """Return the deterministic Chapter 10 notebook document."""

    cells = [
        markdown(
            r"""
            # Chapter 10 — GPU Synthesis from a Structured Atmosphere

            A stellar atmosphere does not look like a spectrum. It is a stack
            of layers: each layer has a temperature, density, composition, and
            populations of absorbers. A spectrum appears only after we ask,
            wavelength by wavelength, how those layers absorb, emit, scatter,
            and finally release radiation through the surface.

            Chapters 3–9 built the ingredients of that calculation one at a
            time. We can now ask the question that turns them into a usable
            forward model:

            > How does one supplied structured atmosphere become one native
            > total, continuum, and normalized spectrum without changing the
            > physics at their boundaries?

            There are two ways to answer this question, and we need both. The
            *physical answer* is a chain from populations to opacity, from
            opacity to the transfer equation, and from the transfer solution
            to emergent flux. The *computational answer* decides which arrays
            can be reused, which must be rebuilt for each star, where their
            precision changes, and when they cross between host memory and an
            accelerator. If either answer is incomplete, a fast result can be
            wrong without looking obviously wrong.

            The organizing idea is that the calculation contains two
            lifetimes. A wavelength window owns grids, catalogs, tabulated
            functions, and kernel invariants that can be reused for many
            stars. A star owns temperature- and density-dependent populations,
            broadening quantities, and hydrogen merge state that must be
            rebuilt. Separating those lifetimes gives speed without turning
            star-dependent physics into stale cached data.

            This chapter executes every quantitative spectrum on **CPU in
            float64**. That fixed route is the portable reference against which
            the chapter checks ordering, cache warmth, and public output. The
            same synthesis implementation can select CUDA or Apple Metal
            (MPS); we will explain that device policy and the float32
            accumulation island, but we will not present an unavailable GPU
            as if it generated the figures on this machine.

            The four six-depth states below are integration fixtures inherited
            from Chapter 5 and exposed in memory as native schema v4. They are
            deliberately varied stage inputs, not claims of converged stellar
            atmospheres. Full-catalog production synthesis remains a separate
            optional integration scale. The compact catalogs make the entire
            route executable inside the book; they do not make the resulting
            line list complete.
            """
        ),
        setup_cell(),
        code(
            """
            from dataclasses import fields
            import inspect
            from pathlib import Path

            import matplotlib.pyplot as plt
            import numpy as np
            import torch

            from book.chapter10_runtime import (
                REGIME_LABELS, REGIMES, RESOLUTION,
                WINDOW_END_NM, WINDOW_START_NM, ZOOM_END_NM, ZOOM_START_NM,
                cache_checkpoint, configure_chapter10_runtime,
                end_to_end_checkpoint, fixture_checkpoint,
                four_regime_checkpoint, grid_checkpoint,
                hydrogen_checkpoint, invariant_checkpoint,
                load_regime_atmosphere, memory_checkpoint,
                prewarm_checkpoint, public_spectrum,
                roundtrip_checkpoint, runtime_policy_checkpoint,
                timing_checkpoint,
            )
            from book.plot_style import (
                PAPER_COLORS, add_quiet_grid, single_panel,
            )

            source_catalog_root = configure_chapter10_runtime()
            """,
            tags=("hide-input",),
        ),
        markdown(
            r"""
            ## 10.1 Begin with an interface that preserves meaning

            A synthesis routine could accept a loose collection of arrays, but
            that would leave dangerous questions unanswered. Does an
            “ion population” mean the actual number density of an ion, or that
            number density divided by its partition function? Does a continuum
            edge coordinate increase with frequency or wavelength? Is
            microturbulence stored in km s\(^{-1}\) or cm s\(^{-1}\)?
            Schema v4 answers those questions before a kernel runs.

            The structured schema names 25 required arrays. Depth is \(D\);
            the exact requested native grid has \(W\) wavelengths; the hidden
            synthesis grid has \(W_{\rm synth}=W+32\). Population fields retain
            their Chapter 3/4 meanings and units. In particular,
            `partition_normalized_populations` is not the same object as
            `ion_stage_populations`.

            The fields fall into four causal groups:

            | What synthesis needs | Exact schema-v4 fields | Why they cannot be inferred later |
            | --- | --- | --- |
            | Layer structure | `temperature`, `gas_pressure`, `electron_density`, `mass_density`, `column_mass`, `microturbulence`, `hc_over_kt` | Together they record the thermodynamic structure from which the helper arrays were built; temperature, densities, column mass, and microturbulence also enter synthesis directly. |
            | Chemical state | `elemental_abundances`, `partition_normalized_populations`, `ion_stage_populations`, `fractional_doppler_widths` | Line strengths require both actual ion populations and partition-normalized populations; one cannot safely stand in for the other. |
            | Species-specific fast paths | `hydrogen_neutral_population`, `helium_neutral_population`, `helium_singly_ionized_population`, `molecular_hydrogen_population`, `hydrogen_partition_normalized_ion_stage_populations`, `carbon_partition_normalized_ion_stage_populations`, `magnesium_neutral_partition_normalized_population`, `aluminum_neutral_partition_normalized_population`, `silicon_neutral_partition_normalized_population`, `iron_neutral_partition_normalized_population` | Continuum and line kernels need particular stages often enough that reconstructing them inside every wavelength kernel would repeat work and risk a different convention. |
            | Continuum-edge geometry | `signed_continuum_edge_frequency_hz`, `continuum_edge_wavelength_nm`, `continuum_edge_midpoint_wavelength_nm`, `continuum_edge_interval_width_squared_over_two_nm2` | Signed edge ordering and precomputed intervals select the same bound-free branches as the atmosphere that produced the state. |

            Every depth-dependent field is ordered from the outermost layer to
            the innermost. `column_mass` is in g cm\(^{-2}\) and increases
            inward. Number densities are in cm\(^{-3}\); `mass_density` is in
            g cm\(^{-3}\); `temperature` is in kelvin. Those units matter
            because the pipeline ultimately needs a mass absorption
            coefficient, not merely a cross-section per absorber.

            Notice where the population calculation sits in the causal chain.
            A schema-v4 atmosphere is already *population bridged*: its
            thermodynamic columns and abundances have been expanded into the
            exact ion, partition-normalized, and species-specific population
            fields. `SynthesisPipeline` consumes those fields; it does not
            silently solve a second equation of state. The public builder in
            Section 10.9 owns the columns-to-populations step when a caller
            starts from physical columns rather than a complete archive.

            The checkpoint below asks only whether the interface is intact and
            whether the four supplied states are genuinely different. It does
            not yet calculate radiation.
            """
        ),
        code(
            """
            fixture = fixture_checkpoint()
            atmospheres = {
                regime: load_regime_atmosphere(regime) for regime in REGIMES
            }

            print("schema version:", fixture.schema_version)
            print("required arrays:", fixture.required_array_count)
            print("depth layers:", fixture.depth_count)
            print("fixture SHA-256:", fixture.source_sha256)
            print()
            print("regime                 T_surface [K]   n_e,surface [cm^-3]")
            for regime, electron_density in zip(
                fixture.regimes, fixture.electron_density_surface_cm3
            ):
                print(
                    f"{REGIME_LABELS[regime]:23s}"
                    f"{atmospheres[regime]['temperature'][0]:14.1f}"
                    f"{electron_density:21.6e}"
                )
            """,
        ),
        markdown(
            r"""
            Those different electron densities already predict a systems
            consequence: the window's hydrogen line template can be shared,
            but its Inglis–Teller merge limit cannot. The interface therefore
            tells us not only what data exist, but also where the reusable
            window state must stop.

            ## 10.2 Runtime policy is one compatible pair

            A *device* is the processor and memory space in which the large
            array operations run. PyTorch names an NVIDIA accelerator `cuda`,
            Apple Metal `mps`, and the ordinary processor `cpu`. A *dtype*
            fixes how many bits represent each floating-point number. Device
            and dtype must be selected together: asking for a mathematically
            useful precision is meaningless if the device cannot execute it.

            `resolve_runtime(requested_device=None, requested_dtype=None)`
            chooses CUDA, then MPS, then CPU. Its omitted dtype is float32 on
            MPS and float64 on CUDA or CPU. Float32 remains legal everywhere;
            MPS float64 is rejected explicitly.

            The priority order answers “what is the fastest supported device
            available to this caller?” It does not define the verification
            route in this chapter. All checkpoints below pass
            `device="cpu"` and `dtype="float64"` explicitly. That choice holds
            the arithmetic policy fixed while we study the pipeline. A later
            CUDA or MPS comparison must be reported as a backend comparison,
            with tolerances appropriate to its precision and reduction order,
            rather than quietly replacing the reference result.

            Why is an accelerator useful here? Many wavelength columns can be
            evaluated together, and many lines can have their local profile
            contributions prepared together. PyTorch expresses that independent
            work as tensor batches. “Batched” does not mean that the four stars
            in this chapter become one hidden atmosphere tensor: they are four
            ordinary public calls that reuse one window cache. Inside transfer,
            wavelength rows form the batch while the outward recurrence through
            depth remains ordered. That combination—wide independent batches
            around a short sequential depth sweep—is the source of useful GPU
            parallelism without changing the radiative-transfer equation.
            """
        ),
        code(
            """
            policy = runtime_policy_checkpoint()
            print("machine default")
            print("  device:", policy.default_device)
            print("  dtype: ", policy.default_dtype)
            print("explicit CPU default:", policy.cpu_default_dtype)
            print("CUDA available:", policy.cuda_available)
            print("MPS available: ", policy.mps_available)
            print("MPS float64 rejected:", policy.mps_float64_rejected)

            from payne_zero_synthesis.device import resolve_runtime
            print("exact signature:", inspect.signature(resolve_runtime))
            """,
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch10-context-compute-crop-v1.png"
                   alt="Conceptual native wavelength strip with sixteen blue and sixteen red context samples, all opacity and transfer work on the expanded strip, and a device-side crop restoring the exact requested interior.">
              <figcaption><strong>Context is computation, not output.</strong>
              Kernels see symmetric native samples beyond both requested
              boundaries. The public grid remains bitwise identical to the
              grid requested by the caller.</figcaption>
            </figure>

            ## 10.3 Grow context outward; never regenerate the interior

            For resolving power \(R\), the native geometric step is
            \(\lambda_{i+1}/\lambda_i=1+1/R\). Sixteen samples are grown
            outward from each endpoint. All opacity and transfer work uses
            \(W_{\rm synth}\); `output_slice` crops on the selected device.
            This context is neither an instrumental line-spread function nor
            extra public wavelength coverage.

            Why calculate wavelengths that the caller did not request? A line
            centered just outside the left boundary can have a profile wing
            inside the requested interval. Transfer and any downstream
            spectral operator also need a small neighborhood at each edge.
            If the calculation stopped exactly at the requested endpoints,
            those edge samples would be treated differently from samples in
            the middle of a wider run.

            There is a second, subtler requirement. The caller's \(W\)
            wavelengths are already the public numerical contract. Rebuilding
            the entire enlarged grid from a new starting point could shift
            interior values by a rounding bit. The implementation therefore
            preserves the requested array exactly and grows 16 samples
            outward on each side. Context changes what the kernels can see,
            but it cannot change the public coordinates.

            Here \(R\) is the source spelling `resolution` at the public
            `synthesize` boundary. It is the intrinsic sampling density
            \(R_{\rm grid}=\lambda/\Delta\lambda\), not the resolving power of
            an observed instrument. Instrumental broadening, if requested,
            belongs to a separate `spectral_operator`.
            """
        ),
        code(
            """
            grid = grid_checkpoint()
            assert grid.synthesis_count == grid.requested_count + 32
            assert grid.interior_bitwise_exact

            print("requested W:       ", grid.requested_count)
            print("synthesis W + 32:  ", grid.synthesis_count)
            print("context per side:  ", grid.context_each_side)
            print("device crop slice: ", grid.output_slice)
            print("bitwise interior:  ", grid.interior_bitwise_exact)
            print(
                "measured local R:   ",
                f"{grid.local_resolving_power_min:.6f}",
                "to",
                f"{grid.local_resolving_power_max:.6f}",
            )
            """,
        ),
        markdown(
            r"""
            ## 10.4 Compile the window once

            Consider synthesizing a thousand atmospheres over the same
            wavelength interval. Parsing the same catalog, transferring the
            same quadrature tables, and rebuilding the same line-profile
            constants a thousand times would not improve any physical answer.
            These are properties of the window and runtime, not of a star.

            `WindowInvariants` owns the context grid; compiled atomic and
            molecular catalogs; device-resident continuum, profile, hydrogen,
            molecular, and transfer tables; component flags; and construction
            timings. Its exact key begins with window bounds, resolution,
            molecular enablement, device, dtype, and metal chunk, then includes
            context policy and source/table file identities.

            An *invariant* here means “unchanged while the window identity is
            unchanged,” not “universal physics.” Change the catalog checksum,
            switch molecules off, move from CPU float64 to CUDA float64, or
            alter `resolution`, and the invariant bundle must change. This is
            why a cache key carries both scientific identity and runtime
            identity.

            The active-family flags are computed during construction. They let
            the forward pass skip an absent family without asking the same
            question at every depth and wavelength. In this compact window all
            of the atomic metal, helium, hydrogen, and enabled molecular
            families are active, so the later ordering check uses the
            complete standard composition path.
            """
        ),
        code(
            """
            invariants = invariant_checkpoint()
            print("WindowInvariants fields:", len(invariants.field_names))
            print(", ".join(invariants.field_names))
            print()
            print("atomic lines:   ", invariants.atomic_line_count)
            print("molecular lines:", invariants.molecular_line_count)
            print(
                "active families:",
                {
                    "metal": invariants.has_metal,
                    "helium": invariants.has_helium,
                    "hydrogen": invariants.has_hydrogen,
                    "molecular": invariants.molecular_line_count > 0,
                },
            )
            print("metal chunk:", invariants.metal_chunk)
            print("resident policy:", invariants.device, invariants.dtype)
            for name, seconds in invariants.build_profile.items():
                print(f"{name:31s} {seconds:.6f} s")
            """,
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch10-window-versus-star-state-v1.png"
                   alt="Conceptual split between one reusable window-invariant bundle fed by grids, catalogs, tables and runtime policy, and separate per-star state boxes fed by two structured atmospheres.">
              <figcaption><strong>One window, many stars.</strong> Window
              identity determines the reusable bundle. Each atmosphere still
              supplies its own populations, density, temperature, column mass,
              and hydrogen merge state.</figcaption>
            </figure>

            A useful classification is stricter than “everything lives on the
            GPU.” Most depth arrays remain host float64 until the kernel that
            needs them uploads them. `column_mass` and a persistent temperature
            tensor are device state. Large opacity/source slabs, transfer
            vectors, and their crop remain on the selected device until result
            construction.

            This split follows data size and reuse. A small host array that is
            read once does not become faster merely because it was uploaded
            early. By contrast, repeatedly moving a full
            \((D,W_{\rm synth})\) opacity slab across the host-device boundary
            would erase much of the benefit of accelerator execution. The
            implementation moves each object when its next consumer needs it,
            and keeps the large spectral intermediates resident.
            """
        ),
        code(
            """
            memory = memory_checkpoint()
            invariant_state = (
                "grids", "catalogs", "profile/continuum/transfer tables",
                "metal/helium/hydrogen-template/molecular invariants",
            )
            print("reusable window state")
            for name in invariant_state:
                print("  ", name)
            print("per-star host state")
            for name in memory.host_state_fields:
                print("  ", name)
            print("per-star/device working state")
            for name in memory.device_state_fields:
                print("  ", name)
            print("WindowInvariants retains no atmosphere mapping or D x W slab.")
            """,
        ),
        markdown(
            r"""
            ## 10.5 Caches accelerate a source-defined calculation

            Reuse creates a new failure mode: an old answer can survive after
            one of its inputs changes. A cache is therefore trustworthy only
            when its identity is at least as specific as the calculation it
            accelerates. “Same filename” is not enough. The window, sampling,
            device, dtype, feature switches, source files, and table files all
            participate in the identity of the reusable state.

            The process cache returns the same resident object for an unchanged
            key. Setting
            `PAYNE_ZERO_SYNTHESIS_DISABLE_INVARIANT_CACHE=1` forces a fresh,
            value-identical build. `clear_window_invariant_cache()` drops the
            resident layer. Persistent atomic and compiled-molecular caches
            remain deletable products derived from the source view.

            The important scientific check is not that a warm call is faster.
            It is that the cache kill switch changes object lifetime without
            changing the computed values. A perturbed `resolution` must produce
            a distinct key; an unchanged request must return the same resident
            bundle; and clearing must remove that bundle. Only after these
            identity checks pass do timing differences have an interpretation.
            """
        ),
        code(
            """
            cache = cache_checkpoint()
            checks = {
                "same-key process hit is same object":
                    cache.process_hit_same_object,
                "disable switch builds a new object":
                    cache.disabled_build_new_object,
                "disabled build is value-identical":
                    cache.disabled_values_equal,
                "resolution perturbation changes key":
                    cache.perturbed_key_is_distinct,
                "clear removes process entries":
                    cache.clear_removes_entries,
            }
            assert all(checks.values())
            for name, passed in checks.items():
                print(f"{name:43s} {passed}")
            """,
        ),
        markdown(
            r"""
            Prewarm has a deliberately narrower contract than general runtime
            selection: it builds one molecular-enabled window on CPU float64.
            Its JSON manifest binds platform, Python, Torch, staged source
            fingerprint, window, and exact required artifact paths and hashes.
            It does not persist a GPU-resident invariant bundle.

            Prewarming is useful because parsing and compilation can be paid
            before an interactive or batch run. It does not “save the
            spectrum,” and it does not authorize a result made from different
            sources. A later process still constructs its device-resident
            bundle, but can reuse the checksum-identified persistent products
            whose values do not depend on the atmosphere.
            """
        ),
        code(
            """
            prewarm = prewarm_checkpoint()
            required = prewarm["required_window_artifacts"]
            print("status:", prewarm["status"])
            print("reused this call:", prewarm["reused"])
            print("schema:", prewarm["schema_version"])
            print("CPU float64, molecular enabled:", prewarm["identity"]["molecular_lines"])
            print("source fingerprint:", prewarm["identity"]["source_fingerprint"])
            for name, record in required.items():
                print(
                    f"{name:20s}",
                    Path(record["path"]).name,
                    record["sha256"],
                )
            """,
        ),
        markdown(
            r"""
            Cache validation is not broader than the source. A malformed atomic
            cache falls back to parsing, and molecular compiled caches validate
            their embedded identity. The pinned atomic loader does not compare
            the metadata of every well-formed archive with an independently
            reconstructed expected record. Likewise, explicitly injected
            `WindowInvariants` are checked against the first seven key fields,
            not every file identity. These are documented source limitations,
            not claims that the chapter silently strengthens.

            ## 10.6 Replace the one star-dependent hydrogen field

            Hydrogen illustrates why “compiled once” cannot mean “independent
            of the atmosphere.” At high series order, neighboring hydrogen
            levels become so closely spaced that plasma perturbations merge
            them into a quasi-continuum. The Inglis–Teller limit estimates
            where that merging occurs, and it depends on electron density.
            Since electron density varies with depth and differs between
            stars, the merge wavenumber is a length-\(D\) star-specific field.

            Everything else in the hydrogen template can still be shared. The
            reusable bundle therefore stores a one-depth placeholder that
            carries the correct immutable line and table structure. When a
            `SynthesisPipeline` receives an atmosphere, it computes the
            depth-dependent merge field from that atmosphere's
            `electron_density`.

            The shared template is built with one placeholder depth. Pipeline
            initialization uses `dataclasses.replace` to install
            `merge_wavenumber_by_depth(electron_density)` for the supplied
            atmosphere. The shared object remains unchanged.

            Copy-and-replace is essential here. Mutating the cached template
            would make the second star inherit the first star's plasma state,
            so results would depend on call order. The next check constructs
            hot and cool pipelines from one shared bundle, verifies that their
            merge arrays differ, and then verifies that the template is still
            byte-for-byte unchanged.
            """
        ),
        code(
            """
            hydrogen = hydrogen_checkpoint()
            assert hydrogen.template_unchanged
            assert hydrogen.hot_cool_state_differs

            print("shared template depths:", hydrogen.template_depth_count)
            print("hot-star merge depths:", hydrogen.hot_depth_count)
            print("cool-star merge depths:", hydrogen.cool_depth_count)
            print("hot and cool merge state differs:", hydrogen.hot_cool_state_differs)
            print("shared template unchanged:", hydrogen.template_unchanged)
            """,
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch10-host-device-precision-map-v1.png"
                   alt="Conceptual host and device precision map: host source parsing and float64 discrete controls, selected-device work, a float32 line-opacity and scattering island, device crop, and three final spectral transfers to host float64.">
              <figcaption><strong>Precision and movement are stage
              properties.</strong> Small host control values are distinct from
              full spectral arrays. “One host copy” means one final
              result-construction stage containing three spectral tensor
              conversions, not one fused memory transaction.</figcaption>
            </figure>

            ## 10.7 Bound memory by depth–wavelength slabs and line chunks

            A direct line-by-line thought experiment exposes the memory
            problem. With \(D\) depths, \(L\) lines, and
            \(W_{\rm synth}\) wavelengths, storing every line's contribution
            at every depth and wavelength would require a dense
            \((D,L,W_{\rm synth})\) tensor. Most entries would be negligible
            because one line affects only a local profile neighborhood, yet the
            allocation would scale with all three axes.

            The pipeline instead asks each bounded line chunk to deposit into
            one shared \((D,W_{\rm synth})\) slab. After a chunk has been
            accumulated, its line-local intermediates can be released. The
            peak spectral storage then scales mainly as \(D W_{\rm synth}\),
            while runtime still scales with the lines actually processed.

            The shared line-opacity slab is always float32. Metal lines are
            chunked at 40,000; molecular deposition is independently bounded
            by 500,000 lines and 200,000 line pairs. No stage constructs a
            dense \((D,L,W_{\rm synth})\) tensor. The estimate below is an
            allocation argument from shapes and dtypes, not an allocator
            benchmark.

            The float32 slab is a deliberate precision island on every
            backend, including the CPU float64 reference route. It halves this
            dominant line-accumulation storage relative to float64 and matches
            the production accumulation order. The surrounding work dtype can
            remain float64 on CPU or CUDA. Because floating-point addition is
            not associative, chunk boundaries and family order are part of the
            reproducible numerical method even though they introduce no new
            physics.
            """
        ),
        code(
            """
            memory = memory_checkpoint()
            mib = 1024.0 ** 2
            print("D:", memory.depth_count)
            print("W_synth:", memory.synthesis_wavelength_count)
            print("active compact-catalog lines:", memory.line_count)
            print(
                "one float32 D x W line slab:",
                f"{memory.float32_line_slab_bytes / mib:.3f} MiB",
            )
            print(
                "four float64 D x W slabs:",
                f"{memory.four_float64_depth_wavelength_slabs_bytes / mib:.3f} MiB",
            )
            print(
                "forbidden dense D x L x W example:",
                f"{memory.hypothetical_dense_depth_line_wavelength_bytes / mib:.3f} MiB",
            )
            assert not memory.dense_depth_line_wavelength_allocated
            """,
        ),
        markdown(
            r"""
            ## 10.8 Execute the exact order once

            The next four cells inspect one cached run. The helper calls
            `SynthesisPipeline.run(keep_slabs=True, spectral_operator=None)`;
            it does not reproduce the pipeline. Retaining slabs is diagnostic.
            The public and batched paths use the default `keep_slabs=False`.

            We can now follow one photon-frequency channel through the complete
            composition. At each depth, the structured atmosphere supplies the
            populations and thermodynamic state. Continuum kernels combine
            them into true absorption and scattering. Line kernels add
            frequency-local absorption in a fixed family order. Those
            coefficients and their source functions define the transfer
            problem. Solving that problem through all depths yields the
            emergent Eddington flux \(H_\nu\). Only after the solve is complete
            do we crop the context and convert the public result to
            \(F_\lambda\).

            ### Continuum first: establish both extinction channels

            Continuum absorption \(\kappa_{\nu,\rm c}\) and scattering
            \(\sigma_{\nu,\rm c}\) arrive first with shape
            \((D,W_{\rm synth})\). The returned diagnostic copies have already
            been cropped to \((D,W)\).

            Absorption and scattering must remain separate. Absorption destroys
            a photon and couples the source to local thermal emission.
            Scattering redirects a photon and couples the source to the
            radiation field. Adding them too early would preserve total
            extinction but erase that distinction in the source function.
            The arrays are mass coefficients, so multiplying by a column-mass
            interval produces a dimensionless optical-depth increment.
            """
        ),
        code(
            """
            complete = end_to_end_checkpoint("solar_dwarf")
            expected_slab_shape = (fixture.depth_count, grid.requested_count)
            assert complete.continuum_absorption.shape == expected_slab_shape
            assert complete.continuum_scattering.shape == expected_slab_shape
            assert np.all(np.isfinite(complete.continuum_absorption))
            assert np.all(np.isfinite(complete.continuum_scattering))

            print("continuum absorption:", complete.continuum_absorption.shape)
            print("continuum scattering:", complete.continuum_scattering.shape)
            print("host diagnostic dtype:", complete.continuum_absorption.dtype)
            print(
                "minimum absorption/scattering:",
                complete.continuum_absorption.min(),
                complete.continuum_scattering.min(),
            )
            """,
        ),
        markdown(
            r"""
            ### Lines next: accumulate into one bounded opacity field

            A single float32 line slab is allocated. Ordinary and autoionizing
            metal chunks are added first, then helium. Accumulation order is
            part of floating-point reproducibility: regrouping may change the
            last bits even though chunk size is not a new physical parameter.

            Each catalog record supplies a transition, not a ready-made opacity
            value. The pipeline combines that record with the atmosphere's
            partition-normalized population, actual ion-stage population,
            temperature, microturbulence, and pressure-broadening state. The
            resulting profile contributes only near its shifted line center,
            which is why local deposition can replace a dense line axis.

            Ordinary and autoionizing metal transitions share the first family
            stage even though their profile routes differ internally. Helium
            follows as its own family. This section does not rederive the
            Chapter 7 profiles; its job is to verify that the independently
            checked kernels enter the full calculation once, in the production
            order, and in the same shared slab.
            """
        ),
        code(
            """
            order = complete.stage_order
            metal_index = order.index("ordinary/autoionizing metal chunks")
            helium_index = order.index("helium")
            assert metal_index < helium_index
            assert complete.line_absorption.shape == (
                fixture.depth_count, grid.requested_count
            )
            assert np.all(np.isfinite(complete.line_absorption))
            assert np.all(complete.line_absorption >= 0.0)

            print("shared line slab:", complete.line_absorption.shape)
            print("returned diagnostic dtype:", complete.line_absorption.dtype)
            print("metal before helium:", metal_index < helium_index)
            print("maximum line mass absorption:", complete.line_absorption.max())
            """,
        ),
        markdown(
            r"""
            Hydrogen uses the star-specific merge state just checked. Molecular
            opacity is added only when `molecular_lines=True`; the standard
            synthesis family is the compiled text-band catalog followed by
            TiO. H\(_2\)O and H\(_3^+\) are not deposited by this standard
            synthesis route.

            Hydrogen comes after helium because it has its own series-merging
            and broadening machinery. The molecular stage then uses the
            molecular populations already represented by the supplied
            atmosphere state together with the enabled, compiled source
            catalogs. Setting `molecular_lines=False` removes this molecular
            *line* stage; it does not remove atomic lines or redefine the
            atmosphere.

            The H\(_2\)O boundary is worth stating precisely. A synthesis
            source compiler exists for one H\(_2\)O format, but the standard
            molecular build invoked here does not call it. H\(_3^+\) has no
            standard synthesis compiler or pipeline wiring. We therefore
            describe the route that actually runs—text bands plus TiO—rather
            than treating every available parser or species mass as an active
            opacity source.
            """
        ),
        code(
            """
            hydrogen_index = order.index(
                "hydrogen with star-specific merge state"
            )
            molecular_index = order.index("molecular text + TiO")
            source_index = order.index(
                "LTE Planck line source + zero line scattering"
            )
            assert helium_index < hydrogen_index < molecular_index < source_index

            print("ordered active component tail")
            for stage in order[metal_index:source_index + 1]:
                print("  ", stage)
            print("molecular source lines:", invariants.molecular_line_count)
            print("molecular runtime chunk maximum: 500000 lines")
            print("molecular pair chunk maximum:    200000 pairs")
            """,
        ),
        markdown(
            r"""
            ### Transfer last: let the combined opacity act through depth

            The standard line source is LTE
            \(S_{\nu,\rm l}=B_\nu(T)\), with zero line scattering. Chapter 9's
            transfer solver returns total and continuum \(H_\nu\) together and
            forms their ratio on device. Optional reference-source and spectral
            operator boundaries exist in the source but are not used here.

            In local thermodynamic equilibrium (LTE), matter at temperature
            \(T\) emits line radiation with the Planck function \(B_\nu(T)\).
            The standard route treats line extinction as true absorption, so
            its line-scattering coefficient is zero. Continuum scattering
            remains present and retains the iterated radiation-field
            contribution developed in Chapter 9.

            The paired solve is a numerical safeguard as well as an
            optimization. The *total* channel sees continuum plus line opacity;
            the *continuum* channel sees the same atmosphere and transfer
            discretization with the line contribution omitted. Solving them
            together keeps wavelength grid, depth ordering, angular quadrature,
            and iteration policy aligned. Their ratio then isolates the line
            depression without importing a separately sampled continuum.

            Here \(H_\nu\) is the Eddington flux per unit frequency returned by
            the transfer engine. It is not yet the surface flux \(F_\nu\), and
            it is not yet a per-wavelength quantity.
            """
        ),
        code(
            """
            assert complete.line_source.shape == (
                fixture.depth_count, grid.requested_count
            )
            assert complete.eddington_flux_total_per_frequency.shape == (
                grid.requested_count,
            )
            assert complete.eddington_flux_continuum_per_frequency.shape == (
                grid.requested_count,
            )
            expected_ratio = (
                complete.eddington_flux_total_per_frequency
                / complete.eddington_flux_continuum_per_frequency
            )
            np.testing.assert_allclose(
                complete.normalized_flux, expected_ratio, rtol=3.0e-7, atol=0.0
            )
            print("line source:", complete.line_source.shape, complete.line_source.dtype)
            print("total H_nu:", complete.eddington_flux_total_per_frequency.shape)
            print("continuum H_nu:", complete.eddington_flux_continuum_per_frequency.shape)
            print("finite normalized ratio:", np.all(np.isfinite(expected_ratio)))
            """,
        ),
        markdown(
            r"""
            Device-side cropping occurs before result construction. The final
            stage converts total \(H_\nu\), continuum \(H_\nu\), and normalized
            flux separately to host NumPy float64. With `keep_slabs=False`, all
            four diagnostic slabs are `None`. `SpectrumResult` contains ten
            fields; the public `Spectrum` contains exactly five.

            Cropping before transfer to the host prevents the 32 context
            samples from leaking into the public contract or consuming host
            bandwidth. The phrase “one final host copy” refers to one boundary
            stage, not one fused memory transaction: the total flux,
            continuum flux, and normalized ratio are three distinct tensor
            conversions. Their public NumPy dtype is float64 even when an
            internal backend uses float32 work.

            The diagnostic slabs have a different lifetime. They are useful
            here because they let us inspect the causal chain, but retaining
            four depth-by-wavelength arrays for every production spectrum would
            defeat the memory design. The default `keep_slabs=False` returns
            only the spectral products and timing/operator metadata needed by
            the caller.
            """
        ),
        code(
            """
            print("SpectrumResult field order")
            for name in complete.result_field_names:
                print("  ", name)
            print("public Spectrum field order")
            for name in complete.spectrum_field_names:
                print("  ", name)

            assert complete.wavelength_nm.size == grid.requested_count
            for values in (
                complete.eddington_flux_total_per_frequency,
                complete.eddington_flux_continuum_per_frequency,
                complete.normalized_flux,
            ):
                assert values.dtype == np.float64
            print("three final spectral host arrays: float64")
            print("diagnostic slabs were requested explicitly for this audit")
            """,
        ),
        markdown(
            r"""
            ## 10.9 The public boundary restores \(F_\lambda\)

            A public API should hide execution machinery without hiding
            scientific meaning. `synthesize` accepts one structured atmosphere,
            one intrinsic wavelength window, the molecular-line switch, a
            compatible device/dtype choice, and an optional downstream spectral
            operator. It returns wavelength, total flux, continuum flux,
            normalized flux, and engine time. It does not expose the line
            chunks, context samples, or temporary opacity slabs.

            The exact signature printed below accepts a path, mapping, or
            initialized atmosphere, followed by the wavelength bounds,
            intrinsic `resolution`, molecular switch, runtime policy, and
            optional spectral operator. Mapping inputs are trusted more than
            archive inputs: the normal archival path calls the schema
            validator, whereas a mapping is copied directly.

            The transfer engine works with \(H_\nu\), but most users want
            surface flux per nanometre. First, plane-parallel angular
            integration gives \(F_\nu=4\pi H_\nu\). Next,
            conservation of energy in corresponding intervals requires
            \(F_\lambda\,d\lambda=-F_\nu\,d\nu\). Since
            \(\nu=c/\lambda\), the magnitude of the Jacobian is
            \(\left|d\nu/d\lambda\right|=c/\lambda^2\). With wavelength in nm,
            public flux is
            \(F_\lambda=4\pi H_\nu c_{\rm nm\,s^{-1}}/\lambda_{\rm nm}^2\).
            `flux_total` and `flux_continuum` are therefore surface
            \(F_\lambda\) per nm.

            Both channels receive the same positive \(4\pi c/\lambda^2\)
            factor, so it cancels from
            `normalized_flux = flux_total / flux_continuum`. This ratio check
            is powerful because it spans the internal \(H_\nu\) result and the
            public \(F_\lambda\) boundary. It does not, by itself, prove that
            either absolute flux is physically correct.

            For safety, a persisted atmosphere path is preferable to an
            arbitrary mapping when data come from outside the current process:
            archive loading validates schema version, required names, shapes,
            and finiteness. A mapping is an expert in-memory route and is
            copied without the same full archive validation.
            """
        ),
        code(
            """
            spectrum = public_spectrum("solar_dwarf")
            from payne_zero_synthesis import Spectrum, synthesize

            assert tuple(field.name for field in fields(Spectrum)) == (
                "wavelength_nm", "flux_total", "flux_continuum",
                "normalized_flux", "seconds",
            )
            np.testing.assert_allclose(
                spectrum.normalized_flux,
                spectrum.flux_total / spectrum.flux_continuum,
                rtol=3.0e-7,
                atol=0.0,
            )
            print("signature:", inspect.signature(synthesize))
            print("public samples:", spectrum.wavelength_nm.size)
            print("host dtypes:", spectrum.flux_total.dtype, spectrum.normalized_flux.dtype)
            print("engine seconds:", f"{spectrum.seconds:.6f}")
            """,
        ),
        markdown(
            r"""
            The complementary public builder holds the supplied electron
            density fixed while reconstructing populations from physical
            columns. It is not timed as a production synthesis stage here.
            Saving writes schema v4; loading validates the archive and restores
            the same 25 public arrays.

            This builder answers a practical boundary question: if a caller
            already has `temperature`, `column_mass`, `gas_pressure`,
            `electron_density`, elemental abundances, microturbulence, and mass
            density, can those physical columns be expanded into the exact
            synthesis schema without inventing a second format? The answer is
            yes. The supplied `electron_density` remains a fixed input to this
            construction; the builder does not solve a new atmosphere or
            certify hydrostatic and radiative closure.

            The save–load round trip checks serialization rather than stellar
            physics. Exact field names and exact recovered arrays mean another
            process can consume the same structured state. They do not turn
            the six-layer Chapter 5 fixture into a converged model.
            """
        ),
        code(
            """
            roundtrip = roundtrip_checkpoint()
            from payne_zero_synthesis import (
                build_structured_atmosphere,
                save_structured_atmosphere,
            )

            print("builder signature")
            print(inspect.signature(build_structured_atmosphere))
            print("save signature")
            print(inspect.signature(save_structured_atmosphere))
            print("saved public fields:", roundtrip.saved_field_count)
            print("field names exact:", roundtrip.field_names_exact)
            print("all arrays exact after reload:", roundtrip.arrays_exact)
            print("fixed electron-density sentinel:", roundtrip.fixed_electron_density_seed)
            assert all(
                (
                    roundtrip.field_names_exact,
                    roundtrip.arrays_exact,
                    roundtrip.fixed_electron_density_seed,
                )
            )
            """,
        ),
        markdown(
            r"""
            ## 10.10 Warmth changes cost, not the spectrum

            The first call in a fresh environment may parse source catalogs,
            build derived caches, construct resident invariants, and execute
            the star-dependent forward pass. A later process can reuse the
            persistent derived products but must reconstruct its in-memory
            objects. A later call in the same process can reuse both. Calling
            all three measurements simply “runtime” would mix different
            workloads.

            “Cold source” starts with empty disposable persistent caches;
            “persistent warm” clears only the process bundle; “process warm”
            reuses the resident bundle. Equality is checked before timing is
            interpreted. These wall times are environment-labelled audit
            values, not portable physics. `Spectrum.seconds` starts before
            pipeline initialization and ends after `run`; it excludes public
            \(H_\nu\)-to-\(F_\lambda\) wrapping and is not a per-stage profiler.

            We should predict the result before looking at the bars. Warmth can
            reduce construction cost, but it must not alter wavelength or
            normalized flux. The test therefore requires bitwise-equal
            normalized arrays across all three lifetimes. Only then may the
            height of a bar be interpreted as overhead avoided on this Python,
            NumPy, Torch, and machine combination.
            """
        ),
        code(
            """
            timing = timing_checkpoint()
            assert timing.outputs_equal
            print("bitwise-equal normalized outputs:", timing.outputs_equal)
            print(
                f"Python {timing.python}; NumPy {timing.numpy}; "
                f"Torch {timing.torch}; machine {timing.machine}"
            )

            figure, axis = single_panel(width=6.8, height=3.8)
            colors = (
                PAPER_COLORS["orange"],
                PAPER_COLORS["blue"],
                PAPER_COLORS["green"],
            )
            axis.bar(timing.labels, timing.seconds, color=colors, width=0.68)
            axis.set_ylabel("External wall time [s]")
            axis.set_title("Cache warmth changes construction cost")
            add_quiet_grid(axis, axis="y")
            axis.set_ylim(0.0, 1.16 * float(timing.seconds.max()))
            for index, value in enumerate(timing.seconds):
                axis.text(index, value, f"{value:.3f}", ha="center", va="bottom")
            plt.show()
            """,
        ),
        markdown(
            r"""
            ## 10.11 One window, four supplied states

            The final check keeps window, catalogs, tables, device, dtype, and
            call order fixed while changing only the supplied atmosphere.
            The plot zooms into the compact molecular/ordinary-line region near
            499 nm. Different spectra therefore trace star-dependent state,
            not a change of synthesis algorithm. The compact catalog makes
            this an integration demonstration rather than a full line-list
            prediction. Markers show the actual native samples at
            `resolution=20000`; connecting segments guide the eye and are not
            additional computed wavelengths.

            This comparison closes the two-lifetime argument. Every curve uses
            the same CPU-float64 `WindowInvariants`; each pipeline replaces only
            its star-dependent depth state. Temperature changes the Planck
            source and excitation. Electron density changes ionization,
            continuum processes, and hydrogen merging. Density and
            microturbulence change line strength and width. Molecular
            populations become especially important in the cool supplied
            state. We should therefore expect distinct line depressions even
            though the algorithm and catalog subset are identical.

            We should not over-interpret which curve is “most realistic.” The
            six-layer inputs are controlled integration fixtures, and the
            compact line sources omit most real transitions. The scientific
            result here is conditional: **given** each structured state, the
            complete implementation produces a finite spectrum through the
            exact public route, and changing only that state changes the
            output.
            """
        ),
        code(
            """
            four = four_regime_checkpoint()
            zoom = (
                (four.wavelength_nm >= ZOOM_START_NM)
                & (four.wavelength_nm <= ZOOM_END_NM)
            )
            print("regime                 minimum normalized flux   engine [s]")
            for regime, minimum, seconds in zip(
                four.regimes, four.minimum_normalized_flux, four.process_seconds
            ):
                print(f"{REGIME_LABELS[regime]:23s}{minimum:23.6f}{seconds:13.4f}")

            figure, axis = single_panel(width=7.4, height=4.2)
            palette = (
                PAPER_COLORS["slate"],
                PAPER_COLORS["blue"],
                PAPER_COLORS["magenta"],
                PAPER_COLORS["orange"],
            )
            for regime, values, color in zip(
                four.regimes, four.normalized_flux, palette
            ):
                axis.plot(
                    four.wavelength_nm[zoom], values[zoom],
                    color=color, lw=1.5, marker="o", ms=3.5,
                    markerfacecolor="white", markeredgewidth=0.8,
                    label=REGIME_LABELS[regime],
                )
            axis.set_xlabel("Vacuum wavelength [nm]")
            axis.set_ylabel(r"Normalized flux $F_\\lambda/F_{\\lambda,c}$")
            axis.set_title("The same invariant window responds to each atmosphere")
            axis.legend(frameon=False, ncol=2)
            add_quiet_grid(axis)
            plt.show()
            """,
        ),
        markdown(
            r"""
            ## 10.12 Chapter summary

            1. A synthesis window owns reusable geometric, catalog, table, and
               device invariants; a star owns depth populations, density,
               temperature, column mass, and hydrogen merge state.
            2. The native requested grid is preserved bitwise inside 16 blue
               and 16 red context samples, and cropped on device.
            3. The exact order is continuum; one float32 line slab; metal,
               helium, hydrogen, and molecular deposition; LTE source; paired
               total/continuum transfer; crop; then final result construction.
            4. Chunking bounds temporary work and fixes addition grouping. No
               dense \((D,L,W)\) tensor is part of the architecture.
            5. Most star arrays remain host float64 until kernel upload. “One
               host copy” is one final stage with three separate spectral
               tensor-to-NumPy conversions.
            6. Process and persistent caches accelerate a source-defined
               calculation. Their kill switches and documented validation
               limits do not change the intended physics.
            7. The public five-field `Spectrum` restores
               \(F_\lambda\) per nm and preserves the total/continuum ratio.
            8. Four same-window outputs differ because their supplied
               atmosphere states differ; these fixtures are not evidence of
               atmospheric closure.

            The complete forward boundary is now one causal sentence:
            a schema-v4 structured atmosphere joins checksum-identified
            `WindowInvariants`; its star-specific populations generate
            continuum and an ordered, bounded line slab; paired
            total/continuum transfer runs on the selected device; the context
            is cropped there; and final host-float64 construction produces
            total \(F_\lambda\), continuum \(F_\lambda\), and their normalized
            ratio. In every numerical result shown in this chapter, that
            selected device/dtype pair was CPU float64.

            ### Next: make the supplied atmosphere physical

            We can now answer the chapter's question exactly for a *supplied*
            atmosphere. The schema preserves its meaning, reusable window state
            remains separate from star state, every opacity family enters in a
            bounded order, and transfer returns the public spectrum without an
            early host crossing. But synthesis is a forward diagnostic: it
            cannot make its input hydrostatic, carry the required stellar flux,
            or repair an inconsistent temperature structure.

            The missing object is therefore not another spectrum kernel. It is
            an 80-layer opacity-sampling atmosphere state in which radiation
            can feed back on pressure and temperature. [Chapter 11 begins with
            a validated seed and constructs that blanketed state for one
            physical atmosphere pass.](/reader.html?ch=11)
            """
        ),
    ]
    return notebook(cells)


if __name__ == "__main__":
    build_notebook()
