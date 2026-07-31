"""Canonical Chapter 15 notebook: labels to an honestly verified spectrum."""

from __future__ import annotations

from book.cells import code, markdown, notebook, setup_cell


TITLE = "From Stellar Labels to a Verified Spectrum"


def build_notebook() -> dict:
    """Return the deterministic Chapter 15 capstone notebook."""

    cells = [
        markdown(
            r"""
            # Chapter 15 — From Stellar Labels to a Verified Spectrum

            A stellar label is only a compact description of the problem we
            want to solve. Effective temperature fixes the required outward
            energy flux; surface gravity sets the pull that pressure must
            balance; composition controls particles and opacity. None of
            those labels is itself an atmosphere, and an atmosphere is not
            yet a spectrum.

            The final task is therefore to connect the causal chain without
            erasing its scientific boundaries:

            \[
            \begin{aligned}
            \text{labels}
              &\longrightarrow \text{atmospheric state}\\
              &\longrightarrow \text{populations and opacity}\\
              &\longrightarrow \text{radiative transfer}
                \longrightarrow F_\lambda .
            \end{aligned}
            \]

            There are two legitimate ways to enter that chain. A learned
            initializer can produce a fast starting state for exploration.
            The physical solver can iterate that state toward its fixed point
            and admit a reusable product only after explicit checks. This
            chapter executes both kinds of evidence, but never treats them as
            interchangeable.

            No new physics kernel is introduced here. The intellectual work
            is composition: deciding which object crosses each boundary,
            which claim that object supports, and which unanswered question
            must remain visible.
            """
        ),
        setup_cell(),
        code(
            """
            from dataclasses import fields

            import matplotlib.pyplot as plt
            import numpy as np

            from book.chapter10_runtime import public_spectrum
            from book.chapter15_runtime import (
                EXPECTED_CASE_NAMES, configure_chapter15_runtime,
                dependency_checkpoint, four_regime_capstone,
                initializer_route_checkpoints, load_case_requests,
                public_interface_checkpoint, reproducibility_checkpoint,
                safety_type_checkpoint, verified_solar_checkpoint,
            )
            from book.chapter15_teaching import (
                acceptance_rows, request_digest, summarize_atmosphere,
                summarize_spectrum,
            )
            from book.plot_style import add_quiet_grid, single_panel

            configure_chapter15_runtime()
            requests = load_case_requests()
            routes = {
                item.name: item for item in initializer_route_checkpoints()
            }
            verified_solar = verified_solar_checkpoint()
            """,
            tags=("hide-input",),
        ),
        markdown(
            r"""
            ## 15.1 Exact types prevent accidental promotion

            A numerical array cannot tell us how it was obtained. A smooth
            temperature profile might be a converged solution, a neural
            prediction, or a corrupted file that merely has the right shape.
            The public types therefore carry the scientific role of the
            result.

            `InitializedAtmosphere` and `LabelSpectrum` carry immutable
            convergence and closure flags. For an initializer prediction
            those flags are `False` and `True`: the structure has not passed
            the exact atmosphere loop, so closure is still required. The
            verified route does not disguise the transition with a new
            wrapper. It writes a structured atmosphere, validates its
            schema-v4 arrays, loads the mapping, and only then gives that
            mapping to the unchanged five-field `Spectrum` interface.

            This is a small software choice with a large scientific benefit.
            Downstream code can inspect the result rather than infer its
            trust level from a filename or a plot.
            """
        ),
        code(
            """
            interfaces = public_interface_checkpoint()
            print("ForwardTimings:", interfaces.forward_timing_fields)
            print("InitializedAtmosphere:", interfaces.initialized_atmosphere_fields)
            print("LabelSpectrum:", interfaces.label_spectrum_fields)
            print("Spectrum:", interfaces.spectrum_fields)
            print("initialize:", interfaces.initialize_signature)
            print("synthesize:", interfaces.synthesize_signature)
            """,
        ),
        markdown(
            r"""
            ## 15.2 Choose the workflow by the question

            Exploratory work asks a conditional question: *if this initialized
            atmosphere is a useful approximation, what spectrum follows from
            it?* That is valuable for rapid visualization, debugging,
            optimization, and line-list work. Its speed comes from replacing
            the expensive fixed-point search with a learned proposal.

            Physical work asks a different question: *does this atmospheric
            structure satisfy the coupled equations and the declared
            acceptance tests?* That route must execute the exact iteration,
            preserve its convergence history, validate the final handoff, and
            test the spectrum independently. A plausible exploratory spectrum
            cannot answer those questions, because synthesis tests the
            consequences of a supplied state—not whether the state itself is
            in physical closure.
            """
        ),
        markdown(
            r"""
            <figure>
              <img src="assets/schematics/textbook/ch15-two-workflow-gates-v1.png"
                   alt="Conceptual two-lane workflow beginning with the same requested stellar labels. The exploratory lane returns an explicitly unconverged LabelSpectrum after initialization and synthesis. The verified lane requires checksum-verified catalogs, exact atmosphere passes, convergence and independent checks, and a schema-v4 physical atmosphere before synthesis returns Spectrum; missing catalogs or failed convergence block that product.">
              <figcaption><strong>Two questions require two contracts.</strong>
              The upper lane is a fast exploratory calculation and keeps its
              unverified state visible in the returned type. The lower lane
              admits a physical product only after exact atmosphere iteration
              and independent gates; neither missing catalogs nor failed
              convergence can be hidden by a plausible initializer.</figcaption>
            </figure>
            """
        ),
        code(
            """
            dependencies = dependency_checkpoint()
            for boundary in (
                dependencies.exploratory_boundary,
                dependencies.verified_boundary,
            ):
                print(boundary.mode)
                print("  entrypoint:", boundary.entrypoint)
                print("  returns:   ", boundary.returned_type)
                print("  available: ", boundary.available)
                print("  blocker:   ", boundary.blocker)
            """,
        ),
        markdown(
            r"""
            ## 15.3 Four requests, not four rewritten workflows

            Hot dwarfs, solar-like dwarfs, low-gravity giants, and cool
            molecule-rich dwarfs stress different parts of the calculation,
            but they should not create four copies of the workflow. The hot
            dwarf challenges ionization and the ultraviolet radiation field.
            The giant combines low density with a long pressure scale height.
            The cool dwarf strengthens molecules and makes C, N, and O
            especially influential.

            The algorithm remains one algorithm. Only the requested labels,
            molecular switch, and initializer family vary. This separation is
            important for both science and testing: a regime-specific failure
            should identify a physical limit, not a hidden fork in the code.

            Each request therefore has one canonical digest. The digest binds
            every label and switch that can change the calculation, while
            excluding wall-clock timings and runtime object addresses that
            cannot define a star.
            """
        ),
        code(
            """
            print("case                    Teff    logg  family       projected  digest")
            for request in requests:
                route = routes[request.name]
                print(
                    f"{request.name:24s}"
                    f"{request.effective_temperature:7.0f}"
                    f"{request.log_surface_gravity:8.2f}  "
                    f"{route.routed_family:11s}"
                    f"{str(route.projection_used):>11s}  "
                    f"{request_digest(request)[:12]}"
                )
                assert route.request_unchanged
            """,
        ),
        markdown(
            r"""
            ## 15.4 Projection changes the initializer query, not the star

            Explicit C/N/O coordinates require `cno8`; there is no fallback
            that may silently erase those abundances. The cool request lies
            below the learned model's temperature support, so it cannot be
            evaluated as an ordinary in-domain prediction.

            The safe response is a projection used only to query the
            initializer. A deterministic nearby point supplies a numerically
            plausible seed, but the original labels remain untouched. The
            exact solver still sees the requested temperature, gravity, and
            mixture. This distinction prevents a common mistake: changing the
            scientific problem merely to make a machine-learning model
            comfortable.

            When more than one nearby candidate is tried, they are ordered
            alternatives. They are not averaged atmospheres, posterior
            samples, or new stellar labels.
            """
        ),
        code(
            """
            cool = requests[3]
            cool_route = routes[cool.name]
            print(cool.canonical_payload())
            print("request SHA-256:", request_digest(cool))
            print("routed family:", cool_route.routed_family)
            print("initializer projection:", cool_route.projection_used)
            print("first initializer query:", cool_route.first_initializer_query)
            print("requested labels unchanged:", cool_route.request_unchanged)
            for message in cool_route.warning_messages:
                print("projection record:", message)
            """,
        ),
        markdown(
            r"""
            ## 15.5 Direct abundance preserves the realized mixture

            The experimental direct-abundance route remains separate from
            these four ordinary/CNO requests. Its boundary accepts 81 public
            `[X/H]` values, quantizes them on the 0.01-dex lattice, and forms
            the 97-slot physical mixture, with 16 no-solar-reference slots
            inheriting iron.

            The distinction between the public and solver vectors is not
            cosmetic. The public vector describes elements for which the
            caller supplies meaningful abundances. The solver vector includes
            additional species slots needed by the equation of state. Once the
            mapping is completed, a mixture hash records the *realized*
            97-value vector—not merely the shorter request.

            The capstone therefore keeps two governing rules from Chapter 14:
            the realized mixture must be preserved exactly, and an exact
            closure trial is mandatory. A failed release gate cannot be
            bypassed because the initializer output looks plausible.
            """,
        ),
        markdown(
            r"""
            ## 15.6 Execute the public exploratory workflow

            We now make one real, compact public call for the solar-like
            request: labels enter the learned initializer, the atmosphere is
            bridged to the synthesis state, and a narrow spectrum is formed.
            This is not a mock call and not a precomputed curve. It exercises
            the same public label interface that a user would call.

            The returned safety fields remain `False/True` because useful
            exploratory output is not physical closure. Notice that the
            spectrum can be finite and line-rich while those flags remain
            unchanged. That is exactly the point: spectral plausibility and
            atmospheric convergence are different propositions.
            """
        ),
        code(
            """
            safety = safety_type_checkpoint()
            print("public workflow executed:", not safety.probe_only)
            print("initializer family:", safety.initializer_family)
            print("InitializedAtmosphere flags:", safety.initialized_flags)
            print("LabelSpectrum flags:", safety.label_spectrum_flags)
            print("wavelength samples:", safety.wavelength_count)
            print("minimum normalized flux:",
                  safety.minimum_normalized_flux)
            assert safety.initialized_flags == (False, True)
            assert safety.label_spectrum_flags == (False, True)
            """,
        ),
        markdown(
            r"""
            ## 15.7 Saving the initialized atmosphere preserves its warning

            `InitializedAtmosphere.save_npz` creates a reusable, validated
            schema-v4 file whose metadata role is
            `learned_initializer_prediction`. Saving does not promote the
            state. The convergence flags survive the round trip, and the
            loaded physical arrays are checked against the arrays in memory.

            Product metadata is optional because an older converged physical
            archive can still satisfy the numerical schema. When metadata is
            present, however, it must be complete and internally consistent.
            This gives us backward compatibility without allowing partial
            provenance to make an unverified prediction appear converged.
            """
        ),
        code(
            """
            print("validated public arrays:", len(safety.saved_validated_names))
            print("loaded mapping fields:", safety.loaded_field_count)
            print("product role:", safety.saved_product_role)
            print("initializer family:", safety.saved_initializer_family)
            print("metadata converged:", safety.metadata_converged)
            print("metadata closure required:", safety.metadata_closure_required)
            print("eligible as verified physical product:",
                  safety.eligible_as_verified_physical_product)
            print("physical arrays preserved:", safety.mapping_arrays_equal)
            """,
        ),
        markdown(
            r"""
            ## 15.8 Schema v4 is the Chapter 10 handoff

            The atmosphere solver and the spectral synthesizer do not share
            internal Python objects. They communicate through a deliberately
            narrow numerical contract. The compact four-regime mappings have
            25 canonical public arrays.
            Depth-dependent arrays share the leading depth axis; elemental
            abundances and continuum-edge tables have their own declared axes.
            All public numerical arrays are float64.

            A valid schema proves that the two programs agree about names,
            shapes, axes, and units. It does not prove that the values solve
            hydrostatic equilibrium or radiative equilibrium. The supplied
            six-depth states below are integration fixtures, not converged
            Chapter 13 products. Their role is to test the handoff and
            synthesis over four regimes before we inspect a full exact solve.
            """
        ),
        code(
            """
            capstone = four_regime_capstone()
            for row in capstone.rows:
                atmosphere = row.atmosphere
                print(
                    row.name,
                    "fields=", atmosphere.required_field_count,
                    "depth=", atmosphere.depth_count,
                    "missing=", atmosphere.missing_fields,
                    "depth axes=", atmosphere.all_depth_axes_match,
                    "float64=", atmosphere.all_public_arrays_float64,
                    "m increasing=", atmosphere.column_mass_strictly_increasing,
                )
            print("compact fixture only:", capstone.compact_fixture_only)
            """,
        ),
        markdown(
            r"""
            ## 15.9 Synthesis preserves the public flux convention

            Chapter 10 returns wavelength and total/continuum spectral flux
            densities per nanometer plus their dimensionless ratio,

            \[
            f_{\rm norm}(\lambda)
            =\frac{F_\lambda}{F_{\lambda,c}}.
            \]

            Keeping all three arrays matters. The total flux is the physical
            prediction, the continuum flux records the normalization model,
            and their ratio is the convenient observable. If only the ratio
            were retained, a change in continuum placement could be mistaken
            for a change in line opacity.

            All four states use the same wavelength grid and exact
            CPU/float64 policy. The one-panel comparison holds the numerical
            experiment fixed and changes only the atmosphere supplied to the
            synthesizer.
            """
        ),
        code(
            """
            print("wavelength samples:", capstone.wavelength_nm.size)
            print("backend / dtype:", capstone.backend, capstone.dtype)
            zoom = (
                (capstone.wavelength_nm >= 498.95)
                & (capstone.wavelength_nm <= 499.15)
            )
            fig, ax = single_panel()
            for row in capstone.rows:
                spectrum = row.spectrum
                public = public_spectrum(row.name)
                ax.plot(
                    public.wavelength_nm[zoom],
                    public.normalized_flux[zoom],
                    lw=1.6,
                    label=row.name.replace("_", " "),
                )
                assert spectrum.normalized_ratio_matches
            ax.set_xlabel("Wavelength [nm]")
            ax.set_ylabel(r"Normalized flux $F_\\lambda/F_{\\lambda,c}$")
            ax.set_title("The same spectral window in four atmospheres")
            add_quiet_grid(ax)
            ax.legend()
            plt.show()
            """,
        ),
        markdown(
            r"""
            ## 15.10 Why the four spectra differ

            That figure is the last physical result in the book, and it is
            worth reading rather than admiring. Every difference in it traces
            back to a mechanism built in an earlier chapter.

            **The Sun sets the baseline.** In the optical, the continuum is not
            dominated by hydrogen bound-free absorption but by H\(^-\), the
            negative hydrogen ion of Chapter 5. H\(^-\) needs two things at
            once: a
            neutral hydrogen atom to attach to, and a free electron to attach.
            At solar temperatures hydrogen is overwhelmingly neutral, and the
            free electrons come almost entirely from metals with low ionization
            potentials — Na, Mg, Al, Si, Fe — because hydrogen itself is far too
            tightly bound to give one up. So the solar continuum opacity scales
            roughly as (neutral H) \(\times\) (electron density), and it is the
            metals, through the Saha balance of Chapter 3, that set the second
            factor.

            **The hot dwarf breaks the first factor.** Raise \(T_{\rm eff}\)
            and hydrogen ionizes. Removing neutral hydrogen removes H\(^-\)'s
            partner, the H\(^-\) continuum collapses, and hydrogen bound-free,
            free-free, and electron scattering take over instead. At the same
            time most metals move to higher ionization stages, so the crowded
            forest of neutral-metal lines that fills the solar optical
            (Chapter 7) thins dramatically. The spectrum becomes bluer, cleaner,
            and dominated by fewer, broader features.

            **The giant breaks the second factor.** Gravity enters through
            pressure: hydrostatic balance gives \(P=gm\) (§1.10), so at equal
            column mass a low-gravity giant sits at lower pressure and lower
            density. Fewer free electrons means weaker H\(^-\) and a lower
            continuum opacity; and at low electron density the Saha balance
            shifts further toward ionization. Lower pressure also weakens the
            collisional damping wings of strong lines (Chapter 7), so the giant's
            lines are narrower than the dwarf's at the same temperature. That
            width difference is the classical luminosity-class diagnostic, and
            here it falls out of the calculation rather than being asserted.

            **The cool dwarf changes what the opacity is made of.** Drop
            \(T_{\rm eff}\) far enough and atoms bind into molecules
            (Chapter 4). TiO carries enormous band systems through the optical
            and water dominates the infrared. These are not isolated lines but
            bands of millions of transitions, and they blanket the spectrum so
            completely that the "continuum" becomes largely notional — the
            normalized flux is measured against a pseudo-continuum that no
            photon ever sees. This is the blanketing of §11.12 in its most
            extreme form, and it is also why such atmospheres are strongly
            convective: opacity that large makes radiative transport
            inefficient, and the instability criterion of §12.19 is met over a
            wide range of depths.

            Every one of those effects was built from physical principles in
            this book. None of them was fitted, and none was inserted as a
            regime-specific special case — the same algorithm produced all four
            curves.

            ## 15.11 Separate breadth from depth

            A trustworthy end-to-end test needs two complementary views.
            Breadth asks whether the same interface works for qualitatively
            different stars. The compact four-regime calculation supplies
            that view cheaply. Depth asks whether at least one case traverses
            the full atmosphere iteration and agrees with an independent
            implementation path. The solar calculation in the next section
            supplies that view.

            The table below is deliberately explicit about what ran. Its
            schema and spectrum columns refer to exact compact Chapter 10
            integration. The public exploratory label call has also run. The
            four rows themselves remain six-depth integration fixtures; none
            is promoted into a converged atmosphere merely because synthesis
            succeeds.
            """
        ),
        code(
            """
            print(
                "case                    family  proj  molecules  "
                "schema  spectrum  exploratory / verified"
            )
            for row in capstone.rows:
                schema_ok = next(
                    gate.passed for gate in row.exploratory_acceptance
                    if gate.gate == "schema-v4 mapping"
                )
                spectrum_ok = next(
                    gate.passed for gate in row.exploratory_acceptance
                    if gate.gate == "finite wavelength-aligned spectrum"
                )
                print(
                    f"{row.name:24s}{row.routed_family:8s}"
                    f"{str(row.projection_used):>6s}"
                    f"{str(row.molecular_lines):>11s}"
                    f"{str(schema_ok):>8s}{str(spectrum_ok):>10s}  "
                    f"{row.exploratory_status} / {row.verified_status}"
                )
            """,
        ),
        markdown(
            r"""
            ## 15.12 Exploratory acceptance contains deliberate blanks

            An exploratory mapping must pass asset, schema, spectrum, and
            provenance checks. Those tests establish that the initializer and
            synthesizer were called coherently and returned a well-formed
            result.

            Structural convergence, flux closure, and hydrostatic acceptance
            answer questions that this fast route never asks. They are
            represented by `None`, not by a convenient `True` and not by
            `False`. In a scientific acceptance table, “not evaluated” is
            distinct from both “passed” and “failed.”
            """
        ),
        code(
            """
            exploratory = capstone.rows[1].exploratory_acceptance
            for gate in exploratory:
                print(f"{gate.gate:42s} {str(gate.passed):5s}  {gate.evidence}")
            assert next(
                row.passed for row in exploratory
                if row.gate == "structural fixed-point convergence"
            ) is None
            """,
        ),
        markdown(
            r"""
            ## 15.13 One exact solar solve closes the implementation loop

            We now replace the compact solar fixture with evidence from a full
            80-layer atmosphere calculation. The request is
            \(T_{\rm eff}=5777\,{\rm K}\), \(\log g=4.44\),
            \([{\rm M/H}]=[\alpha/{\rm M}]=0\), and
            \(\xi=2\,{\rm km\,s^{-1}}\), with convection and molecules
            enabled.

            The learned atmosphere is only the seed. The exact loop then
            recomputes populations, opacity, transfer, flux corrections,
            pressure, density, and the remapped state. After the fourth pass,
            the declared deep-layer metric reached

            \[
            \max\left|\frac{\Delta T}{T}\right|
            =4.778548\times10^{-4}
            <5.0\times10^{-4}.
            \]

            The accepted archive contains 27 numerical arrays: the 25 required
            public atmosphere fields, the schema version, and one retained
            hydrogen-ion population helper. Microturbulence is already one of
            the 25 public fields. Its 80 layers span the photospheric structure
            needed by the synthesizer. A narrow 498.95–499.15 nm spectrum is
            then computed from that converged state.

            This evidence is stored as a compact, timing-free golden product,
            so reading and validating the chapter does not require shipping
            the multi-gigabyte source catalogs.
            """
        ),
        code(
            """
            print("accepted status:", verified_solar.status)
            print("converged on pass:", verified_solar.converged_on_pass)
            print(
                "deep |dT/T|:",
                verified_solar.deep_temperature_change,
                "<",
                verified_solar.deep_temperature_change_threshold,
            )
            print(
                "atmosphere:",
                verified_solar.atmosphere_array_count,
                "arrays on",
                verified_solar.atmosphere_depth_count,
                "layers",
            )
            print(
                "temperature range [K]:",
                verified_solar.minimum_temperature_k,
                verified_solar.maximum_temperature_k,
            )
            print(
                "column-mass range [g cm^-2]:",
                verified_solar.minimum_column_mass,
                verified_solar.maximum_column_mass,
            )
            print(
                "spectrum:",
                verified_solar.wavelength_count,
                "samples; min/max normalized flux:",
                verified_solar.minimum_normalized_flux,
                verified_solar.maximum_normalized_flux,
            )
            print()
            for gate in verified_solar.acceptance:
                print(
                    f"{gate.gate:42s} "
                    f"{str(gate.passed):5s}  {gate.evidence}"
                )
            """,
        ),
        markdown(
            r"""
            The word *accepted* still needs a scope. The exact temperature
            convergence, schema, finite spectrum, internal flux ratio, and
            three-run array parity all pass. This publication did not retain
            an independently reviewed flux-error threshold, hydrostatic
            residual threshold, or a separate standard-optical-grid
            acceptance record. Those rows therefore remain `None`.

            This mixed table is stronger than an all-green dashboard: it tells
            us exactly which statement may be reproduced from the retained
            evidence.
            """
        ),
        markdown(
            r"""
            ## 15.14 Gates fail independently

            Acceptance is not a single boolean because the coupled workflow
            can be wrong in more than one way. A broken column-mass order is a
            schema and physical-boundary failure even if every value remains
            finite. It says that deeper layers no longer follow shallower
            layers along the intended coordinate.

            Conversely, a perturbed normalized spectrum can fail
            \(f_{\rm norm}=F_\lambda/F_{\lambda,c}\) without changing a single
            atmosphere array. The short experiment below introduces those two
            failures separately. A useful test suite should identify the
            damaged boundary instead of reducing both cases to “the model
            failed.”
            """
        ),
        code(
            """
            from book.chapter10_runtime import load_regime_atmosphere, public_spectrum
            from payne_zero_synthesis.atmosphere import REQUIRED_ATMOSPHERE_ARRAYS

            broken_atmosphere = load_regime_atmosphere("solar_dwarf")
            broken_atmosphere["column_mass"] = (
                broken_atmosphere["column_mass"][::-1].copy()
            )
            atmosphere_failure = summarize_atmosphere(
                broken_atmosphere,
                required_fields=REQUIRED_ATMOSPHERE_ARRAYS,
            )
            reference_spectrum = public_spectrum("solar_dwarf")
            broken_normalized = reference_spectrum.normalized_flux.copy()
            broken_normalized[0] += 0.01
            spectral_failure = summarize_spectrum(
                wavelength_nm=reference_spectrum.wavelength_nm,
                flux_total=reference_spectrum.flux_total,
                flux_continuum=reference_spectrum.flux_continuum,
                normalized_flux=broken_normalized,
                ratio_rtol=3.0e-7,
                ratio_atol=0.0,
            )
            print("column-mass gate:", atmosphere_failure.column_mass_strictly_increasing)
            print("spectral ratio gate:", spectral_failure.normalized_ratio_matches)
            """,
        ),
        markdown(
            r"""
            ## 15.15 Reproducibility is an identity, not a runtime anecdote

            Reproducibility means more than running the same notebook twice.
            The exact solar result was produced along three routes:

            1. the staged implementation in this textbook,
            2. a fresh repeat of that staged implementation, and
            3. the pinned, read-only source implementation.

            All three atmosphere archives are byte-identical. The wavelength,
            total flux, continuum flux, and normalized flux arrays are also
            bitwise identical. Only the measured synthesis time changes, as it
            should: timing depends on caches and machine state, while physical
            arrays define the calculation.

            The book-wide canonical digest complements that direct comparison.
            It binds requests, source, schema, assets, compact catalogs,
            backend, dtype, thread declaration, cache policy, and availability.
            Wall-clock fields are reported but excluded from the identity.
            """
        ),
        code(
            """
            reproducibility = reproducibility_checkpoint()
            print("exact comparison roles:", verified_solar.run_roles)
            print(
                "atmosphere archives byte-identical:",
                verified_solar.atmosphere_archives_byte_identical,
            )
            print(
                "physical spectrum arrays bitwise-identical:",
                verified_solar.spectrum_physical_arrays_bitwise_identical,
            )
            print("atmosphere archive SHA-256:",
                  verified_solar.atmosphere_archive_sha256)
            print("spectrum payload SHA-256:",
                  verified_solar.spectrum_payload_sha256)
            print()
            print("provenance SHA-256:", reproducibility.sha256)
            print("repeatable:", reproducibility.repeatable)
            print("timing fields excluded:",
                  reproducibility.timing_fields_excluded)
            print("runtime:", reproducibility.provenance["runtime"])
            print("availability:", reproducibility.provenance["availability"])
            """,
        ),
        markdown(
            r"""
            ## 15.16 Time the boundaries that actually ran

            The executed exploratory call reports initializer, population
            bridge, and synthesis time independently. These quantities answer
            a practical profiling question: where did this particular call
            spend its time?

            The exact solar solve did run, but its accepted identity records
            the four atmosphere passes rather than a “golden” wall time.
            Cold and warm runtimes depend on compilation, caches, thread
            scheduling, and hardware. Freezing one observed duration would
            turn a performance anecdote into a false scientific invariant.
            """
        ),
        code(
            """
            print("initializer [s]:", safety.initializer_seconds)
            print("population bridge [s]:", safety.population_bridge_seconds)
            print("synthesis [s]:", safety.synthesis_seconds)
            print(
                "exact solar atmosphere passes:",
                verified_solar.converged_on_pass,
            )
            print("exact atmosphere wall time: excluded from product identity")
            print("compact synthesis evidence:")
            for row in capstone.rows:
                print(
                    f"  {row.name:24s}"
                    f"{row.spectrum.wavelength_count:6d} wavelength samples"
                )
            print("timings excluded from identity:",
                  reproducibility.timing_fields_excluded)
            """,
        ),
        markdown(
            r"""
            ## 15.17 Model card and unrun evidence

            The intended physical model is LTE, one-dimensional, static, and
            uses local mixing-length convection. Fixed-thread atmosphere runs
            are the strict reproducibility target; alternate thread counts may
            move last bits. These assumptions define the class of stars and
            phenomena the calculation can represent; parity with an
            implementation cannot make the assumptions disappear.

            The compact reader does not bundle the approximately 6.8 GB full
            source catalogs. They are required to rebuild the exact solar
            product, but not to validate or study the published numerical
            arrays. This is a data-distribution choice, not a fallback to
            approximate physics: the acceptance record identifies the catalog
            policy and the hashes of the products made with the full data.

            Only the solar request has completed a retained full exact solve in
            this capstone. The hot dwarf, giant, and cool molecule-rich
            requests still provide compact end-to-end synthesis coverage. They
            must not be described as converged atmosphere products. Cool
            molecule-rich cases are especially fragile, and catalog coverage
            plus the separate water-line policy remain explicit limitations.
            """
        ),
        code(
            """
            print("model limitations:")
            for limitation in reproducibility.provenance["limitations"]:
                print(" -", limitation)
            print()
            print("missing initializer exports:",
                  dependencies.missing_initializer_exports)
            print("missing runner symbols:",
                  dependencies.missing_runner_symbols)
            print("public solve staged:",
                  dependencies.solve_structured_atmosphere_staged)
            print("full exact catalogs ready:",
                  dependencies.exact_source_catalogs_ready)
            print("catalog blocker:",
                  dependencies.exact_source_catalog_blocker)
            print(
                "catalogs required to rebuild accepted solar product:",
                verified_solar.catalogs_required_to_rebuild,
            )
            print(
                "catalogs required to validate accepted solar product:",
                verified_solar.catalogs_required_to_validate,
            )
            """,
        ),
        markdown(
            r"""
            ## 15.18 Chapter summary

            1. Exploratory and verified workflows have different exact return
               types, safety flags, and acceptance requirements.
            2. Initializer projection changes only a query. Requested labels
               and mixture remain the physical target.
            3. Schema v4 is the only atmosphere-to-synthesis handoff, with 25
               canonical arrays and optional product metadata.
            4. `Spectrum` retains wavelength, total flux, continuum flux,
               normalized flux, and timing; the ratio is checked explicitly.
            5. Structural, flux, hydrostatic, optical, schema, and spectral
               gates are separate claims and can fail independently.
            6. The exact solar atmosphere converges on pass four and crosses
               the schema-v4 boundary into an eight-sample narrow spectrum.
            7. Staged, repeated-staged, and pinned-source calculations produce
               byte-identical atmosphere archives and bitwise-identical
               physical spectrum arrays. Wall-clock timings are excluded.
            8. That solar evidence passes structural convergence, schema,
               finite-spectrum, ratio, parity, and provenance gates. Flux
               closure, hydrostatic residual, and a separately retained
               optical-grid acceptance remain unevaluated.
            9. The compact four-regime spectra provide breadth across hot,
               solar, giant, and cool molecular conditions. Only the solar
               case has a retained full exact solve in this chapter.
            10. Full catalogs are needed to rebuild the solar golden product,
                but not to validate or inspect it.

            The final honest choice is:

            `labels → InitializedAtmosphere → LabelSpectrum` for exploratory
            work, with closure still required; or

            `labels → exact solve → converged schema-v4 Path → Spectrum +
            independent acceptance rows` for a physical claim. The claim is
            only as broad as the rows that actually passed.

            The course ends where trustworthy modeling begins: not with one
            attractive curve, but with a typed result whose assumptions,
            provenance, and unpassed gates remain visible.
            """
        ),
    ]
    return notebook(cells)


if __name__ == "__main__":
    build_notebook()
