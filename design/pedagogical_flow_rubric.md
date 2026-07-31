# Pedagogical Flow Rubric

This rubric turns the requested teaching standard into a repeatable chapter gate. The reference
experience is Lecture 1 of `https://agent4astro.com/`, inspected on 2026-07-29. Its subject matter
is not copied. What matters is the causal architecture of the explanation.

## The Reference Pattern

The reference lecture does not begin with internal machinery. It begins with a familiar experience,
identifies why that experience is misleading, and states one central claim. It then says, in effect:
the claim is easy to repeat but hard to believe, so we will build the mechanism and watch it work.

Its strongest recurring moves are:

1. **Experience before abstraction.** A concrete sentence fragment creates the need for a
   next-token distribution before “tokenization” is defined.
2. **One durable spine.** The central claim returns after training, sampling, context, memory, and
   hallucination; new sections elaborate it rather than start unrelated stories.
3. **Just-in-time machinery.** Data, tensors, configuration, and gradients appear at the first
   moment the build requires them.
4. **Code as evidence.** A cell tests the immediately preceding claim, and the next paragraph reads
   the actual output rather than merely announcing that the code ran.
5. **Explicit mathematical altitude.** A difficult derivation says what intuition must be retained
   and what detail may be revisited later.
6. **Production complexity after the transparent core.** The small model is built before the
   lecture explains what a production assistant adds.
7. **Limitations become the course map.** Finite context, statelessness, and ungrounded generation
   each open a specific later lecture.
8. **Earned synthesis.** The close restates a mental model the reader has demonstrated, maps its
   open boundaries forward, summarizes capabilities, and provides a direct next navigation link.

## Required Chapter Arc

Every chapter must pass through these acts, even if the headings differ.

### Act 1 — A question with stakes

- Begin with an observable, failure, or physical tension a reader can picture.
- State the chapter's central question in ordinary language.
- Explain why the previous chapter cannot answer it.
- State one compact claim the reader will earn by computation.

### Act 2 — The smallest physical picture

- Introduce only the objects needed for the question.
- Use a conceptual schematic when geometry, flow, coupling, or branching is hard to hold in prose.
- Define each new term beside a concrete example or limiting case.
- State assumptions before they silently enter an equation.

### Act 3 — Mathematics because intuition needs precision

- Motivate every equation by naming the ambiguity it resolves.
- Define every symbol, sign, unit, coordinate direction, and limiting behavior.
- Derive important results in short causal steps.
- Tell the reader what the equation predicts before evaluating it.

### Act 4 — A small executable test

- Show one conceptual operation per visible code cell.
- State reads, writes, shape, units, dtype, and device before the code depends on them.
- Predict the output or invariant before execution.
- Interpret the printed number or plot immediately afterward.
- Use an analytic limit, conservation law, monotonic trend, round trip, or parity oracle to decide
  whether the result is trustworthy.

### Act 5 — Production convergence

- Introduce the exact Payne Zero function, field, schema, or branch only after its physical role is
  clear.
- If the production routine is short, execute the exact source.
- If it is long, teach exact stages without changing order or inventing a parallel API.
- Name any component loaded from later in the book as an integration fixture and state why it is
  temporarily needed.
- End the stage with a source or numerical parity check.

### Act 6 — An honest boundary

- State what the reader has built in exact output terms.
- State what the object cannot yet claim.
- Do not wrap an incomplete calculation in a production-looking class.
- Convert each important missing dependency into a reason for a later section or chapter.

### Act 7 — Close the loop

- Return to the opening question and answer the part now earned.
- Give a concise summary containing no new concepts.
- Embed worthwhile variations of an assumption, limit, or implementation choice in the main text,
  where the reader can predict and interpret them immediately.
- End with a direct link whose prose explains why the next chapter is necessary.

## Paragraph-Level Gate

For every prose paragraph, an editor must be able to answer at least one of:

- What question from the previous paragraph does this answer?
- What new question does this create?
- What equation, code cell, plot, or implementation boundary does this prepare?
- What observed result does this interpret?
- What assumption or limitation does this make honest?

If none applies, the paragraph is probably decorative, redundant, premature, or misplaced.

## Code-and-Output Gate

For every visible code cell:

1. the preceding prose states its purpose and contract;
2. the cell is bite-sized and uses canonical code;
3. the output is visible unless silence itself is the check;
4. the following prose interprets the actual output;
5. the result changes what the reader knows or what the next step requires.

Two visible code cells may not touch without a prose bridge.
The chapter has no detached exercise set; useful questions are resolved inside the teaching flow.

## Whole-Chapter Audit

A chapter is not accepted until all answers are “yes.”

- Does one central question organize the chapter?
- Does every section inherit an unresolved need from the previous section?
- Are new terms introduced just in time?
- Are exact production names delayed until their role is understood?
- Is Payne Zero referenced where it adds implementation information rather than as verbal branding?
- Can the notebook run without the external Payne Zero or paper checkout?
- Are all loaded data roles explicit and provenance-bound?
- Is every numerical plot one claim, professionally rendered, and interpreted in the next prose?
- Is every conceptual schematic original to the textbook and scientifically audited?
- Does the ending distinguish “computed” from “not yet computed”?
- Does the summary introduce no new material?
- Does the next-chapter link identify a genuine dependency rather than merely announce a title?
- Can a final-year undergraduate follow the causal chain without outside reading?

## Neighbor and Whole-Book Audit

After each chapter wave:

- read the final two sections of chapter \(n-1\), all of chapter \(n\), and the opening two sections
  of chapter \(n+1\) as one continuous argument;
- remove duplicated definitions and repeated motivation;
- verify that every forward promise is fulfilled exactly once;
- verify notation, units, array axes, names, and assumptions at the handoff;
- confirm that the 15 chapter summaries form a faithful compressed version of the entire book.
