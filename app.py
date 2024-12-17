import streamlit as st
from openai import OpenAI

# Replace 'key' with your actual OpenAI API key.
client = OpenAI(api_key=st.secrets['API_KEY'])

########################################
# HELPER FUNCTIONS
########################################

def run_completion(messages, model, verbose=False):
    if verbose:
        st.write("**Requesting Model:**", model)
        st.write("**Messages:**", messages)
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    response = completion.choices[0].message.content
    if verbose:
        st.write("**Response:**", response)
    return response

########################################
# PROMPT TEMPLATES
########################################

SUMMARIZER_SYSTEM = """You are a specialized research assistant focused on analyzing scholarly research text and identifying AI alignment implications.
You must provide a *comprehensive, systematic overview* of the given research text, including:
- The problem domain, key hypotheses, and methods described.
- The core findings or theoretical contributions, including any experimental details if provided.
- Relevance to AI alignment: potential links to alignment problems, interpretability, safety, robustness, or analogous scenarios in the paper’s domain.
- Highlight specific angles or considerations if additional user-provided contexts are given.

Your output should be factual, structured, and thorough. If you speculate, label it as speculation. Avoid vague language. Make this overview as complete and explicit as possible, given the input text.
"""

SUMMARIZER_USER = """Below is a research text. Provide a comprehensive, systematic overview of its contents and highlight potential AI alignment implications.

Research Text:
{user_text}

Additional Angles to Consider (if any):
{additional_context}
"""

IDEATOR_SYSTEM = """You are an AI alignment research ideation agent.
Your goals:
1. Given the systematic overview, propose multiple innovative, forward-looking AI alignment research directions derived from the paper’s content.
2. Emphasize novelty, scientific feasibility, and impact on alignment (e.g. interpretability, corrigibility, safe exploration).
3. Incorporate any additional user-provided angles into your ideation.
4. When asked to refine, you will produce a critique and then improved versions of the ideas in a structured, parseable format.

IMPORTANT: When refining, output in a predictable format:

CRITIQUE_START [Your critique here] CRITIQUE_END

IMPROVED_START [Your improved ideas here, stand-alone and well-structured] IMPROVED_END


If ideas seem repetitive or stuck, try introducing a fresh perspective or analogy.
"""

IDEATOR_IDEATION_USER = """Below is the systematic overview of the research text. Using this overview and any additional angles, propose several original AI alignment research directions.

Systematic Overview:
{overview}

Additional Angles:
{additional_context}
"""

IDEATOR_REFINE_USER = """Below are the current alignment ideas. First, critique them clearly. Then rewrite improved ideas in the specified parseable format.

Current Ideas:
{ideas}

Remember:
- Use the CRITIQUE_START/END and IMPROVED_START/END markers.
- Make improved ideas standalone, clear, and grounded. If repetitive, introduce a fresh angle.
"""

TERMINATION_SYSTEM = """You are an evaluator of alignment research directions.
1. Assess if the final improved set of ideas is novel, clear, and feasible.
2. If good, respond with "Good enough".
3. If still vague or repetitive, respond with "Needs more iteration".
4. Respond with one of these strings only, no explanation.
"""

TERMINATION_USER = """Evaluate these improved alignment ideas:

{refined_ideas}

Are they "Good enough" or "Needs more iteration"?
"""

DIFFERENT_ANGLE_SYSTEM = """You are a resourceful alignment ideation assistant who must break out of previous patterns.
If prompted, produce a fundamentally different set of approaches that avoid previous repetitive angles, possibly drawing analogies from unrelated disciplines, new theoretical lenses, or untested frameworks.
"""

DIFFERENT_ANGLE_USER = """Current ideas seem repetitive. Provide a fundamentally different set of alignment directions. Aim for genuine novelty and break from earlier patterns.

Previous Refined Ideas:
{refined_ideas}
"""

########################################
# WORKFLOW FUNCTIONS
########################################

def summarize_text(user_text, additional_context, model, verbose):
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        {"role": "user", "content": SUMMARIZER_USER.format(user_text=user_text, additional_context=additional_context)}
    ]
    return run_completion(messages, model, verbose=verbose)

def generate_ideas(overview, additional_context, model, verbose):
    messages = [
        {"role": "system", "content": IDEATOR_SYSTEM},
        {"role": "user", "content": IDEATOR_IDEATION_USER.format(overview=overview, additional_context=additional_context)}
    ]
    return run_completion(messages, model, verbose=verbose)

def refine_ideas(ideas, model, verbose):
    messages = [
        {"role": "system", "content": IDEATOR_SYSTEM},
        {"role": "user", "content": IDEATOR_REFINE_USER.format(ideas=ideas)}
    ]
    return run_completion(messages, model, verbose=verbose)

def check_termination(refined_ideas, model, verbose):
    messages = [
        {"role": "system", "content": TERMINATION_SYSTEM},
        {"role": "user", "content": TERMINATION_USER.format(refined_ideas=refined_ideas)}
    ]
    return run_completion(messages, model, verbose=verbose)

def try_different_angle(refined_ideas, model, verbose):
    messages = [
        {"role": "system", "content": DIFFERENT_ANGLE_SYSTEM},
        {"role": "user", "content": DIFFERENT_ANGLE_USER.format(refined_ideas=refined_ideas)}
    ]
    return run_completion(messages, model, verbose=verbose)

########################################
# STREAMLIT UI
########################################

st.title("Alignment Research Automator (Enhanced MVP)")

model_choice = st.selectbox("Select LLM Model:", ["gpt-4o", "o1-mini", "o1-preview"])
user_text = st.text_area("Paste your research text (abstract or section):", height=300)
additional_context = st.text_area("Optional additional angles or considerations:", height=100)
verbose = st.checkbox("Show verbose debug info")

max_iterations = 3  # max refinement iterations before trying different angle

if st.button("Run Pipeline"):
    if not user_text.strip():
        st.warning("Please provide some text before running the pipeline.")
    else:
        # Step 1: Summarize/Overview
        overview = summarize_text(user_text, additional_context, model_choice, verbose)
        st.subheader("Comprehensive Systematic Overview")
        st.write(overview)

        # Step 2: Initial Ideas
        ideas = generate_ideas(overview, additional_context, model_choice, verbose)
        st.subheader("Initial Alignment Ideas")
        st.write(ideas)

        refined_ideas = ideas
        iteration_count = 0

        # Step 3: Refinement loop with termination checks
        while iteration_count < max_iterations:
            # Refine ideas
            refinement_output = refine_ideas(refined_ideas, model_choice, verbose)

            # Parse refinement output:
            # Expecting:
            # CRITIQUE_START ... CRITIQUE_END
            # IMPROVED_START ... IMPROVED_END
            critique_start = refinement_output.find("CRITIQUE_START")
            critique_end = refinement_output.find("CRITIQUE_END")
            improved_start = refinement_output.find("IMPROVED_START")
            improved_end = refinement_output.find("IMPROVED_END")

            if critique_start == -1 or critique_end == -1 or improved_start == -1 or improved_end == -1:
                # If parsing fails, just treat everything as improved ideas
                # (fallback scenario)
                final_refined_ideas = refinement_output.strip()
                critique_section = "Could not parse critique."
            else:
                critique_section = refinement_output[critique_start+len("CRITIQUE_START"):critique_end].strip()
                final_refined_ideas = refinement_output[improved_start+len("IMPROVED_START"):improved_end].strip()

            st.subheader(f"Iteration {iteration_count+1} Critique")
            st.write(critique_section)
            st.subheader(f"Iteration {iteration_count+1} Refined Ideas")
            st.write(final_refined_ideas)

            # Check termination
            verdict = check_termination(final_refined_ideas, model_choice, verbose)
            st.write("Termination Check:", verdict)

            if "Good enough" in verdict:
                st.success("The ideas are considered good enough!")
                break
            else:
                # Needs more iteration
                iteration_count += 1
                refined_ideas = final_refined_ideas

        # If max iterations reached and still not good enough, try different angle
        if iteration_count == max_iterations and "Needs more iteration" in verdict:
            st.warning("Ideas still not good enough after multiple refinements. Attempting a different angle...")
            diff_ideas = try_different_angle(refined_ideas, model_choice, verbose)
            st.subheader("Different Angle Ideas")
            st.write(diff_ideas)

            # Final termination check after different angle attempt
            diff_verdict = check_termination(diff_ideas, model_choice, verbose)
            st.write("Final Termination Check After Different Angle:", diff_verdict)
            if "Good enough" in diff_verdict:
                st.success("The different angle produced good enough ideas!")
            else:
                st.info("Even after a different angle, more iteration might be needed. Consider revising input or approach.")
