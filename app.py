import streamlit as st
from openai import OpenAI
import re

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
- The domain and key hypotheses.
- Methods, results, theoretical contributions, or experimental details.
- Potential relevance to AI alignment: interpretability, safety, robustness, or related concepts.
- Integrate any additional user-specified angles.

Your output should be factual, structured, and thorough. If you speculate, label it as speculation. Avoid vague language.
"""

SUMMARIZER_USER = """Below is a research text. Provide a comprehensive, systematic overview of its contents and highlight potential AI alignment implications.

Research Text:
{user_text}

Additional Angles to Consider (if any):
{additional_context}
"""

IDEATOR_SYSTEM = """You are an AI alignment research ideation agent.
Your tasks:
1. Given a systematic overview of a research paper, propose several innovative, forward-looking alignment research directions.
2. Consider user-provided additional angles.
3. When refining:
   - First produce a detailed critique of the current ideas. Identify which ideas are weak, repetitive, or less promising.
   - Then produce an improved set of ideas, potentially discarding weaker ones and expanding upon the most promising subset. You do not need to preserve all ideas if some are weak.
   - Make these improved ideas clearly stand-alone and more feasible. Incorporate suggestions from previous feedback.
   - If stuck in a repetitive pattern, introduce a fresh angle from a different domain.

IMPORTANT FORMAT WHEN REFINING:

CRITIQUE_START [Critique of previous ideas, including which ones to drop or improve, and why] CRITIQUE_END

IMPROVED_START [Improved standalone ideas, focusing on the best subset, more detailed, feasible, and novel] IMPROVED_END

This formatting is mandatory.
"""

IDEATOR_IDEATION_USER = """Below is the systematic overview of the research text. Using this overview and any additional angles, propose several original AI alignment research directions.

Systematic Overview:
{overview}

Additional Angles:
{additional_context}
"""

IDEATOR_REFINE_USER = """Below are the current alignment ideas. 
Refine them by:
1. Critiquing them and indicating which ones are weaker or repetitive.
2. Improve by focusing on the best subset, expanding and enhancing them without necessarily keeping all.

Use the required format:

CRITIQUE_START [Your critique here] CRITIQUE_END

IMPROVED_START [Your improved ideas here, well-structured and standalone] IMPROVED_END


Current Ideas:
{ideas}

Also consider this feedback from previous termination check if provided:
{termination_feedback}
"""

TERMINATION_SYSTEM = """You are an evaluator of alignment research directions.
Output exactly one line:
- If final improved ideas are good enough, say: "Good enough"
- If not, say: "Needs more iteration: <reason>", where <reason> is a short explanation for what is lacking (e.g. novelty, clarity, feasibility).
No other text.
"""

TERMINATION_USER = """Evaluate these improved alignment ideas:

{refined_ideas}

Are they "Good enough" or "Needs more iteration"? If needs iteration, provide a reason after a colon.
"""

DIFFERENT_ANGLE_SYSTEM = """You are a resourceful alignment ideation assistant who must break out of previous patterns.
If prompted, produce a fundamentally different set of approaches that avoid previous repetitive angles, drawing analogies from unrelated fields, new frameworks, or novel directions.
"""

DIFFERENT_ANGLE_USER = """Current ideas seem repetitive. Provide a fundamentally different set of alignment directions to break free from previous patterns.

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

def refine_ideas(ideas, termination_feedback, model, verbose):
    messages = [
        {"role": "system", "content": IDEATOR_SYSTEM},
        {"role": "user", "content": IDEATOR_REFINE_USER.format(ideas=ideas, termination_feedback=termination_feedback)}
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

st.title("Alignment Research Automator (Enhanced)")

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
        with st.expander("Comprehensive Systematic Overview (closed by default)", expanded=False):
            st.write(overview)

        # Step 2: Initial Ideas
        ideas = generate_ideas(overview, additional_context, model_choice, verbose)
        with st.expander("Initial Alignment Ideas (closed by default)", expanded=False):
            st.write(ideas)

        refined_ideas = ideas
        iteration_count = 0
        termination_feedback = ""

        # Step 3: Refinement loop with termination checks
        final_good_enough = False
        while iteration_count < max_iterations:
            refinement_output = refine_ideas(refined_ideas, termination_feedback, model_choice, verbose)

            # Parse refinement output
            critique_pattern = r"CRITIQUE_START\s*(.*?)\s*CRITIQUE_END"
            improved_pattern = r"IMPROVED_START\s*(.*?)\s*IMPROVED_END"

            critique_match = re.search(critique_pattern, refinement_output, re.DOTALL)
            improved_match = re.search(improved_pattern, refinement_output, re.DOTALL)

            if critique_match and improved_match:
                critique_section = critique_match.group(1).strip()
                final_refined_ideas = improved_match.group(1).strip()
            else:
                # Parsing failed, fallback
                critique_section = "Could not parse critique."
                final_refined_ideas = refinement_output.strip()

            iteration_label = f"Iteration {iteration_count+1}"
            with st.expander(f"{iteration_label} Critique (closed by default)", expanded=False):
                st.write(critique_section)
            with st.expander(f"{iteration_label} Refined Ideas (closed by default)", expanded=False):
                st.write(final_refined_ideas)

            verdict = check_termination(final_refined_ideas, model_choice, verbose)
            st.write("Termination Check:", verdict)

            if verdict.startswith("Good enough"):
                final_good_enough = True
                break
            else:
                # Extract reason if "Needs more iteration"
                reason_match = re.match(r"Needs more iteration:\s*(.*)", verdict)
                if reason_match:
                    termination_feedback = "Previous termination feedback: " + reason_match.group(1).strip()
                else:
                    termination_feedback = "No specific feedback provided."
                iteration_count += 1
                refined_ideas = final_refined_ideas

        if not final_good_enough and iteration_count == max_iterations:
            st.warning("Ideas still not good enough after multiple refinements. Attempting a different angle...")
            diff_ideas = try_different_angle(refined_ideas, model_choice, verbose)
            with st.expander("Different Angle Ideas (closed by default)", expanded=False):
                st.write(diff_ideas)
            diff_verdict = check_termination(diff_ideas, model_choice, verbose)
            st.write("Final Termination Check After Different Angle:", diff_verdict)
            if diff_verdict.startswith("Good enough"):
                final_good_enough = True
                final_refined_ideas = diff_ideas
            else:
                st.info("Even after a different angle, more iteration might be needed. Consider revising input or approach.")

        # Display final good enough ideas openly if achieved
        if final_good_enough:
            with st.expander("Final Accepted Ideas (open by default)", expanded=True):
                st.write(final_refined_ideas)
