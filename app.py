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
You must provide a comprehensive, systematic overview of the given research text, including:
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
   - Produce a detailed critique of the current ideas, identifying which are weak or repetitive.
   - Then produce an improved set of ideas, potentially discarding weaker ones and expanding upon the best subset.
   - Make improved ideas stand-alone, more feasible, and incorporate previous feedback.
   - If stuck in a repetitive pattern, introduce a fresh angle from another domain.

IMPORTANT FORMAT WHEN REFINING:
Use `CRITIQUE_START` and `CRITIQUE_END` to delimit the critique.
Use `IMPROVED_START` and `IMPROVED_END` to delimit the improved ideas.
"""

IDEATOR_IDEATION_USER = """Below is the systematic overview of the research text. Using this overview and any additional angles, propose several original AI alignment research directions.

Systematic Overview:
{overview}

Additional Angles:
{additional_context}
"""

IDEATOR_REFINE_USER = """Below are the current alignment ideas. 
Refine them by:
1. Critiquing them and indicating which are weaker or repetitive.
2. Improve by focusing on the best subset, enhancing and expanding them without necessarily keeping all.

Remember to use the specified format:
- `CRITIQUE_START`/`CRITIQUE_END` for critique
- `IMPROVED_START`/`IMPROVED_END` for improved ideas

Current Ideas:
{ideas}

Additional Termination Feedback:
{termination_feedback}
"""

TERMINATION_SYSTEM = """You are an evaluator of alignment research directions.
Output exactly one line:
- If final improved ideas are good enough, say: "Good enough"
- If not, say: "Needs more iteration: <reason>", where <reason> is a short explanation.
No other text.
"""

TERMINATION_USER = """Evaluate these improved alignment ideas:

{refined_ideas}

Are they "Good enough" or "Needs more iteration"? If needs iteration, follow the format: "Needs more iteration: <reason>"
"""

DIFFERENT_ANGLE_SYSTEM = """You are a resourceful alignment ideation assistant who must break out of previous patterns.
If prompted, produce a fundamentally different set of approaches that avoid previous repetitive angles, using fresh analogies, frameworks, or directions.
"""

DIFFERENT_ANGLE_USER = """Current ideas seem repetitive. Provide a fundamentally different set of alignment directions to break free from previous patterns.

Previous Refined Ideas:
{refined_ideas}
"""

ACCOMMODATE_FEEDBACK_SYSTEM = """You are an AI alignment ideation agent.
You have received improved ideas and a critique from the termination checker that says they are not good enough. Your role:
1. Review the provided ideas and the termination checker's critique.
2. Reason about how to accommodate the critique to improve the ideas further. (This reasoning should appear inside `CRITIQUE_START` and `CRITIQUE_END`.)
3. After that reasoning, produce improved ideas that address the critique, focusing on the best subset and adding new details or angles if needed. (This should appear inside `IMPROVED_START` and `IMPROVED_END`.)

Use the same formatting rules for critique and improved ideas.
"""

ACCOMMODATE_FEEDBACK_USER = """Here are the current ideas and the termination checker's critique.

Current Ideas:
{refined_ideas}

Termination Checker Critique:
{termination_reason}

Please first reason about how to accommodate this critique (inside CRITIQUE_START/END).
Then produce improved ideas addressing it (inside IMPROVED_START/END).
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

def accommodate_feedback(refined_ideas, termination_reason, model, verbose):
    messages = [
        {"role": "system", "content": ACCOMMODATE_FEEDBACK_SYSTEM},
        {"role": "user", "content": ACCOMMODATE_FEEDBACK_USER.format(refined_ideas=refined_ideas, termination_reason=termination_reason)}
    ]
    return run_completion(messages, model, verbose=verbose)

def parse_refinement_output(refinement_output):
    critique_pattern = r"CRITIQUE_START\s*(.*?)\s*CRITIQUE_END"
    improved_pattern = r"IMPROVED_START\s*(.*?)\s*IMPROVED_END"

    critique_match = re.search(critique_pattern, refinement_output, re.DOTALL)
    improved_match = re.search(improved_pattern, refinement_output, re.DOTALL)

    if critique_match and improved_match:
        critique_section = critique_match.group(1).strip()
        final_refined_ideas = improved_match.group(1).strip()
    else:
        # fallback if parsing fails
        critique_section = "Could not parse critique."
        final_refined_ideas = refinement_output.strip()

    return critique_section, final_refined_ideas

########################################
# STREAMLIT UI
########################################

st.title("Alignment Research Automator")

model_choice = st.selectbox("Select LLM Model:", ["gpt-4o", "o1-mini", "o1-preview"])
verbose = st.checkbox("Show verbose debug info")

# User-defined parameters with defaults
max_iterations = st.number_input("Max Refinement Iterations:", min_value=1, value=3)
enable_different_angle = st.checkbox("Attempt Different Angle if stuck?", value=True)

user_text = st.text_area("Paste your research text (abstract or section):", height=300)
additional_context = st.text_area("Optional additional angles or considerations:", height=100)

if st.button("Run Pipeline"):
    if not user_text.strip():
        st.warning("Please provide some text before running the pipeline.")
    else:
        # Step 1: Summarize/Overview
        overview = summarize_text(user_text, additional_context, model_choice, verbose)
        with st.expander("Comprehensive Systematic Overview"):
            st.write(overview)

        # Step 2: Initial Ideas
        ideas = generate_ideas(overview, additional_context, model_choice, verbose)
        with st.expander("Initial Alignment Ideas"):
            st.write(ideas)

        refined_ideas = ideas
        iteration_count = 0
        termination_feedback = ""

        final_good_enough = False
        while iteration_count < max_iterations:
            # Refine ideas
            refinement_output = refine_ideas(refined_ideas, termination_feedback, model_choice, verbose)
            critique_section, final_refined_ideas = parse_refinement_output(refinement_output)

            with st.expander(f"Iteration {iteration_count+1} Critique"):
                st.write(critique_section)
            with st.expander(f"Iteration {iteration_count+1} Refined Ideas"):
                st.write(final_refined_ideas)

            verdict = check_termination(final_refined_ideas, model_choice, verbose)
            st.write("Termination Check:", verdict)

            if verdict.startswith("Good enough"):
                final_good_enough = True
                break
            else:
                # Needs more iteration
                # Extract reason if "Needs more iteration"
                reason_match = re.match(r"Needs more iteration:\s*(.*)", verdict)
                if reason_match:
                    termination_reason = reason_match.group(1).strip()
                else:
                    termination_reason = "No specific feedback provided."

                # Now accommodate feedback directly
                feedback_output = accommodate_feedback(final_refined_ideas, termination_reason, model_choice, verbose)
                # Parse again after accommodating feedback
                fb_critique_section, fb_final_ideas = parse_refinement_output(feedback_output)

                with st.expander(f"Accommodating Feedback after Iteration {iteration_count+1}"):
                    st.write("Critique Reasoning:")
                    st.write(fb_critique_section)
                    st.write("New Improved Ideas:")
                    st.write(fb_final_ideas)

                refined_ideas = fb_final_ideas
                iteration_count += 1
                # Check again after accommodation
                verdict = check_termination(refined_ideas, model_choice, verbose)
                st.write("Termination Check after accommodation:", verdict)
                if verdict.startswith("Good enough"):
                    final_good_enough = True
                    break
                else:
                    reason_match = re.match(r"Needs more iteration:\s*(.*)", verdict)
                    if reason_match:
                        termination_feedback = "Previous termination feedback: " + reason_match.group(1).strip()
                    else:
                        termination_feedback = "No specific feedback provided."

        if not final_good_enough and enable_different_angle:
            st.warning("Ideas still not good enough after multiple refinements. Attempting a different angle...")
            diff_ideas = try_different_angle(refined_ideas, model_choice, verbose)
            with st.expander("Different Angle Ideas"):
                st.write(diff_ideas)
            diff_verdict = check_termination(diff_ideas, model_choice, verbose)
            st.write("Final Termination Check After Different Angle:", diff_verdict)
            if diff_verdict.startswith("Good enough"):
                final_good_enough = True
                final_refined_ideas = diff_ideas
            else:
                st.info("Even after a different angle, more iteration might be needed. Consider revising input or approach.")

        if final_good_enough:
            with st.expander("Final Accepted Ideas", expanded=True):
                st.write(final_refined_ideas)
