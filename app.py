import streamlit as st
import re
import anthropic
import io
from openai import OpenAI
from PyPDF2 import PdfReader

# Initialize instructions
st.title("Alignment Research Automator")
st.write("This tool takes a research text, summarizes it, proposes alignment research directions, critiques them, and refines them based on termination criteria. The goal is to produce actionable, high-quality alignment research directions.")
st.write("You can choose different LLMs for each stage (Summarizer, Ideator, Critic, Terminator) and run multiple iterations. If after N iterations it still isn't good enough, final output and a strengths/weaknesses assessment is displayed.")
st.write("You can upload a PDF or paste text. If both are provided, PDF content takes precedence.")
st.write("Within each expander, you can chat with an LLM to further explore the displayed content. You can also download the contents of the expander as markdown.")
st.write("**New**: Choose 'Manual Mode' to manually intervene at each step in the pipeline, or leave it off for an automatic run. Manual mode allows repeated back-and-forth with the model during each step before proceeding to the next step. No pipeline state is lost when you do this step-by-step process.")

# Ensure session state keys exist
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False
if "conversation_states" not in st.session_state:
    st.session_state.conversation_states = {}  # key -> list of messages
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}
if "manual_current_step" not in st.session_state:
    st.session_state.manual_current_step = 0
if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False

# Secrets
openai_client = OpenAI(api_key=st.secrets['API_KEY'])
anthropic_client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])

########################################
# MODEL HANDLING
########################################

def is_o1_model(m):
    return m.startswith("o1-")

def is_gpt4o_model(m):
    return m == "gpt-4o"

def is_anthropic_model(m):
    return m.startswith("anthropic")

def adjust_messages_for_o1(messages):
    adjusted = []
    for msg in messages:
        if msg["role"] == "system":
            # Convert to user
            content = "INSTRUCTIONS:\n" + msg["content"]
            adjusted.append({"role": "user", "content": content})
        else:
            adjusted.append(msg)
    return adjusted

def run_completion(messages, model, verbose=False):
    if is_o1_model(model):
        messages = adjust_messages_for_o1(messages)
        if verbose:
            st.write("**Using O1 Model:**", model)
            st.write("**Messages:**", messages)
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        response = completion.choices[0].message.content
        if verbose:
            st.write("**Response:**", response)
        return response
    elif is_gpt4o_model(model):
        if verbose:
            st.write("**Using GPT-4o Model:**", model)
            st.write("**Messages:**", messages)
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        response = completion.choices[0].message.content
        if verbose:
            st.write("**Response:**", response)
        return response
    elif is_anthropic_model(model):
        if verbose:
            st.write("**Using Anthropic Model:**", model)
            st.write("**Messages:**", messages)
        system_str_list = [m["content"] for m in messages if m["role"] == "system"]
        user_assistant_msgs = [m for m in messages if m["role"] != "system"]
        system_str = "\n".join(system_str_list) if system_str_list else None

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",  # Example placeholder model name
            "max_tokens": 1000,
            "messages": user_assistant_msgs
        }
        if system_str:
            kwargs["system"] = system_str

        response = anthropic_client.messages.create(**kwargs)
        response = response.content[0].text
        if verbose:
            st.write("**Response:**", response)
        return response.strip()

########################################
# PROMPTS
########################################

SUMMARIZER_SYSTEM = """You are a specialized research assistant focused on analyzing scholarly research text and identifying AI alignment implications.
Provide a comprehensive, systematic overview:
- Domain, key hypotheses.
- Methods, results, theoretical contributions.
Be factual, thorough, and substantive no vague language.
"""

SUMMARIZER_USER = """Below is a research text. Provide the comprehensive overview as requested above. It should be extremely substantive and should serve as a functional stand-in for someone reading the paper. Leave out none of the 'meat.'

Research Text:
{user_text}
"""

IDEATOR_SYSTEM = """You are an AI alignment research ideation agent.
Steps:
1. Given the overview, propose original alignment research directions.
2. Then we will critique them (no fix at that step).
3. Then you will produce updated research directions incorporating the critique seamlessly so the final output is standalone and improved.
"""

IDEATOR_IDEATION_USER = """Use the overview and the additional angles below to propose several original AI alignment research directions.

Systematic Overview:
{overview}

Additional angles to consider optionally inputted by the user:
{additional_context}
"""

CRITIC_SYSTEM = """You are a critic focused on assessing alignment research directions for clarity, novelty, feasibility. Do not fix them, just critique:
- Identify weaknesses, if any
- Suggest what is missing, if anything
- Indicate what could be improved, if anything

Just critique, do not provide improved ideas yet. Do not critique things that are already great, and you can ask things like "throw out ideas X,Y,Z and just focus on refining A and B" or "keep all ideas but make them all better in XYZ ways"
"""

CRITIC_USER = """Critique the following AI alignment research directions for clarity, novelty, feasibility, and actionable value.

Directions:
{ideas}
"""

RE_IDEATE_SYSTEM = """You are an AI alignment ideation agent revisiting the directions after critique.
Instructions:
- Consider the critique carefully.
- Produce a new standalone set of improved directions that address the critique.
- The new ideas should read as original, not a response to critique. Do not mention the critique explicitly, just integrate improvements.

Use format:
CRITIQUE_START
(Brief reasoning about what needed improvement)
CRITIQUE_END

IMPROVED_START
(Your improved, standalone directions)
IMPROVED_END
"""

RE_IDEATE_USER = """Here are the previous directions and the critique:

Previous Directions:
{ideas}

Critique:
{critique}

Now produce improved directions as per instructions.
"""

TERMINATOR_SYSTEM = """You are an evaluator of alignment research directions.
If good enough to present back to the user, say "Good enough".
If not, say "Needs more iteration: <reason>" where <reason> is clear and specific.
No other text.
"""

TERMINATOR_USER = """Evaluate these improved alignment directions:

{refined_ideas}

Are they good enough or need more iteration?
"""

TERMINATOR_ASSESS_SYSTEM = """You are an evaluator.
Given the final alignment directions, provide a strengths and weaknesses assessment since they were not approved after multiple iterations. Be factual and concise.
"""

TERMINATOR_ASSESS_USER = """These are the final directions after all attempts:

{final_ideas}

Please provide their strengths and weaknesses.
"""

########################################
# WORKFLOW FUNCTIONS
########################################

def summarize_text(user_text, model, verbose):
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        {"role": "user", "content": SUMMARIZER_USER.format(user_text=user_text)}
    ]
    return run_completion(messages, model, verbose=verbose)

def ideate(overview, additional_context, model, verbose):
    messages = [
        {"role": "system", "content": IDEATOR_SYSTEM},
        {"role": "user", "content": IDEATOR_IDEATION_USER.format(overview=overview, additional_context=additional_context)}
    ]
    return run_completion(messages, model, verbose=verbose)

def critique(ideas, model, verbose):
    messages = [
        {"role": "system", "content": CRITIC_SYSTEM},
        {"role": "user", "content": CRITIC_USER.format(ideas=ideas)}
    ]
    return run_completion(messages, model, verbose=verbose)

def re_ideate(ideas, critique_text, model, verbose):
    messages = [
        {"role": "system", "content": RE_IDEATE_SYSTEM},
        {"role": "user", "content": RE_IDEATE_USER.format(ideas=ideas, critique=critique_text)}
    ]
    return run_completion(messages, model, verbose=verbose)

def check_termination(refined_ideas, model, verbose):
    messages = [
        {"role": "system", "content": TERMINATOR_SYSTEM},
        {"role": "user", "content": TERMINATOR_USER.format(refined_ideas=refined_ideas)}
    ]
    return run_completion(messages, model, verbose=verbose)

def assess_strengths_weaknesses(final_ideas, model, verbose):
    messages = [
        {"role": "system", "content": TERMINATOR_ASSESS_SYSTEM},
        {"role": "user", "content": TERMINATOR_ASSESS_USER.format(final_ideas=final_ideas)}
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
        critique_section = "Could not parse critique."
        final_refined_ideas = refinement_output.strip()

    return critique_section, final_refined_ideas

########################################
# CHAT WITHIN EXPANDER
########################################

def ensure_conversation_state(key):
    if key not in st.session_state.conversation_states:
        st.session_state.conversation_states[key] = []

def chat_interface(key, base_content, model, verbose):
    ensure_conversation_state(key)
    # Display existing chat messages
    for msg in st.session_state.conversation_states[key]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    prompt = st.chat_input("Ask a question or refine this content further", key=f"chat_input_{key}")
    if prompt:
        # User message
        st.session_state.conversation_states[key].append({"role":"user","content":prompt})
        # Call LLM with base_content + conversation
        conv_messages = [{"role":"system","content":"You are a helpful assistant. You have the following content:\n\n" + base_content + "\n\nUser and assistant messages follow. Answer user queries helpfully."}]
        for m in st.session_state.conversation_states[key]:
            conv_messages.append(m)
        response = run_completion(conv_messages, model, verbose)
        st.session_state.conversation_states[key].append({"role":"assistant","content":response})
        st.rerun()

def download_expander_content(label, content):
    return st.download_button(
        label="Download contents as markdown",
        data=content.encode("utf-8"),
        file_name=f"{label.replace(' ','_')}.md",
        mime="text/markdown",
    )

########################################
# PDF UPLOAD
########################################

pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
pdf_text = ""
if pdf_file is not None:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"

llm_options = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]

summarizer_model = st.selectbox("Summarizer Model:", llm_options, index=0)
ideator_model = st.selectbox("Ideator Model:", llm_options, index=0)
critic_model = st.selectbox("Critic Model:", llm_options, index=0)
terminator_model = st.selectbox("Terminator Model:", llm_options, index=0)

verbose = st.checkbox("Show verbose debug info")

max_iterations = st.number_input("Max Refinement Iterations:", min_value=1, value=3)
enable_different_angle = st.checkbox("Attempt Different Angle if stuck?", value=False)

manual_mode = st.checkbox("Manual Mode", value=st.session_state.manual_mode)
st.session_state.manual_mode = manual_mode

user_text = st.text_area("Paste your research text (abstract or section):", height=300)
additional_context = st.text_area("Optional additional angles or considerations:", height=100)

run_pipeline = st.button("Run Pipeline")

########################################
# Manual Mode Step-by-Step Execution
########################################

def run_summarization_step():
    st.session_state.pipeline_results["overview"] = summarize_text(
        st.session_state.final_input_text,
        summarizer_model,
        verbose
    )

def run_ideation_step():
    st.session_state.pipeline_results["initial_ideas"] = ideate(
        st.session_state.pipeline_results["overview"],
        additional_context,
        ideator_model,
        verbose
    )

def run_critique_step(current_ideas):
    return critique(current_ideas, critic_model, verbose)

def run_re_ideate_step(current_ideas, critique_text):
    return re_ideate(current_ideas, critique_text, ideator_model, verbose)

def run_termination_check(refined_ideas):
    return check_termination(refined_ideas, terminator_model, verbose)

def run_strengths_weaknesses_assessment(current_ideas):
    return assess_strengths_weaknesses(current_ideas, terminator_model, verbose)

########################################
# Main Execution Logic
########################################

if run_pipeline:
    # Reset everything when pipeline is run
    st.session_state.conversation_states.clear()
    st.session_state.pipeline_results.clear()
    st.session_state.pipeline_ran = False
    st.session_state.manual_current_step = 0

    # Decide which text to use
    st.session_state.final_input_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()

    if not st.session_state.final_input_text:
        st.warning("Please provide text or upload a PDF before running the pipeline.")
    else:
        if not manual_mode:
            # AUTOMATIC PIPELINE
            overview = summarize_text(st.session_state.final_input_text, summarizer_model, verbose)
            initial_ideas = ideate(overview, additional_context, ideator_model, verbose)

            iteration_count = 0
            final_good_enough = False
            current_ideas = initial_ideas

            iteration_data = []

            while iteration_count < max_iterations:
                critique_text = critique(current_ideas, critic_model, verbose)
                refinement_output = re_ideate(current_ideas, critique_text, ideator_model, verbose)
                critique_section, improved_ideas = parse_refinement_output(refinement_output)

                verdict = check_termination(improved_ideas, terminator_model, verbose)
                if verdict.startswith("Good enough"):
                    final_good_enough = True
                    current_ideas = improved_ideas
                    iteration_data.append((critique_text, critique_section, improved_ideas, verdict))
                    break
                else:
                    iteration_data.append((critique_text, critique_section, improved_ideas, verdict))
                    current_ideas = improved_ideas
                    iteration_count += 1

            # If not good enough after max iterations
            if not final_good_enough:
                assessment = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
                st.session_state.pipeline_results["assessment"] = assessment
            else:
                st.session_state.pipeline_results["assessment"] = None

            # Store pipeline results
            st.session_state.pipeline_results["overview"] = overview
            st.session_state.pipeline_results["initial_ideas"] = initial_ideas
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.session_state.pipeline_results["final_good_enough"] = final_good_enough
            st.session_state.pipeline_results["final_ideas"] = current_ideas
            st.session_state.pipeline_results["additional_context"] = additional_context

            st.session_state.pipeline_ran = True
            st.rerun()

        else:
            # MANUAL MODE
            # We will handle step by step. The steps:
            # 0: Summarize
            # 1: Ideate
            # 2: Critique
            # 3: Re-Ideate
            # 4: Check Termination
            # If not done, keep looping until max iterations or user is satisfied

            # Initialize iteration data if not present
            if "iteration_data" not in st.session_state.pipeline_results:
                st.session_state.pipeline_results["iteration_data"] = []
            st.session_state.pipeline_ran = True
            st.session_state.manual_current_step = 0  # Start with Summarization
            st.rerun()

# If pipeline has started in manual mode, handle the step-by-step UI
if st.session_state.pipeline_ran and st.session_state.manual_mode and st.session_state.final_input_text:
    iteration_data = st.session_state.pipeline_results.get("iteration_data", [])
    current_step = st.session_state.manual_current_step

    # We keep track of up to max_iterations
    # Steps in each iteration: Summarize (only once at iteration 0), Ideate, Critique, Re-ideate, Terminate
    # Once we finalize, we either go next iteration or conclude

    if current_step == 0:
        st.subheader("Step 1: Summarization")
        if "overview" not in st.session_state.pipeline_results:
            st.info("Click 'Run Summarization' below to generate the summary. You can rerun it or chat with it before proceeding.")
        else:
            st.write(st.session_state.pipeline_results["overview"])
            download_expander_content("overview", st.session_state.pipeline_results["overview"])
            chat_interface("overview", st.session_state.pipeline_results["overview"], summarizer_model, verbose)

        if st.button("Run Summarization"):
            run_summarization_step()
            st.rerun()

        if "overview" in st.session_state.pipeline_results:
            if st.button("Proceed to Next Step (Ideation)"):
                st.session_state.manual_current_step = 1
                st.rerun()

    elif current_step == 1:
        st.subheader("Step 2: Ideation")
        if "initial_ideas" not in st.session_state.pipeline_results:
            st.info("Click 'Run Ideation' to generate alignment ideas.")
        else:
            st.write(st.session_state.pipeline_results["initial_ideas"])
            download_expander_content("initial_alignment_ideas", st.session_state.pipeline_results["initial_ideas"])
            chat_interface("initial_ideas", st.session_state.pipeline_results["initial_ideas"], ideator_model, verbose)

        if st.button("Run Ideation"):
            run_ideation_step()
            st.rerun()

        if "initial_ideas" in st.session_state.pipeline_results:
            # We haven't done any iterative steps yet, so set up iteration data
            if len(iteration_data) == 0:
                iteration_data.append(("", "", st.session_state.pipeline_results["initial_ideas"], "")) 
                st.session_state.pipeline_results["iteration_data"] = iteration_data
            if st.button("Proceed to Next Step (Critique)"):
                st.session_state.manual_current_step = 2
                st.rerun()

    elif current_step == 2:
        st.subheader("Step 3: Critique")
        # The current ideas to critique is iteration_data[-1][2]
        current_ideas = iteration_data[-1][2]
        critique_text = iteration_data[-1][0]  # if it exists
        if not critique_text:
            st.info("Click 'Run Critique' to critique the current ideas.")
        else:
            st.write(critique_text)
            download_expander_content(f"iteration_{len(iteration_data)}_critique", critique_text)
            chat_interface(f"iteration_{len(iteration_data)}_critique", critique_text, critic_model, verbose)

        if st.button("Run Critique"):
            new_critique = run_critique_step(current_ideas)
            # Overwrite iteration_data for this iteration
            # iteration_data entry: (critique_text, critique_section, improved_ideas, verdict)
            # We'll fill out partial. We'll set improved_ideas and verdict to "" until we re-ideate.
            iteration_data[-1] = (new_critique, "", current_ideas, "")
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.rerun()

        if iteration_data[-1][0]:  # If we have a critique
            if st.button("Proceed to Next Step (Re-Ideate)"):
                st.session_state.manual_current_step = 3
                st.rerun()

    elif current_step == 3:
        st.subheader("Step 4: Re-Ideate")
        # We have the critique in iteration_data[-1][0], current_ideas in iteration_data[-1][2]
        critique_text = iteration_data[-1][0]
        current_ideas = iteration_data[-1][2]
        refinement_output = iteration_data[-1][1]  # stored in critique_section, but let's reuse

        if not refinement_output:  
            st.info("Click 'Run Re-Ideate' to incorporate the critique into new directions.")
        else:
            st.write("**Internal Reasoning (CRITIQUE_START/END):**")
            st.write(refinement_output)
            download_expander_content(f"iteration_{len(iteration_data)}_internal_reasoning", refinement_output)
            chat_interface(f"iteration_{len(iteration_data)}_internal_reasoning", refinement_output, ideator_model, verbose)

            st.write("**Refined Directions:**")
            st.write(iteration_data[-1][2])  # we update improved ideas below after parse
            download_expander_content(f"iteration_{len(iteration_data)}_refined_directions", iteration_data[-1][2])
            chat_interface(f"iteration_{len(iteration_data)}_refined_directions", iteration_data[-1][2], ideator_model, verbose)

        if st.button("Run Re-Ideate"):
            refinement_out = run_re_ideate_step(current_ideas, critique_text)
            # parse
            critique_section, improved_ideas = parse_refinement_output(refinement_out)
            # iteration_data entry: (critique_text, critique_section, improved_ideas, verdict)
            iteration_data[-1] = (critique_text, critique_section, improved_ideas, "")
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.rerun()

        if iteration_data[-1][1]:  # We have some re-ideation
            if st.button("Proceed to Next Step (Termination Check)"):
                st.session_state.manual_current_step = 4
                st.rerun()

    elif current_step == 4:
        st.subheader("Step 5: Termination Check")
        # iteration_data[-1] has (critique_text, critique_section, improved_ideas, verdict)
        improved_ideas = iteration_data[-1][2]
        verdict = iteration_data[-1][3]
        if not verdict:
            st.info("Click 'Run Termination Check' to see if it's good enough or needs more iteration.")
        else:
            st.write(f"Termination Check: {verdict}")

        if st.button("Run Termination Check"):
            new_verdict = run_termination_check(improved_ideas)
            iteration_data[-1] = (iteration_data[-1][0],
                                  iteration_data[-1][1],
                                  iteration_data[-1][2],
                                  new_verdict)
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.rerun()

        # If we have a verdict, check if "Good enough"
        if iteration_data[-1][3]:
            new_verdict = iteration_data[-1][3]
            if new_verdict.startswith("Good enough"):
                st.success("The directions are approved!")
                st.session_state.pipeline_results["final_good_enough"] = True
                st.session_state.pipeline_results["final_ideas"] = iteration_data[-1][2]
                st.session_state.manual_current_step = 999  # done
                st.rerun()
            else:
                # Not good enough
                if len(iteration_data) < max_iterations:
                    st.warning("It needs more iteration. You can proceed to next iteration if you'd like.")
                    if st.button("Proceed to Another Iteration"):
                        # Start another iteration
                        # That means we treat iteration_data[-1][2] as the new ideas, go to step (2) critique again
                        iteration_data.append(("", "", iteration_data[-1][2], ""))
                        st.session_state.pipeline_results["iteration_data"] = iteration_data
                        st.session_state.manual_current_step = 2  # go back to Critique
                        st.rerun()
                else:
                    st.warning("Reached max iterations. We'll finalize with a strengths & weaknesses assessment.")
                    st.session_state.pipeline_results["final_good_enough"] = False
                    st.session_state.pipeline_results["final_ideas"] = iteration_data[-1][2]
                    # Move to final step
                    st.session_state.manual_current_step = 999
                    st.rerun()

    elif current_step == 999:
        # Conclude
        final_good_enough = st.session_state.pipeline_results.get("final_good_enough", False)
        final_ideas = st.session_state.pipeline_results.get("final_ideas", "")
        iteration_data = st.session_state.pipeline_results["iteration_data"]

        if final_good_enough:
            # Show final ideas
            with st.expander("Final Accepted Ideas", expanded=True):
                st.write(final_ideas)
                download_expander_content("final_accepted_ideas", final_ideas)
                chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
        else:
            # Not good enough
            st.warning("Not approved after max iterations. Displaying final output and assessing strengths & weaknesses.")
            with st.expander("Final Non-Approved Ideas"):
                st.write(final_ideas)
                download_expander_content("final_non_approved_ideas", final_ideas)
                chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

            if "assessment" not in st.session_state.pipeline_results:
                assessment = run_strengths_weaknesses_assessment(final_ideas)
                st.session_state.pipeline_results["assessment"] = assessment
            else:
                assessment = st.session_state.pipeline_results["assessment"]

            with st.expander("Strengths & Weaknesses Assessment", expanded=True):
                st.write(assessment)
                download_expander_content("strengths_weaknesses_assessment", assessment)
                chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)

        st.success("Manual pipeline run complete.")
        st.stop()

########################################
# Automatic Mode Results Display (if pipeline ran automatically)
########################################

if st.session_state.pipeline_ran and not st.session_state.manual_mode:
    overview = st.session_state.pipeline_results["overview"]
    initial_ideas = st.session_state.pipeline_results["initial_ideas"]
    iteration_data = st.session_state.pipeline_results["iteration_data"]
    final_good_enough = st.session_state.pipeline_results["final_good_enough"]
    final_ideas = st.session_state.pipeline_results["final_ideas"]
    assessment = st.session_state.pipeline_results["assessment"]
    additional_context = st.session_state.pipeline_results["additional_context"]

    with st.expander("Comprehensive Systematic Overview"):
        st.write(overview)
        download_expander_content("overview", overview)
        chat_interface("overview", overview, ideator_model, verbose)

    with st.expander("Initial Alignment Ideas"):
        st.write(initial_ideas)
        download_expander_content("initial_alignment_ideas", initial_ideas)
        chat_interface("initial_ideas", initial_ideas, ideator_model, verbose)

    for i, (critique_text, critique_section, improved_ideas, verdict) in enumerate(iteration_data, start=1):
        with st.expander(f"Iteration {i} Critique"):
            st.write(critique_text)
            download_expander_content(f"iteration_{i}_critique", critique_text)
            chat_interface(f"iteration_{i}_critique", critique_text, ideator_model, verbose)

        with st.expander(f"Iteration {i} Internal Reasoning (CRITIQUE_START/END)"):
            st.write(critique_section)
            download_expander_content(f"iteration_{i}_internal_reasoning", critique_section)
            chat_interface(f"iteration_{i}_internal_reasoning", critique_section, ideator_model, verbose)

        with st.expander(f"Iteration {i} Refined Directions"):
            st.write(improved_ideas)
            download_expander_content(f"iteration_{i}_refined_directions", improved_ideas)
            chat_interface(f"iteration_{i}_refined_directions", improved_ideas, ideator_model, verbose)

        st.write(f"Termination Check: {verdict}")

    if final_good_enough:
        with st.expander("Final Accepted Ideas", expanded=True):
            st.write(final_ideas)
            download_expander_content("final_accepted_ideas", final_ideas)
            chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
    else:
        st.warning("Not approved after max iterations. Displaying final output and assessing strengths & weaknesses.")
        with st.expander("Final Non-Approved Ideas"):
            st.write(final_ideas)
            download_expander_content("final_non_approved_ideas", final_ideas)
            chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

        if assessment is not None:
            with st.expander("Strengths & Weaknesses Assessment", expanded=True):
                st.write(assessment)
                download_expander_content("strengths_weaknesses_assessment", assessment)
                chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
