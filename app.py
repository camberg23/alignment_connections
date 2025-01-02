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
st.write("In 'Manual' mode, each step can be re-run, refined, or advanced to the next step at your discretion. In 'Automatic' mode, the entire pipeline runs in one shot.")

# Ensure session state keys exist
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False
if "conversation_states" not in st.session_state:
    st.session_state.conversation_states = {}  # key -> list of messages
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}
if "manual_step" not in st.session_state:
    st.session_state.manual_step = "not_started"

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
            "model": "claude-3-5-sonnet-20241022",  # Example model name
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
Be factual, thorough, and substantive; no vague language.
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

Just critique, do not provide improved ideas yet.
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
# CHAT-LIKE EXPANDER
########################################

def ensure_conversation_state(key):
    if key not in st.session_state.conversation_states:
        st.session_state.conversation_states[key] = []

def chat_interface(key, base_content, model, verbose):
    ensure_conversation_state(key)
    for msg in st.session_state.conversation_states[key]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    prompt = st.chat_input("Ask a question or refine this content further", key=f"chat_input_{key}")
    if prompt:
        st.session_state.conversation_states[key].append({"role":"user","content":prompt})
        conv_messages = [
            {"role":"system","content":f"You are a helpful assistant. You have the following content:\n\n{base_content}\n\nUser and assistant messages follow. Answer user queries helpfully."}
        ]
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
max_iterations = st.number_input("Max Refinement Iterations (Auto Mode):", min_value=1, value=3)
enable_different_angle = st.checkbox("Attempt Different Angle if stuck? (Auto Mode)", value=False)

user_text = st.text_area("Paste your research text (abstract or section):", height=300)
additional_context = st.text_area("Optional additional angles or considerations:", height=100)

# New: Automatic vs Manual mode
process_mode = st.radio("Process Mode:", ["Automatic", "Manual"], index=0)

run_pipeline = st.button("Run Pipeline")

########################################
# MANUAL MODE HANDLERS
########################################

def manual_summarize():
    final_input_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()
    if not final_input_text:
        st.warning("No text to summarize.")
        return

    # Summarize
    st.session_state.pipeline_results["overview"] = summarize_text(
        final_input_text, summarizer_model, verbose
    )
    st.session_state.manual_step = "summarized"

def manual_ideate():
    if "overview" not in st.session_state.pipeline_results:
        st.warning("No overview found. Summarize first.")
        return
    st.session_state.pipeline_results["initial_ideas"] = ideate(
        st.session_state.pipeline_results["overview"], 
        additional_context, 
        ideator_model, 
        verbose
    )
    st.session_state.manual_step = "ideated"

def manual_critique():
    if "initial_ideas" not in st.session_state.pipeline_results:
        st.warning("No ideas to critique. Ideate first.")
        return
    st.session_state.pipeline_results["critique_text"] = critique(
        st.session_state.pipeline_results["initial_ideas"], critic_model, verbose
    )
    st.session_state.manual_step = "critiqued"

def manual_re_ideate():
    if "critique_text" not in st.session_state.pipeline_results:
        st.warning("No critique to refine from. Critique first.")
        return
    # Re-ideate
    refinement_output = re_ideate(
        st.session_state.pipeline_results["initial_ideas"], 
        st.session_state.pipeline_results["critique_text"], 
        ideator_model, 
        verbose
    )
    critique_section, improved_ideas = parse_refinement_output(refinement_output)
    st.session_state.pipeline_results["refinement_output"] = refinement_output
    st.session_state.pipeline_results["critique_section"] = critique_section
    st.session_state.pipeline_results["improved_ideas"] = improved_ideas
    st.session_state.manual_step = "refined"

def manual_terminate():
    if "improved_ideas" not in st.session_state.pipeline_results:
        st.warning("No improved ideas to evaluate.")
        return
    verdict = check_termination(st.session_state.pipeline_results["improved_ideas"], terminator_model, verbose)
    st.session_state.pipeline_results["verdict"] = verdict
    if verdict.startswith("Good enough"):
        st.session_state.manual_step = "approved"
    else:
        st.session_state.manual_step = "needs_iteration"

def manual_assess():
    # Only if we ended with "Needs more iteration" or we hit manual end
    if "improved_ideas" not in st.session_state.pipeline_results:
        st.warning("No final ideas to assess.")
        return
    assessment = assess_strengths_weaknesses(st.session_state.pipeline_results["improved_ideas"], terminator_model, verbose)
    st.session_state.pipeline_results["assessment"] = assessment
    st.session_state.manual_step = "assessed"

########################################
# MAIN PIPELINE LOGIC
########################################

if run_pipeline:
    st.session_state.conversation_states.clear()
    st.session_state.pipeline_results.clear()
    st.session_state.pipeline_ran = True
    st.session_state.manual_step = "started"

    if process_mode == "Automatic":
        # Exactly as the old pipeline
        final_input_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()

        if not final_input_text:
            st.warning("Please provide text or upload a PDF before running the pipeline.")
        else:
            overview = summarize_text(final_input_text, summarizer_model, verbose)
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
                # Assess strengths & weaknesses
                assessment = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
                st.session_state.pipeline_results["assessment"] = assessment
            else:
                st.session_state.pipeline_results["assessment"] = None

            # Store pipeline results in session state
            st.session_state.pipeline_results["overview"] = overview
            st.session_state.pipeline_results["initial_ideas"] = initial_ideas
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.session_state.pipeline_results["final_good_enough"] = final_good_enough
            st.session_state.pipeline_results["final_ideas"] = current_ideas
            st.session_state.pipeline_results["additional_context"] = additional_context

    else:
        # Manual mode: we simply set st.session_state.manual_step = "started"
        # Then we rely on step-by-step user interaction below
        pass

########################################
# RENDER RESULTS
########################################

if process_mode == "Manual" and st.session_state.pipeline_ran:
    # Show each step in order
    if st.session_state.manual_step in ["started", "not_started"]:
        st.subheader("Manual Step 1: Summarize")
        if st.button("Run Summarizer"):
            manual_summarize()
        st.write("Once satisfied, click 'Next Step' to proceed.")
        if "overview" in st.session_state.pipeline_results:
            with st.expander("Summarizer Output"):
                st.write(st.session_state.pipeline_results["overview"])
                download_expander_content("overview", st.session_state.pipeline_results["overview"])
                chat_interface("manual_summarizer", st.session_state.pipeline_results["overview"], summarizer_model, verbose)
        if st.button("Next Step", key="summarizer_next"):
            if "overview" not in st.session_state.pipeline_results:
                st.warning("No summary found. Summarize first.")
            else:
                st.session_state.manual_step = "summarized"

    if st.session_state.manual_step == "summarized":
        st.subheader("Manual Step 2: Ideate")
        if st.button("Run Ideator"):
            manual_ideate()
        if "initial_ideas" in st.session_state.pipeline_results:
            with st.expander("Ideation Output"):
                st.write(st.session_state.pipeline_results["initial_ideas"])
                download_expander_content("initial_alignment_ideas", st.session_state.pipeline_results["initial_ideas"])
                chat_interface("manual_ideator", st.session_state.pipeline_results["initial_ideas"], ideator_model, verbose)
        if st.button("Next Step", key="ideator_next"):
            if "initial_ideas" not in st.session_state.pipeline_results:
                st.warning("No ideas found. Please run ideation first.")
            else:
                st.session_state.manual_step = "ideated"

    if st.session_state.manual_step == "ideated":
        st.subheader("Manual Step 3: Critique")
        if st.button("Run Critique"):
            manual_critique()
        if "critique_text" in st.session_state.pipeline_results:
            with st.expander("Critique Output"):
                st.write(st.session_state.pipeline_results["critique_text"])
                download_expander_content("iteration_critique", st.session_state.pipeline_results["critique_text"])
                chat_interface("manual_critic", st.session_state.pipeline_results["critique_text"], critic_model, verbose)
        if st.button("Next Step", key="critique_next"):
            if "critique_text" not in st.session_state.pipeline_results:
                st.warning("No critique found. Please run critique first.")
            else:
                st.session_state.manual_step = "critiqued"

    if st.session_state.manual_step == "critiqued":
        st.subheader("Manual Step 4: Re-Ideate (Refine Ideas)")
        if st.button("Run Re-Ideation"):
            manual_re_ideate()
        if "refinement_output" in st.session_state.pipeline_results:
            with st.expander("Refinement Output"):
                st.write(st.session_state.pipeline_results["refinement_output"])
                download_expander_content("iteration_refinement_output", st.session_state.pipeline_results["refinement_output"])
            with st.expander("Extracted CRITIQUE_START/END"):
                st.write(st.session_state.pipeline_results["critique_section"])
                download_expander_content("iteration_internal_reasoning", st.session_state.pipeline_results["critique_section"])
            if "improved_ideas" in st.session_state.pipeline_results:
                with st.expander("Refined Directions"):
                    st.write(st.session_state.pipeline_results["improved_ideas"])
                    download_expander_content("iteration_refined_directions", st.session_state.pipeline_results["improved_ideas"])
                    chat_interface("manual_refined_ideas", st.session_state.pipeline_results["improved_ideas"], ideator_model, verbose)

        if st.button("Next Step", key="refine_next"):
            if "improved_ideas" not in st.session_state.pipeline_results:
                st.warning("No refined ideas found. Please run re-ideation first.")
            else:
                st.session_state.manual_step = "refined"

    if st.session_state.manual_step == "refined":
        st.subheader("Manual Step 5: Terminator Check")
        if st.button("Run Terminator Check"):
            manual_terminate()
        if "verdict" in st.session_state.pipeline_results:
            st.write(f"Termination Check: {st.session_state.pipeline_results['verdict']}")
        if st.button("Next Step", key="terminate_next"):
            if "verdict" not in st.session_state.pipeline_results:
                st.warning("No verdict found. Please run terminator check first.")
            else:
                if st.session_state.pipeline_results["verdict"].startswith("Good enough"):
                    st.session_state.manual_step = "approved"
                else:
                    st.session_state.manual_step = "needs_iteration"

    if st.session_state.manual_step == "approved":
        st.subheader("Final Accepted Ideas")
        final_ideas = st.session_state.pipeline_results.get("improved_ideas", "No ideas found.")
        st.write(final_ideas)
        download_expander_content("final_accepted_ideas", final_ideas)
        chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
        st.success("Pipeline complete. Ideas were approved by the Terminator.")

    if st.session_state.manual_step == "needs_iteration":
        st.warning("Terminator said more iteration is needed. You can either keep refining or do a Strengths & Weaknesses Assessment as the final step.")
        if st.button("Run Strengths & Weaknesses Assessment"):
            manual_assess()

    if st.session_state.manual_step == "assessed":
        st.subheader("Strengths & Weaknesses Assessment")
        assessment = st.session_state.pipeline_results.get("assessment", "")
        st.write(assessment)
        download_expander_content("strengths_weaknesses_assessment", assessment)
        chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
        st.info("End of pipeline in manual mode. You can refine more or simply stop here.")

########################################
# AUTOMATIC MODE RESULTS DISPLAY
########################################

if process_mode == "Automatic" and st.session_state.pipeline_ran:
    overview = st.session_state.pipeline_results.get("overview", "")
    initial_ideas = st.session_state.pipeline_results.get("initial_ideas", "")
    iteration_data = st.session_state.pipeline_results.get("iteration_data", [])
    final_good_enough = st.session_state.pipeline_results.get("final_good_enough", False)
    final_ideas = st.session_state.pipeline_results.get("final_ideas", "")
    assessment = st.session_state.pipeline_results.get("assessment", None)
    additional_context = st.session_state.pipeline_results.get("additional_context", "")

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
        if final_ideas:
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
