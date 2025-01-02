import streamlit as st
import re
import anthropic
import io
from openai import OpenAI
from PyPDF2 import PdfReader

###############################################################################
# PAGE SETUP & STATE
###############################################################################
st.title("Alignment Research Automator")

st.write(
    "This app can run an AI alignment research pipeline in either Automatic or Manual mode."
    "\n\n**Automatic**: Summarize → Ideate → Critique → Re-Ideate → Terminator (→ Assess if not good)."
    "\n\n**Manual**: You control each step. Each step's output appears in an expander, along with a text-based chat."
)

# Session state keys
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False
if "manual_step" not in st.session_state:
    st.session_state.manual_step = "not_started"
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}
if "conversation_states" not in st.session_state:
    st.session_state.conversation_states = {}

###############################################################################
# SECRETS (ADJUST TO YOUR NEEDS)
###############################################################################
openai_client = OpenAI(api_key=st.secrets["API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

###############################################################################
# MODEL UTILS
###############################################################################
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
            # Convert system to user with "INSTRUCTIONS"
            content = "INSTRUCTIONS:\n" + msg["content"]
            adjusted.append({"role": "user", "content": content})
        else:
            adjusted.append(msg)
    return adjusted

def run_completion(messages, model, verbose=False):
    """
    Dispatch to different LLM endpoints based on model name.
    """
    if is_o1_model(model):
        messages = adjust_messages_for_o1(messages)
        if verbose:
            st.write("Using O1 Model:", model)
            st.write("Messages:", messages)
        completion = openai_client.chat.completions.create(model=model, messages=messages)
        response = completion.choices[0].message.content
        if verbose:
            st.write("Response:", response)
        return response

    elif is_gpt4o_model(model):
        if verbose:
            st.write("Using GPT-4o Model:", model)
            st.write("Messages:", messages)
        completion = openai_client.chat.completions.create(model=model, messages=messages)
        response = completion.choices[0].message.content
        if verbose:
            st.write("Response:", response)
        return response

    elif is_anthropic_model(model):
        if verbose:
            st.write("Using Anthropic Model:", model)
            st.write("Messages:", messages)
        system_str_list = [m["content"] for m in messages if m["role"] == "system"]
        user_assistant_msgs = [m for m in messages if m["role"] != "system"]
        system_str = "\n".join(system_str_list) if system_str_list else None

        kwargs = {
            "model": "claude-3-5-sonnet-20241022",  # Example name
            "max_tokens": 1000,
            "messages": user_assistant_msgs
        }
        if system_str:
            kwargs["system"] = system_str

        response = anthropic_client.messages.create(**kwargs)
        response = response.content[0].text
        if verbose:
            st.write("Response:", response)
        return response.strip()

    else:
        return "Unrecognized model."

###############################################################################
# PROMPTS
###############################################################################
SUMMARIZER_SYSTEM = """You are a specialized research assistant focusing on AI alignment. Summarize text thoroughly, with no vagueness."""
SUMMARIZER_USER = """Below is the research text. Provide a comprehensive, factual overview:
{user_text}
"""

IDEATOR_SYSTEM = """You are an AI alignment ideation agent."""
IDEATOR_IDEATION_USER = """Systematic Overview:
{overview}

Optional Additional Angles:
{additional_context}
"""

CRITIC_SYSTEM = """You are a critic analyzing AI alignment directions for clarity, novelty, feasibility, actionability."""
CRITIC_USER = """Directions:
{ideas}
"""

RE_IDEATE_SYSTEM = """You are refining directions after critique. 
Use format:
CRITIQUE_START
(...)
CRITIQUE_END
IMPROVED_START
(...)
IMPROVED_END
"""
RE_IDEATE_USER = """Previous Directions:
{ideas}

Critique:
{critique}
"""

TERMINATOR_SYSTEM = """You are an evaluator. If directions are good, say 'Good enough'. Else 'Needs more iteration: <reason>'."""
TERMINATOR_USER = """Evaluate these improved directions:
{refined_ideas}
"""

ASSESS_SYSTEM = """You are an evaluator providing strengths & weaknesses of the final directions."""
ASSESS_USER = """Final directions:
{final_ideas}
"""

###############################################################################
# PIPELINE FUNCTIONS
###############################################################################
def summarize_text(user_text, model, verbose=False):
    msgs = [
        {"role":"system","content":SUMMARIZER_SYSTEM},
        {"role":"user","content":SUMMARIZER_USER.format(user_text=user_text)},
    ]
    return run_completion(msgs, model, verbose)

def ideate(overview, additional_context, model, verbose=False):
    msgs = [
        {"role":"system","content":IDEATOR_SYSTEM},
        {
            "role":"user",
            "content":IDEATOR_IDEATION_USER.format(overview=overview, additional_context=additional_context)
        }
    ]
    return run_completion(msgs, model, verbose)

def critique(ideas, model, verbose=False):
    msgs = [
        {"role":"system","content":CRITIC_SYSTEM},
        {"role":"user","content":CRITIC_USER.format(ideas=ideas)}
    ]
    return run_completion(msgs, model, verbose)

def re_ideate(ideas, critique_text, model, verbose=False):
    msgs = [
        {"role":"system","content":RE_IDEATE_SYSTEM},
        {"role":"user","content":RE_IDEATE_USER.format(ideas=ideas, critique=critique_text)}
    ]
    return run_completion(msgs, model, verbose)

def check_termination(refined_ideas, model, verbose=False):
    msgs = [
        {"role":"system","content":TERMINATOR_SYSTEM},
        {"role":"user","content":TERMINATOR_USER.format(refined_ideas=refined_ideas)},
    ]
    return run_completion(msgs, model, verbose)

def assess_strengths_weaknesses(final_ideas, model, verbose=False):
    msgs = [
        {"role":"system","content":ASSESS_SYSTEM},
        {"role":"user","content":ASSESS_USER.format(final_ideas=final_ideas)},
    ]
    return run_completion(msgs, model, verbose)

def parse_refinement_output(refinement_output):
    c_pat = r"CRITIQUE_START\s*(.*?)\s*CRITIQUE_END"
    i_pat = r"IMPROVED_START\s*(.*?)\s*IMPROVED_END"
    c_match = re.search(c_pat, refinement_output, re.DOTALL)
    i_match = re.search(i_pat, refinement_output, re.DOTALL)

    if c_match and i_match:
        critique_section = c_match.group(1).strip()
        final_refined_ideas = i_match.group(1).strip()
    else:
        critique_section = "Could not parse critique."
        final_refined_ideas = refinement_output.strip()

    return critique_section, final_refined_ideas

###############################################################################
# CHAT UTILS (TEXT INPUT + BUTTON)
###############################################################################
def ensure_conversation_state(key):
    if key not in st.session_state.conversation_states:
        st.session_state.conversation_states[key] = []

def chat_interface(key, base_content, model, verbose=False):
    """
    A no-frills text-based chat:
    - Renders existing messages from session_state
    - Has a text_input + "Send" button
    - On "Send", we add user message + model response to conversation
    - We do NOT call st.rerun() here => UI remains stable
    """
    ensure_conversation_state(key)

    # Show conversation
    for msg in st.session_state.conversation_states[key]:
        if msg["role"] == "user":
            st.markdown(f"**User:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    # Input row
    user_msg = st.text_input("Message:", key=f"{key}_input")
    if st.button("Send", key=f"{key}_send"):
        if user_msg.strip():
            # Add user message
            st.session_state.conversation_states[key].append({"role":"user","content":user_msg.strip()})
            # Build request
            conv_msgs = [
                {
                    "role":"system",
                    "content":(
                        "You are a helpful assistant. "
                        f"Here is some reference content:\n\n{base_content}\n\n"
                        "User messages and assistant replies follow."
                    )
                }
            ]
            conv_msgs.extend(st.session_state.conversation_states[key])
            # LLM
            resp = run_completion(conv_msgs, model, verbose)
            # Add assistant
            st.session_state.conversation_states[key].append({"role":"assistant","content":resp})
            # Clear user input
            st.session_state[f"{key}_input"] = ""

def download_expander_content(label, content):
    return st.download_button(
        label="Download as markdown",
        data=content.encode("utf-8"),
        file_name=f"{label.replace(' ','_')}.md",
        mime="text/markdown",
    )

###############################################################################
# FILE UPLOAD & MODEL SELECTION
###############################################################################
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
pdf_text = ""
if pdf_file is not None:
    reader = PdfReader(pdf_file)
    for page in reader.pages:
        pdf_text += page.extract_text() + "\n"

model_list = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]
summarizer_model = st.selectbox("Summarizer Model:", model_list, index=0)
ideator_model = st.selectbox("Ideator Model:", model_list, index=0)
critic_model = st.selectbox("Critic Model:", model_list, index=0)
terminator_model = st.selectbox("Terminator Model:", model_list, index=0)

verbose = st.checkbox("Show verbose debug info")
max_iterations = st.number_input("Max Refinement Iterations (Auto Mode):", min_value=1, value=3)
enable_different_angle = st.checkbox("Attempt Different Angle if stuck? (Auto Mode)", value=False)

user_text = st.text_area("Paste your research text:", height=300)
additional_context = st.text_area("Optional additional context:", height=100)

process_mode = st.radio("Process Mode:", ["Automatic", "Manual"], index=0)
run_pipeline = st.button("Run Pipeline")

###############################################################################
# MANUAL STEP FUNCTIONS
###############################################################################
def manual_summarize():
    final_input_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()
    if not final_input_text:
        st.warning("No text to summarize.")
        return
    out = summarize_text(final_input_text, summarizer_model, verbose)
    st.session_state.pipeline_results["overview"] = out
    st.session_state.manual_step = "summarized"
    st.experimental_rerun()

def manual_ideate():
    if "overview" not in st.session_state.pipeline_results:
        st.warning("No overview found. Summarize first.")
        return
    out = ideate(st.session_state.pipeline_results["overview"], additional_context, ideator_model, verbose)
    st.session_state.pipeline_results["initial_ideas"] = out
    st.session_state.manual_step = "ideated"
    st.experimental_rerun()

def manual_critique():
    if "initial_ideas" not in st.session_state.pipeline_results:
        st.warning("No ideas to critique.")
        return
    out = critique(st.session_state.pipeline_results["initial_ideas"], critic_model, verbose)
    st.session_state.pipeline_results["critique_text"] = out
    st.session_state.manual_step = "critiqued"
    st.experimental_rerun()

def manual_re_ideate():
    if "critique_text" not in st.session_state.pipeline_results:
        st.warning("No critique to refine from.")
        return
    refinement_output = re_ideate(
        st.session_state.pipeline_results["initial_ideas"],
        st.session_state.pipeline_results["critique_text"],
        ideator_model,
        verbose
    )
    c_sec, improved = parse_refinement_output(refinement_output)
    st.session_state.pipeline_results["refinement_output"] = refinement_output
    st.session_state.pipeline_results["critique_section"] = c_sec
    st.session_state.pipeline_results["improved_ideas"] = improved
    st.session_state.manual_step = "refined"
    st.experimental_rerun()

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
    st.experimental_rerun()

def manual_assess():
    if "improved_ideas" not in st.session_state.pipeline_results:
        st.warning("No final ideas to assess.")
        return
    out = assess_strengths_weaknesses(st.session_state.pipeline_results["improved_ideas"], terminator_model, verbose)
    st.session_state.pipeline_results["assessment"] = out
    st.session_state.manual_step = "assessed"
    st.experimental_rerun()

###############################################################################
# MAIN PIPELINE LOGIC
###############################################################################
if run_pipeline:
    # If it's the first run in the session:
    if not st.session_state.pipeline_ran:
        st.session_state.pipeline_ran = True
        # For Automatic mode, we can reset pipeline results
        # For Manual mode, we do the same but start step=started
        st.session_state.pipeline_results.clear()
        st.session_state.manual_step = "started"

    if process_mode == "Automatic":
        final_input_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()
        if not final_input_text:
            st.warning("No text provided.")
        else:
            # Summarize
            overview = summarize_text(final_input_text, summarizer_model, verbose)
            # Ideate
            initial_ideas = ideate(overview, additional_context, ideator_model, verbose)

            iteration_count = 0
            final_good_enough = False
            current_ideas = initial_ideas
            iteration_data = []

            while iteration_count < max_iterations:
                critique_text = critique(current_ideas, critic_model, verbose)
                refinement_output = re_ideate(current_ideas, critique_text, ideator_model, verbose)
                c_sec, improved_ideas = parse_refinement_output(refinement_output)
                verdict = check_termination(improved_ideas, terminator_model, verbose)

                iteration_data.append((critique_text, c_sec, improved_ideas, verdict))

                if verdict.startswith("Good enough"):
                    final_good_enough = True
                    current_ideas = improved_ideas
                    break
                else:
                    current_ideas = improved_ideas
                    iteration_count += 1

            # If not good after max iterations
            if not final_good_enough:
                assess_out = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
                st.session_state.pipeline_results["assessment"] = assess_out
            else:
                st.session_state.pipeline_results["assessment"] = None

            # Store
            st.session_state.pipeline_results["overview"] = overview
            st.session_state.pipeline_results["initial_ideas"] = initial_ideas
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.session_state.pipeline_results["final_good_enough"] = final_good_enough
            st.session_state.pipeline_results["final_ideas"] = current_ideas
            st.session_state.pipeline_results["additional_context"] = additional_context

###############################################################################
# MANUAL MODE UI
###############################################################################
if process_mode == "Manual" and st.session_state.pipeline_ran:
    step = st.session_state.manual_step

    if step in ["started","not_started"]:
        st.subheader("Manual Step 1: Summarize")
        st.write("Click 'Run Summarizer' to produce the overview from your text/PDF.")
        if st.button("Run Summarizer"):
            manual_summarize()

        if "overview" in st.session_state.pipeline_results:
            with st.expander("Summarizer Output", expanded=True):
                st.write(st.session_state.pipeline_results["overview"])
                download_expander_content("overview", st.session_state.pipeline_results["overview"])
                chat_interface("manual_summarizer", st.session_state.pipeline_results["overview"], summarizer_model, verbose)

    if step == "summarized":
        st.subheader("Manual Step 2: Ideate")
        st.write("Click 'Run Ideator' to propose alignment research directions.")
        if st.button("Run Ideator"):
            manual_ideate()

        if "initial_ideas" in st.session_state.pipeline_results:
            with st.expander("Ideation Output", expanded=True):
                st.write(st.session_state.pipeline_results["initial_ideas"])
                download_expander_content("initial_ideas", st.session_state.pipeline_results["initial_ideas"])
                chat_interface("manual_ideator", st.session_state.pipeline_results["initial_ideas"], ideator_model, verbose)

    if step == "ideated":
        st.subheader("Manual Step 3: Critique")
        st.write("Click 'Run Critique' to critique the proposed directions.")
        if st.button("Run Critique"):
            manual_critique()

        if "critique_text" in st.session_state.pipeline_results:
            with st.expander("Critique Output", expanded=True):
                st.write(st.session_state.pipeline_results["critique_text"])
                download_expander_content("iteration_critique", st.session_state.pipeline_results["critique_text"])
                chat_interface("manual_critic", st.session_state.pipeline_results["critique_text"], critic_model, verbose)

    if step == "critiqued":
        st.subheader("Manual Step 4: Re-Ideate (Refine)")
        st.write("Click 'Run Re-Ideation' to produce improved directions.")
        if st.button("Run Re-Ideation"):
            manual_re_ideate()

        if "refinement_output" in st.session_state.pipeline_results:
            with st.expander("Refinement Output", expanded=True):
                st.write(st.session_state.pipeline_results["refinement_output"])
            with st.expander("Extracted CRITIQUE_START/END", expanded=False):
                st.write(st.session_state.pipeline_results["critique_section"])
            if "improved_ideas" in st.session_state.pipeline_results:
                with st.expander("Refined Directions", expanded=True):
                    st.write(st.session_state.pipeline_results["improved_ideas"])
                    chat_interface("manual_refined_ideas", st.session_state.pipeline_results["improved_ideas"], ideator_model, verbose)

    if step == "refined":
        st.subheader("Manual Step 5: Terminator Check")
        st.write("Check if these refined directions are good enough or need more iteration.")
        if st.button("Run Terminator Check"):
            manual_terminate()

        if "verdict" in st.session_state.pipeline_results:
            st.write(f"Terminator Verdict: {st.session_state.pipeline_results['verdict']}")

    if step == "approved":
        st.subheader("Final Accepted Ideas")
        final_ideas = st.session_state.pipeline_results.get("improved_ideas","No ideas found.")
        st.write(final_ideas)
        download_expander_content("final_accepted_ideas", final_ideas)
        chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
        st.success("Pipeline complete. The directions were approved by the Terminator.")

    if step == "needs_iteration":
        st.warning("Terminator says more iteration is needed. You can refine further or do a Strengths/Weaknesses assessment.")
        if st.button("Run Strengths & Weaknesses Assessment"):
            manual_assess()

    if step == "assessed":
        st.subheader("Strengths & Weaknesses Assessment")
        assessment = st.session_state.pipeline_results.get("assessment","")
        st.write(assessment)
        download_expander_content("strengths_weaknesses_assessment", assessment)
        chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
        st.info("End of pipeline in manual mode. You may refine more or conclude here.")

###############################################################################
# AUTOMATIC MODE RESULT DISPLAY
###############################################################################
if process_mode == "Automatic" and st.session_state.pipeline_ran:
    overview = st.session_state.pipeline_results.get("overview","")
    initial_ideas = st.session_state.pipeline_results.get("initial_ideas","")
    iteration_data = st.session_state.pipeline_results.get("iteration_data",[])
    final_good_enough = st.session_state.pipeline_results.get("final_good_enough",False)
    final_ideas = st.session_state.pipeline_results.get("final_ideas","")
    assessment = st.session_state.pipeline_results.get("assessment",None)

    with st.expander("Comprehensive Systematic Overview", expanded=True):
        st.write(overview)
        download_expander_content("overview", overview)
        chat_interface("overview", overview, ideator_model, verbose)

    with st.expander("Initial Alignment Ideas", expanded=False):
        st.write(initial_ideas)
        download_expander_content("initial_alignment_ideas", initial_ideas)
        chat_interface("initial_ideas", initial_ideas, ideator_model, verbose)

    for i, (critique_text, critique_section, improved_ideas, verdict) in enumerate(iteration_data, start=1):
        with st.expander(f"Iteration {i} Critique", expanded=False):
            st.write(critique_text)
            download_expander_content(f"iteration_{i}_critique", critique_text)
            chat_interface(f"iteration_{i}_critique", critique_text, ideator_model, verbose)

        with st.expander(f"Iteration {i} Internal Reasoning (Critique Section)", expanded=False):
            st.write(critique_section)
            download_expander_content(f"iteration_{i}_internal_reasoning", critique_section)
            chat_interface(f"iteration_{i}_internal_reasoning", critique_section, ideator_model, verbose)

        with st.expander(f"Iteration {i} Refined Directions", expanded=False):
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
        st.warning("Not approved after max iterations. Displaying final output + Strengths/Weaknesses.")
        with st.expander("Final Non-Approved Ideas", expanded=True):
            st.write(final_ideas)
            download_expander_content("final_non_approved_ideas", final_ideas)
            chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

        if assessment:
            with st.expander("Strengths & Weaknesses Assessment", expanded=True):
                st.write(assessment)
                download_expander_content("strengths_weaknesses_assessment", assessment)
                chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
