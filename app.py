import streamlit as st
import re
import anthropic
import io
from openai import OpenAI
from PyPDF2 import PdfReader

###############################################################################
# PAGE SETUP AND STATE
###############################################################################
st.title("Alignment Research Automator — Fixed UI Version")

st.write(
    "This app implements an AI alignment research pipeline in either Automatic or Manual mode."
    "\n\n- **Automatic**: Summarize → Ideate → Critique → Re-Ideate → Terminator → (Assess if needed)."
    "\n- **Manual**: You run each step yourself, see outputs in expanders, and can chat about them."
    "\n\nWe avoid the UI 'disappearing' by storing all data in session state and using expanders with `expanded=True`."
)

# Initialize needed session state variables
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False
if "manual_step" not in st.session_state:
    st.session_state.manual_step = "not_started"
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}
if "conversation_states" not in st.session_state:
    st.session_state.conversation_states = {}

###############################################################################
# SECRETS (ADJUST AS NEEDED)
###############################################################################
openai_client = OpenAI(api_key=st.secrets["API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

###############################################################################
# MODEL DETECTION / DISPATCH
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
            content = "INSTRUCTIONS:\n" + msg["content"]
            adjusted.append({"role": "user", "content": content})
        else:
            adjusted.append(msg)
    return adjusted

def run_completion(messages, model, verbose=False):
    """Dispatch to the chosen model endpoint."""
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
        return "Unrecognized model endpoint."

###############################################################################
# PROMPTS
###############################################################################
SUMMARIZER_SYSTEM = """You are a specialized research assistant focusing on AI alignment. Summarize text thoroughly, no vagueness."""
SUMMARIZER_USER = """Below is the research text. Provide a comprehensive, factual overview:
{user_text}
"""

IDEATOR_SYSTEM = """You are an AI alignment ideation agent."""
IDEATOR_IDEATION_USER = """Systematic Overview:
{overview}

Additional Angles:
{additional_context}
"""

CRITIC_SYSTEM = """You are a critic analyzing AI alignment directions for clarity, novelty, feasibility, etc."""
CRITIC_USER = """Directions:
{ideas}
"""

RE_IDEATE_SYSTEM = """You refine directions after critique. Use format:
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

ASSESS_SYSTEM = """You are an evaluator providing strengths & weaknesses of final directions."""
ASSESS_USER = """Final directions:
{final_ideas}
"""

###############################################################################
# PIPELINE FUNCTIONS
###############################################################################
def summarize_text(user_text, model, verbose=False):
    msgs = [
        {"role":"system", "content":SUMMARIZER_SYSTEM},
        {"role":"user", "content":SUMMARIZER_USER.format(user_text=user_text)}
    ]
    return run_completion(msgs, model, verbose)

def ideate(overview, additional_context, model, verbose=False):
    msgs = [
        {"role":"system", "content":IDEATOR_SYSTEM},
        {"role":"user", "content":IDEATOR_IDEATION_USER.format(
            overview=overview, additional_context=additional_context
        )}
    ]
    return run_completion(msgs, model, verbose)

def critique(ideas, model, verbose=False):
    msgs = [
        {"role":"system", "content":CRITIC_SYSTEM},
        {"role":"user", "content":CRITIC_USER.format(ideas=ideas)}
    ]
    return run_completion(msgs, model, verbose)

def re_ideate(ideas, critique_text, model, verbose=False):
    msgs = [
        {"role":"system", "content":RE_IDEATE_SYSTEM},
        {"role":"user", "content":RE_IDEATE_USER.format(ideas=ideas, critique=critique_text)}
    ]
    return run_completion(msgs, model, verbose)

def check_termination(refined_ideas, model, verbose=False):
    msgs = [
        {"role":"system", "content":TERMINATOR_SYSTEM},
        {"role":"user", "content":TERMINATOR_USER.format(refined_ideas=refined_ideas)}
    ]
    return run_completion(msgs, model, verbose)

def assess_strengths_weaknesses(final_ideas, model, verbose=False):
    msgs = [
        {"role":"system", "content":ASSESS_SYSTEM},
        {"role":"user", "content":ASSESS_USER.format(final_ideas=final_ideas)}
    ]
    return run_completion(msgs, model, verbose)

def parse_refinement_output(refinement_output):
    critique_pat = r"CRITIQUE_START\s*(.*?)\s*CRITIQUE_END"
    improved_pat = r"IMPROVED_START\s*(.*?)\s*IMPROVED_END"

    c_match = re.search(critique_pat, refinement_output, re.DOTALL)
    i_match = re.search(improved_pat, refinement_output, re.DOTALL)

    if c_match and i_match:
        critique_section = c_match.group(1).strip()
        final_refined_ideas = i_match.group(1).strip()
    else:
        critique_section = "Could not parse critique."
        final_refined_ideas = refinement_output.strip()

    return critique_section, final_refined_ideas

###############################################################################
# CHAT: TEXT INPUT + "SEND" BUTTON
###############################################################################
def ensure_conversation_state(key):
    if key not in st.session_state.conversation_states:
        st.session_state.conversation_states[key] = []

def chat_interface(key, base_content, model, verbose=False):
    """
    A simple text-based chat with no forced st.rerun(). 
    The conversation remains in st.session_state.conversation_states[key].
    """
    ensure_conversation_state(key)

    # Display existing messages
    for msg in st.session_state.conversation_states[key]:
        if msg["role"] == "user":
            st.markdown(f"**User:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:** {msg['content']}")

    user_msg = st.text_input("Message:", key=f"{key}_input")
    if st.button("Send", key=f"{key}_send"):
        if user_msg.strip():
            # Add user message
            st.session_state.conversation_states[key].append({"role":"user","content":user_msg.strip()})
            # Build prompt
            conv_msgs = [
                {
                    "role":"system",
                    "content":(
                        "You are a helpful assistant. "
                        f"Here is reference content:\n\n{base_content}\n\n"
                        "User messages and assistant replies follow."
                    )
                }
            ]
            conv_msgs.extend(st.session_state.conversation_states[key])
            # LLM call
            response = run_completion(conv_msgs, model, verbose)
            # Add assistant reply
            st.session_state.conversation_states[key].append({"role":"assistant","content":response})
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

model_options = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]
summarizer_model = st.selectbox("Summarizer Model:", model_options, index=0)
ideator_model = st.selectbox("Ideator Model:", model_options, index=0)
critic_model = st.selectbox("Critic Model:", model_options, index=0)
terminator_model = st.selectbox("Terminator Model:", model_options, index=0)

verbose = st.checkbox("Show verbose debug info")
max_iterations = st.number_input("Max Refinement Iterations (Auto Mode):", min_value=1, value=3)
enable_different_angle = st.checkbox("Attempt Different Angle if stuck? (Auto Mode)", value=False)

user_text = st.text_area("Paste your research text:", height=300)
additional_context = st.text_area("Optional angles/considerations:", height=100)

process_mode = st.radio("Process Mode:", ["Automatic", "Manual"], index=0)
run_pipeline = st.button("Run Pipeline")

###############################################################################
# MANUAL MODE STEP FUNCS
###############################################################################
def manual_summarize():
    final_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()
    if not final_text:
        st.warning("No text to summarize.")
        return
    out = summarize_text(final_text, summarizer_model, verbose)
    st.session_state.pipeline_results["overview"] = out
    st.session_state.manual_step = "summarized"
    st.rerun()

def manual_ideate():
    if "overview" not in st.session_state.pipeline_results:
        st.warning("No overview found. Summarize first.")
        return
    out = ideate(
        st.session_state.pipeline_results["overview"], 
        additional_context, 
        ideator_model, 
        verbose
    )
    st.session_state.pipeline_results["initial_ideas"] = out
    st.session_state.manual_step = "ideated"
    st.rerun()

def manual_critique():
    if "initial_ideas" not in st.session_state.pipeline_results:
        st.warning("No ideas to critique.")
        return
    out = critique(st.session_state.pipeline_results["initial_ideas"], critic_model, verbose)
    st.session_state.pipeline_results["critique_text"] = out
    st.session_state.manual_step = "critiqued"
    st.rerun()

def manual_re_ideate():
    if "critique_text" not in st.session_state.pipeline_results:
        st.warning("No critique to refine from.")
        return
    refine = re_ideate(
        st.session_state.pipeline_results["initial_ideas"], 
        st.session_state.pipeline_results["critique_text"],
        ideator_model,
        verbose
    )
    c_sec, improved = parse_refinement_output(refine)
    st.session_state.pipeline_results["refinement_output"] = refine
    st.session_state.pipeline_results["critique_section"] = c_sec
    st.session_state.pipeline_results["improved_ideas"] = improved
    st.session_state.manual_step = "refined"
    st.rerun()

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
    st.rerun()

def manual_assess():
    if "improved_ideas" not in st.session_state.pipeline_results:
        st.warning("No final ideas to assess.")
        return
    out = assess_strengths_weaknesses(st.session_state.pipeline_results["improved_ideas"], terminator_model, verbose)
    st.session_state.pipeline_results["assessment"] = out
    st.session_state.manual_step = "assessed"
    st.rerun()

###############################################################################
# MAIN PIPELINE LOGIC
###############################################################################
if run_pipeline:
    if not st.session_state.pipeline_ran:
        st.session_state.pipeline_ran = True
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
                crit = critique(current_ideas, critic_model, verbose)
                refine = re_ideate(current_ideas, crit, ideator_model, verbose)
                c_sec, improved = parse_refinement_output(refine)
                verdict = check_termination(improved, terminator_model, verbose)

                iteration_data.append((crit, c_sec, improved, verdict))

                if verdict.startswith("Good enough"):
                    final_good_enough = True
                    current_ideas = improved
                    break
                else:
                    current_ideas = improved
                    iteration_count += 1

            # If not good after max_iterations
            if not final_good_enough:
                assess_out = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
                st.session_state.pipeline_results["assessment"] = assess_out
            else:
                st.session_state.pipeline_results["assessment"] = None

            # Save results
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

    # STEP 1: Summarize
    if step in ["started", "not_started"]:
        st.subheader("Manual Step 1: Summarize")
        st.write("Press 'Run Summarizer' to produce an overview from the provided text/PDF.")
        if st.button("Run Summarizer"):
            manual_summarize()

        if "overview" in st.session_state.pipeline_results:
            with st.expander("Summarizer Output", expanded=True):
                st.write(st.session_state.pipeline_results["overview"])
                download_expander_content("overview", st.session_state.pipeline_results["overview"])
                # Chat about summarizer output
                chat_interface("manual_summarizer", st.session_state.pipeline_results["overview"], summarizer_model, verbose)

    # STEP 2: Ideate
    if step == "summarized":
        st.subheader("Manual Step 2: Ideate")
        st.write("Press 'Run Ideator' to propose alignment directions.")
        if st.button("Run Ideator"):
            manual_ideate()

        if "initial_ideas" in st.session_state.pipeline_results:
            with st.expander("Ideation Output", expanded=True):
                st.write(st.session_state.pipeline_results["initial_ideas"])
                download_expander_content("initial_ideas", st.session_state.pipeline_results["initial_ideas"])
                chat_interface("manual_ideator", st.session_state.pipeline_results["initial_ideas"], ideator_model, verbose)

    # STEP 3: Critique
    if step == "ideated":
        st.subheader("Manual Step 3: Critique")
        st.write("Press 'Run Critique' to critique the proposed directions.")
        if st.button("Run Critique"):
            manual_critique()

        if "critique_text" in st.session_state.pipeline_results:
            with st.expander("Critique Output", expanded=True):
                st.write(st.session_state.pipeline_results["critique_text"])
                download_expander_content("iteration_critique", st.session_state.pipeline_results["critique_text"])
                chat_interface("manual_critic", st.session_state.pipeline_results["critique_text"], critic_model, verbose)

    # STEP 4: Re-Ideate
    if step == "critiqued":
        st.subheader("Manual Step 4: Re-Ideate (Refine)")
        st.write("Press 'Run Re-Ideation' to refine the directions using the critique.")
        if st.button("Run Re-Ideation"):
            manual_re_ideate()

        if "refinement_output" in st.session_state.pipeline_results:
            with st.expander("Refinement Output", expanded=True):
                st.write(st.session_state.pipeline_results["refinement_output"])

            if "critique_section" in st.session_state.pipeline_results:
                with st.expander("Extracted CRITIQUE_START/END", expanded=True):
                    st.write(st.session_state.pipeline_results["critique_section"])

            if "improved_ideas" in st.session_state.pipeline_results:
                with st.expander("Refined Directions", expanded=True):
                    st.write(st.session_state.pipeline_results["improved_ideas"])
                    chat_interface("manual_refined_ideas", st.session_state.pipeline_results["improved_ideas"], ideator_model, verbose)

    # STEP 5: Terminator
    if step == "refined":
        st.subheader("Manual Step 5: Terminator Check")
        st.write("Check if refined directions are good enough or need iteration.")
        if st.button("Run Terminator Check"):
            manual_terminate()

        if "verdict" in st.session_state.pipeline_results:
            st.write(f"Terminator Verdict: {st.session_state.pipeline_results['verdict']}")

    # If approved
    if step == "approved":
        st.subheader("Final Accepted Ideas")
        final_ideas = st.session_state.pipeline_results.get("improved_ideas","No final ideas.")
        st.write(final_ideas)
        download_expander_content("final_accepted_ideas", final_ideas)
        chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
        st.success("Pipeline complete. Directions were approved by Terminator.")

    # If needs iteration
    if step == "needs_iteration":
        st.warning("Terminator says more iteration needed. Refine further or do Strengths/Weaknesses.")
        if st.button("Run Strengths & Weaknesses Assessment"):
            manual_assess()

    if step == "assessed":
        st.subheader("Strengths & Weaknesses Assessment")
        assessment = st.session_state.pipeline_results.get("assessment","No assessment.")
        st.write(assessment)
        download_expander_content("strengths_weaknesses_assessment", assessment)
        chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
        st.info("End of pipeline in manual mode. You can refine more or stop here.")

###############################################################################
# AUTOMATIC MODE RESULTS
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

    with st.expander("Initial Alignment Ideas", expanded=True):
        st.write(initial_ideas)
        download_expander_content("initial_alignment_ideas", initial_ideas)
        chat_interface("initial_ideas", initial_ideas, ideator_model, verbose)

    for i, (critique_text, critique_section, improved_ideas, verdict) in enumerate(iteration_data, start=1):
        with st.expander(f"Iteration {i} Critique", expanded=True):
            st.write(critique_text)
            download_expander_content(f"iteration_{i}_critique", critique_text)
            chat_interface(f"iteration_{i}_critique", critique_text, ideator_model, verbose)

        with st.expander(f"Iteration {i} Internal Reasoning (CRITIQUE_START/END)", expanded=True):
            st.write(critique_section)
            download_expander_content(f"iteration_{i}_internal_reasoning", critique_section)
            chat_interface(f"iteration_{i}_internal_reasoning", critique_section, ideator_model, verbose)

        with st.expander(f"Iteration {i} Refined Directions", expanded=True):
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
        st.warning("Not approved after max iterations. Displaying final output + Strengths/Weaknesses if present.")
        with st.expander("Final Non-Approved Ideas", expanded=True):
            st.write(final_ideas)
            download_expander_content("final_non_approved_ideas", final_ideas)
            chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

        if assessment:
            with st.expander("Strengths & Weaknesses Assessment", expanded=True):
                st.write(assessment)
                download_expander_content("strengths_weaknesses_assessment", assessment)
                chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
