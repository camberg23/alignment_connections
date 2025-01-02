import streamlit as st
import re
import anthropic
import io
from openai import OpenAI
from PyPDF2 import PdfReader

########################################
# PAGE SETUP AND STATE
########################################

st.title("Alignment Research Automator")
st.write(
    "This tool summarizes research text, proposes alignment directions, critiques them, "
    "and refines them based on termination criteria. The goal is to produce actionable, "
    "high-quality alignment research directions."
)
st.write(
    "You can choose different LLMs for each stage (Summarizer, Ideator, Critic, Terminator) "
    "and run multiple iterations. If after N iterations it's still not good, you'll see a final "
    "output plus strengths/weaknesses."
)
st.write(
    "Upload a PDF or paste text (PDF takes precedence). Within each expander, you can chat "
    "further about that step. Pipeline runs on 'Run Pipeline'. In **Manual** mode, you proceed "
    "step-by-step. In **Automatic** mode, everything runs at once."
)

# Session state initialization
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False
if "conversation_states" not in st.session_state:
    st.session_state.conversation_states = {}
if "pipeline_results" not in st.session_state:
    st.session_state.pipeline_results = {}
if "manual_step" not in st.session_state:
    st.session_state.manual_step = "not_started"

########################################
# SECRETS (ADJUST)
########################################

openai_client = OpenAI(api_key=st.secrets["API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

########################################
# MODEL UTILS
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
            adjusted.append({"role":"user", "content":content})
        else:
            adjusted.append(msg)
    return adjusted

def run_completion(messages, model, verbose=False):
    if is_o1_model(model):
        messages = adjust_messages_for_o1(messages)
        if verbose:
            st.write("Using O1 Model:", model)
            st.write("Messages:", messages)
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
        response = completion.choices[0].message.content
        if verbose:
            st.write("Response:", response)
        return response

    elif is_gpt4o_model(model):
        if verbose:
            st.write("Using GPT-4o Model:", model)
            st.write("Messages:", messages)
        completion = openai_client.chat.completions.create(
            model=model,
            messages=messages
        )
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
            "model": "claude-3-5-sonnet-20241022",  
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
        return "Model not recognized."

########################################
# PROMPTS
########################################

SUMMARIZER_SYSTEM = """You are a specialized research assistant analyzing scholarly text and identifying AI alignment implications.
Provide a thorough, factual overview:
- Domain, key hypotheses
- Methods, results, theoretical contributions
No vagueness, be comprehensive.
"""

SUMMARIZER_USER = """Below is research text. Provide a comprehensive overview as requested. 
It should be extremely substantive, covering the 'meat' of the paper:

Research Text:
{user_text}
"""

IDEATOR_SYSTEM = """You are an AI alignment research ideation agent.
Given an overview, produce alignment directions, critique them, then refine.
"""

IDEATOR_IDEATION_USER = """Use the overview plus any user angles to propose original alignment research directions.

Systematic Overview:
{overview}

User Angles:
{additional_context}
"""

CRITIC_SYSTEM = """You are a critic of alignment directions. Critique them for clarity, novelty, feasibility, actionability."""
CRITIC_USER = """Directions:
{ideas}
Critique them thoroughly.
"""

RE_IDEATE_SYSTEM = """You are an AI alignment agent refining directions after a critique.
Incorporate the critique seamlessly, do not mention it explicitly.

Format:
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

Refine them as instructed.
"""

TERMINATOR_SYSTEM = """You are an evaluator. 
If directions are good, say 'Good enough'.
If not, say 'Needs more iteration: <reason>'.
No other text.
"""

TERMINATOR_USER = """Evaluate these improved directions:

{refined_ideas}
"""

TERMINATOR_ASSESS_SYSTEM = """You are an evaluator.
Given the final directions, provide strengths & weaknesses, concise and factual.
"""

TERMINATOR_ASSESS_USER = """Final directions after all attempts:
{final_ideas}

Provide strengths & weaknesses.
"""

########################################
# PIPELINE FUNCS
########################################

def summarize_text(user_text, model, verbose=False):
    msgs = [
        {"role":"system", "content": SUMMARIZER_SYSTEM},
        {"role":"user", "content": SUMMARIZER_USER.format(user_text=user_text)}
    ]
    return run_completion(msgs, model, verbose)

def ideate(overview, additional_context, model, verbose=False):
    msgs = [
        {"role":"system", "content": IDEATOR_SYSTEM},
        {"role":"user", "content": IDEATOR_IDEATION_USER.format(
            overview=overview,
            additional_context=additional_context
        )}
    ]
    return run_completion(msgs, model, verbose)

def critique(ideas, model, verbose=False):
    msgs = [
        {"role":"system", "content": CRITIC_SYSTEM},
        {"role":"user", "content": CRITIC_USER.format(ideas=ideas)}
    ]
    return run_completion(msgs, model, verbose)

def re_ideate(ideas, critique_text, model, verbose=False):
    msgs = [
        {"role":"system", "content": RE_IDEATE_SYSTEM},
        {"role":"user", "content": RE_IDEATE_USER.format(
            ideas=ideas, 
            critique=critique_text
        )}
    ]
    return run_completion(msgs, model, verbose)

def check_termination(refined_ideas, model, verbose=False):
    msgs = [
        {"role":"system", "content": TERMINATOR_SYSTEM},
        {"role":"user", "content": TERMINATOR_USER.format(refined_ideas=refined_ideas)}
    ]
    return run_completion(msgs, model, verbose)

def assess_strengths_weaknesses(final_ideas, model, verbose=False):
    msgs = [
        {"role":"system", "content": TERMINATOR_ASSESS_SYSTEM},
        {"role":"user", "content": TERMINATOR_ASSESS_USER.format(final_ideas=final_ideas)}
    ]
    return run_completion(msgs, model, verbose)

def parse_refinement_output(refinement_output):
    critique_pattern = r"CRITIQUE_START\s*(.*?)\s*CRITIQUE_END"
    improved_pattern = r"IMPROVED_START\s*(.*?)\s*IMPROVED_END"
    c_match = re.search(critique_pattern, refinement_output, re.DOTALL)
    i_match = re.search(improved_pattern, refinement_output, re.DOTALL)
    if c_match and i_match:
        critique_section = c_match.group(1).strip()
        final_ideas = i_match.group(1).strip()
    else:
        critique_section = "Could not parse critique."
        final_ideas = refinement_output.strip()
    return critique_section, final_ideas

########################################
# CHAT WITHIN EXPANDER
########################################

def ensure_conversation_state(key):
    if key not in st.session_state.conversation_states:
        st.session_state.conversation_states[key] = []

def chat_interface(key, base_content, model, verbose=False):
    """
    Identical to original approach: st.chat_input => conversation in st.session_state => st.rerun() after sending
    We do NOT forcibly modify st.session_state["chat_input_..."] ourselves, avoiding Streamlit's “cannot be modified” error.
    """
    ensure_conversation_state(key)

    # Display any messages so far
    for msg in st.session_state.conversation_states[key]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # A chat input at the bottom
    prompt = st.chat_input("Ask a question or refine this content further", key=f"chat_input_{key}")
    if prompt:
        # Store user message
        st.session_state.conversation_states[key].append({"role": "user", "content": prompt})
        # Build conversation
        conv = [
            {
                "role":"system",
                "content":(
                    "You are a helpful assistant. "
                    f"Here is base content:\n\n{base_content}\n\n"
                    "User messages and assistant responses follow."
                )
            }
        ]
        conv.extend(st.session_state.conversation_states[key])
        # Run
        response = run_completion(conv, model, verbose)
        st.session_state.conversation_states[key].append({"role": "assistant", "content": response})
        st.rerun()

def download_expander_content(label, content):
    return st.download_button(
        label="Download contents as markdown",
        data=content.encode("utf-8"),
        file_name=f"{label.replace(' ','_')}.md",
        mime="text/markdown"
    )

########################################
# PDF UPLOAD & MODELS
########################################

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

user_text = st.text_area("Paste your research text (abstract/section):", height=300)
additional_context = st.text_area("Optional additional angles or considerations:", height=100)

process_mode = st.radio("Process Mode:", ["Automatic", "Manual"], index=0)
run_pipeline = st.button("Run Pipeline")

########################################
# MANUAL MODE STEP FUNCTIONS
########################################

def manual_summarize():
    final_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()
    if not final_text:
        st.warning("No text to summarize.")
        return
    out = summarize_text(final_text, summarizer_model, verbose)
    st.session_state.pipeline_results["overview"] = out
    st.session_state.manual_step = "summarized"

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

def manual_critique():
    if "initial_ideas" not in st.session_state.pipeline_results:
        st.warning("No initial ideas to critique.")
        return
    out = critique(st.session_state.pipeline_results["initial_ideas"], critic_model, verbose)
    st.session_state.pipeline_results["critique_text"] = out
    st.session_state.manual_step = "critiqued"

def manual_re_ideate():
    if "critique_text" not in st.session_state.pipeline_results:
        st.warning("No critique to refine from.")
        return
    refinement = re_ideate(
        st.session_state.pipeline_results["initial_ideas"],
        st.session_state.pipeline_results["critique_text"],
        ideator_model,
        verbose
    )
    csec, improved = parse_refinement_output(refinement)
    st.session_state.pipeline_results["refinement_output"] = refinement
    st.session_state.pipeline_results["critique_section"] = csec
    st.session_state.pipeline_results["improved_ideas"] = improved
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
    if "improved_ideas" not in st.session_state.pipeline_results:
        st.warning("No final ideas to assess.")
        return
    assess_out = assess_strengths_weaknesses(
        st.session_state.pipeline_results["improved_ideas"], 
        terminator_model, 
        verbose
    )
    st.session_state.pipeline_results["assessment"] = assess_out
    st.session_state.manual_step = "assessed"

########################################
# MAIN PIPELINE LOGIC
########################################

if run_pipeline:
    # If pipeline hasn't run yet in this session:
    if not st.session_state.pipeline_ran:
        st.session_state.pipeline_ran = True
        
        # If Automatic mode, we can safely clear out old results/conversations
        # If Manual mode, we do NOT wipe conversation states so we can chat from a clean slate
        if process_mode == "Automatic":
            st.session_state.conversation_states.clear()
            st.session_state.pipeline_results.clear()
        else:
            # Just ensure pipeline_results is empty for a new run, but keep conversation
            st.session_state.pipeline_results.clear()

    if process_mode == "Automatic":
        final_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()
        if not final_text:
            st.warning("No text provided.")
        else:
            # Summarize
            overview = summarize_text(final_text, summarizer_model, verbose)
            # Ideate
            initial_ideas = ideate(overview, additional_context, ideator_model, verbose)

            iteration_data = []
            iteration_count = 0
            final_good_enough = False
            current_ideas = initial_ideas

            while iteration_count < max_iterations:
                crit = critique(current_ideas, critic_model, verbose)
                refinement = re_ideate(current_ideas, crit, ideator_model, verbose)
                csec, improved = parse_refinement_output(refinement)
                verdict = check_termination(improved, terminator_model, verbose)
                iteration_data.append((crit, csec, improved, verdict))

                if verdict.startswith("Good enough"):
                    final_good_enough = True
                    current_ideas = improved
                    break
                else:
                    current_ideas = improved
                    iteration_count += 1

            # If not good after all iterations
            if not final_good_enough:
                sw = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
                st.session_state.pipeline_results["assessment"] = sw
            else:
                st.session_state.pipeline_results["assessment"] = None

            # Save pipeline outputs
            st.session_state.pipeline_results["overview"] = overview
            st.session_state.pipeline_results["initial_ideas"] = initial_ideas
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.session_state.pipeline_results["final_good_enough"] = final_good_enough
            st.session_state.pipeline_results["final_ideas"] = current_ideas
            st.session_state.pipeline_results["additional_context"] = additional_context

    else:
        # Manual mode: just set step=started if not already
        if st.session_state.manual_step == "not_started":
            st.session_state.manual_step = "started"

########################################
# MANUAL MODE UI
########################################

if process_mode == "Manual" and st.session_state.pipeline_ran:
    # Step 1: Summarize
    if st.session_state.manual_step in ["started", "not_started"]:
        st.subheader("Manual Step 1: Summarize")
        if st.button("Run Summarizer"):
            manual_summarize()

        if "overview" in st.session_state.pipeline_results:
            with st.expander("Summarizer Output"):
                st.write(st.session_state.pipeline_results["overview"])
                download_expander_content("overview", st.session_state.pipeline_results["overview"])
                chat_interface(
                    key="manual_summarizer",
                    base_content=st.session_state.pipeline_results["overview"],
                    model=summarizer_model,
                    verbose=verbose
                )

        if st.button("Next Step"):
            if "overview" not in st.session_state.pipeline_results:
                st.warning("No summary found. Run Summarizer first.")
            else:
                st.session_state.manual_step = "summarized"

    # Step 2: Ideate
    if st.session_state.manual_step == "summarized":
        st.subheader("Manual Step 2: Ideate")
        if st.button("Run Ideator"):
            manual_ideate()

        if "initial_ideas" in st.session_state.pipeline_results:
            with st.expander("Ideation Output"):
                st.write(st.session_state.pipeline_results["initial_ideas"])
                download_expander_content("initial_alignment_ideas", st.session_state.pipeline_results["initial_ideas"])
                chat_interface(
                    key="manual_ideator",
                    base_content=st.session_state.pipeline_results["initial_ideas"],
                    model=ideator_model,
                    verbose=verbose
                )

        if st.button("Next Step"):
            if "initial_ideas" not in st.session_state.pipeline_results:
                st.warning("No ideas found.")
            else:
                st.session_state.manual_step = "ideated"

    # Step 3: Critique
    if st.session_state.manual_step == "ideated":
        st.subheader("Manual Step 3: Critique")
        if st.button("Run Critique"):
            manual_critique()

        if "critique_text" in st.session_state.pipeline_results:
            with st.expander("Critique Output"):
                st.write(st.session_state.pipeline_results["critique_text"])
                download_expander_content("iteration_critique", st.session_state.pipeline_results["critique_text"])
                chat_interface(
                    key="manual_critic",
                    base_content=st.session_state.pipeline_results["critique_text"],
                    model=critic_model,
                    verbose=verbose
                )

        if st.button("Next Step"):
            if "critique_text" not in st.session_state.pipeline_results:
                st.warning("No critique found.")
            else:
                st.session_state.manual_step = "critiqued"

    # Step 4: Re-Ideate
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
                    chat_interface(
                        key="manual_refined_ideas",
                        base_content=st.session_state.pipeline_results["improved_ideas"],
                        model=ideator_model,
                        verbose=verbose
                    )

        if st.button("Next Step"):
            if "improved_ideas" not in st.session_state.pipeline_results:
                st.warning("No refined ideas found.")
            else:
                st.session_state.manual_step = "refined"

    # Step 5: Terminator
    if st.session_state.manual_step == "refined":
        st.subheader("Manual Step 5: Terminator Check")
        if st.button("Run Terminator Check"):
            manual_terminate()

        if "verdict" in st.session_state.pipeline_results:
            st.write(f"Termination Check: {st.session_state.pipeline_results['verdict']}")

        if st.button("Next Step"):
            if "verdict" not in st.session_state.pipeline_results:
                st.warning("No verdict found.")
            else:
                if st.session_state.pipeline_results["verdict"].startswith("Good enough"):
                    st.session_state.manual_step = "approved"
                else:
                    st.session_state.manual_step = "needs_iteration"

    if st.session_state.manual_step == "approved":
        st.subheader("Final Accepted Ideas")
        final_ideas = st.session_state.pipeline_results.get("improved_ideas", "")
        st.write(final_ideas)
        download_expander_content("final_accepted_ideas", final_ideas)
        chat_interface(
            key="final_accepted_ideas",
            base_content=final_ideas,
            model=ideator_model,
            verbose=verbose
        )
        st.success("Pipeline complete! Directions approved by Terminator.")

    if st.session_state.manual_step == "needs_iteration":
        st.warning("Terminator said more iteration is needed. Refine further or do a Strengths/Weaknesses assessment.")
        if st.button("Run Strengths & Weaknesses Assessment"):
            manual_assess()

    if st.session_state.manual_step == "assessed":
        st.subheader("Strengths & Weaknesses Assessment")
        assesstxt = st.session_state.pipeline_results.get("assessment", "")
        st.write(assesstxt)
        download_expander_content("strengths_weaknesses_assessment", assesstxt)
        chat_interface(
            key="strengths_weaknesses_assessment",
            base_content=assesstxt,
            model=ideator_model,
            verbose=verbose
        )
        st.info("End of pipeline in manual mode. You may refine or finish here.")

########################################
# AUTOMATIC MODE RESULTS
########################################

if process_mode == "Automatic" and st.session_state.pipeline_ran:
    overview = st.session_state.pipeline_results.get("overview", "")
    initial_ideas = st.session_state.pipeline_results.get("initial_ideas", "")
    iteration_data = st.session_state.pipeline_results.get("iteration_data", [])
    final_good_enough = st.session_state.pipeline_results.get("final_good_enough", False)
    final_ideas = st.session_state.pipeline_results.get("final_ideas", "")
    assessment = st.session_state.pipeline_results.get("assessment", None)

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

        with st.expander(f"Iteration {i} Internal Reasoning"):
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
        st.warning("Not approved after max iterations. Final output + strengths/weaknesses shown.")
        with st.expander("Final Non-Approved Ideas"):
            st.write(final_ideas)
            download_expander_content("final_non_approved_ideas", final_ideas)
            chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

        if assessment:
            with st.expander("Strengths & Weaknesses Assessment", expanded=True):
                st.write(assessment)
                download_expander_content("strengths_weaknesses_assessment", assessment)
                chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
