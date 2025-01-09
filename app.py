import streamlit as st
import re
import anthropic
import io
from openai import OpenAI
from PyPDF2 import PdfReader

# Initialize instructions
st.title("Alignment Research Ideator")
st.write("This tool takes a research text, summarizes it, proposes alignment research directions, critiques them, and refines them based on termination criteria. The goal is to produce actionable, high-quality alignment research directions.")
# st.write("You can choose different LLMs for each stage (Summarizer, Ideator, Critic, Terminator) and run multiple iterations. If after N iterations it still isn't good enough, final output and a strengths/weaknesses assessment is displayed.")
# st.write("You can upload a PDF or paste text. If both are provided, PDF content takes precedence.")
# st.write("Within each expander, you can chat with an LLM to further explore the displayed content. You can also download the contents of the expander as markdown.")
# st.write("**New**: Choose 'Manual Mode' to manually intervene at each step in the pipeline. You can go back and forth with each step before proceeding. When you press 'Proceed', the next step will automatically run, incorporating the entire conversation from the current step into the next step's prompt. Previous-step conversations remain available in collapsible sections with download buttons.")

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
if "final_input_text" not in st.session_state:
    st.session_state.final_input_text = ""

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
# CHAT AND UTILS
########################################

def ensure_conversation_state(key):
    if key not in st.session_state.conversation_states:
        st.session_state.conversation_states[key] = []

def chat_interface(key, base_content, model, verbose):
    """
    Displays a chat interface for the given 'key' with the base_content.
    The entire conversation is stored in st.session_state.conversation_states[key].
    """
    ensure_conversation_state(key)
    # Show existing chat messages
    for msg in st.session_state.conversation_states[key]:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # New user input
    prompt = st.chat_input("Ask a question or refine this content further", key=f"chat_input_{key}")
    if prompt:
        st.session_state.conversation_states[key].append({"role":"user","content":prompt})
        # We'll pass the entire conversation to the model as context, plus the base_content
        # so it knows what it's referencing
        conv_messages = [
            {
                "role":"system",
                "content":"You are a helpful assistant. You have the following content:\n\n" 
                          + base_content 
                          + "\n\nUser and assistant messages follow. Answer user queries helpfully."
            }
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

def get_conversation_transcript(key):
    """Combine all user/assistant messages into a single text block."""
    ensure_conversation_state(key)
    lines = []
    for m in st.session_state.conversation_states[key]:
        role = m["role"].upper()
        text = m["content"]
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

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
# Helper: run the Summarizer step
########################################

def do_summarization():
    # Combine the raw text with any conversation from a "pre_summarizer" key if needed
    return summarize_text(
        st.session_state.final_input_text,
        summarizer_model,
        verbose
    )

########################################
# Manual Mode Step-by-Step Execution
########################################

def run_critique_step(current_ideas):
    return critique(current_ideas, critic_model, verbose)

def run_re_ideate_step(current_ideas, critique_text):
    return re_ideate(current_ideas, critique_text, ideator_model, verbose)

def run_termination_check(refined_ideas):
    return check_termination(refined_ideas, terminator_model, verbose)

def run_strengths_weaknesses_assessment(current_ideas):
    return assess_strengths_weaknesses(current_ideas, terminator_model, verbose)

def proceed_from_summarization():
    summary = st.session_state.pipeline_results["overview"]
    conversation_text = get_conversation_transcript("overview")  # entire chat with summary
    combined_text = summary + "\n\n----\nConversation:\n" + conversation_text
    output = ideate(combined_text, additional_context, ideator_model, verbose)
    st.session_state.pipeline_results["initial_ideas"] = output
    if "iteration_data" not in st.session_state.pipeline_results:
        st.session_state.pipeline_results["iteration_data"] = []
    if len(st.session_state.pipeline_results["iteration_data"]) == 0:
        st.session_state.pipeline_results["iteration_data"].append(("", "", output, ""))
    st.session_state.manual_current_step = 1
    st.rerun()

def proceed_from_ideation():
    iteration_data = st.session_state.pipeline_results["iteration_data"]
    current_ideas = iteration_data[-1][2]
    conversation_text = get_conversation_transcript("initial_ideas")
    combined_text = current_ideas + "\n\n----\nConversation:\n" + conversation_text
    new_critique = run_critique_step(combined_text)
    iteration_data[-1] = (new_critique, "", current_ideas, "")
    st.session_state.pipeline_results["iteration_data"] = iteration_data
    st.session_state.manual_current_step = 2
    st.rerun()

def proceed_from_critique():
    iteration_data = st.session_state.pipeline_results["iteration_data"]
    critique_key = f"iteration_{len(iteration_data)}_critique"
    old_critique = iteration_data[-1][0]
    conversation_text = get_conversation_transcript(critique_key)
    combined_critique = old_critique + "\n\n----\nConversation:\n" + conversation_text
    current_ideas = iteration_data[-1][2]
    refinement_out = run_re_ideate_step(current_ideas, combined_critique)
    critique_section, improved_ideas = parse_refinement_output(refinement_out)
    iteration_data[-1] = (old_critique, critique_section, improved_ideas, "")
    st.session_state.pipeline_results["iteration_data"] = iteration_data
    st.session_state.manual_current_step = 3
    st.rerun()

def proceed_from_reideate():
    iteration_data = st.session_state.pipeline_results["iteration_data"]
    idx = len(iteration_data)
    critique_section_key = f"iteration_{idx}_internal_reasoning"
    refined_dir_key = f"iteration_{idx}_refined_directions"

    improved_ideas = iteration_data[-1][2]
    critique_section = iteration_data[-1][1]

    conv_critique_section = get_conversation_transcript(critique_section_key)
    conv_refined = get_conversation_transcript(refined_dir_key)
    combined_for_termination = (
        improved_ideas
        + "\n\n----\nCRITIQUE_SECTION_CONV:\n"
        + critique_section
        + "\n\n----\nCRITIQUE_SECTION_CHAT:\n"
        + conv_critique_section
        + "\n\n----\nREFINED_DIR_CHAT:\n"
        + conv_refined
    )
    new_verdict = run_termination_check(combined_for_termination)
    iteration_data[-1] = (
        iteration_data[-1][0],
        iteration_data[-1][1],
        iteration_data[-1][2],
        new_verdict,
    )
    st.session_state.pipeline_results["iteration_data"] = iteration_data
    st.session_state.manual_current_step = 4
    st.rerun()

def proceed_after_termination():
    iteration_data = st.session_state.pipeline_results["iteration_data"]
    verdict = iteration_data[-1][3]
    final_ideas = iteration_data[-1][2]

    if verdict.startswith("Good enough"):
        st.session_state.pipeline_results["final_good_enough"] = True
        st.session_state.pipeline_results["final_ideas"] = final_ideas
        st.session_state.manual_current_step = 999
        st.rerun()
    else:
        if len(iteration_data) < max_iterations:
            iteration_data.append(("", "", final_ideas, ""))
            st.session_state.pipeline_results["iteration_data"] = iteration_data
            st.session_state.manual_current_step = 2
            st.rerun()
        else:
            st.session_state.pipeline_results["final_good_enough"] = False
            st.session_state.pipeline_results["final_ideas"] = final_ideas
            st.session_state.manual_current_step = 999
            st.rerun()

########################################
# Main Execution Logic
########################################

if run_pipeline:
    st.session_state.conversation_states.clear()
    st.session_state.pipeline_results.clear()
    st.session_state.pipeline_ran = False
    st.session_state.manual_current_step = 0
    st.session_state.final_input_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()

    if not st.session_state.final_input_text:
        st.warning("Please provide text or upload a PDF before running the pipeline.")
    else:
        if not manual_mode:
            # AUTOMATIC PIPELINE
            overview = summarize_text(st.session_state.final_input_text, summarizer_model, verbose)

            # >>> MINIMAL CHANGE: show the overview expander immediately <<<
            with st.expander("Comprehensive Systematic Overview", expanded=False):
                st.write(overview)
                download_expander_content("overview_immediate", overview)
                chat_interface("overview_immediate", overview, ideator_model, verbose)
            # >>> END OF MINIMAL CHANGE <<<

            initial_ideas = ideate(overview, additional_context, ideator_model, verbose)

            # >>> MINIMAL CHANGE: show the initial ideas expander immediately <<<
            with st.expander("Initial Alignment Ideas", expanded=False):
                st.write(initial_ideas)
                download_expander_content("initial_alignment_ideas_immediate", initial_ideas)
                chat_interface("initial_alignment_ideas_immediate", initial_ideas, ideator_model, verbose)
            # >>> END OF MINIMAL CHANGE <<<

            iteration_count = 0
            final_good_enough = False
            current_ideas = initial_ideas

            iteration_data = []

            while iteration_count < max_iterations:
                critique_text = critique(current_ideas, critic_model, verbose)

                # >>> MINIMAL CHANGE: show the critique text immediately <<<
                with st.expander(f"Iteration {iteration_count+1} Critique", expanded=False):
                    st.write(critique_text)
                    download_expander_content(f"iteration_{iteration_count+1}_critique_immediate", critique_text)
                    chat_interface(f"iteration_{iteration_count+1}_critique_immediate", critique_text, ideator_model, verbose)
                # >>> END OF MINIMAL CHANGE <<<

                refinement_output = re_ideate(current_ideas, critique_text, ideator_model, verbose)
                critique_section, improved_ideas = parse_refinement_output(refinement_output)

                # >>> MINIMAL CHANGE: show the refined directions immediately <<<
                with st.expander(f"Iteration {iteration_count+1} Refined Directions", expanded=False):
                    st.write(improved_ideas)
                    download_expander_content(f"iteration_{iteration_count+1}_refined_directions_immediate", improved_ideas)
                    chat_interface(f"iteration_{iteration_count+1}_refined_directions_immediate", improved_ideas, ideator_model, verbose)
                # >>> END OF MINIMAL CHANGE <<<

                verdict = check_termination(improved_ideas, terminator_model, verbose)

                # >>> MINIMAL CHANGE: show the termination check result immediately <<<
                st.write(f"**Termination Check (Iteration {iteration_count+1}):** {verdict}")
                # >>> END OF MINIMAL CHANGE <<<

                iteration_data.append((critique_text, critique_section, improved_ideas, verdict))

                if verdict.startswith("Good enough"):
                    final_good_enough = True
                    current_ideas = improved_ideas
                    break
                else:
                    current_ideas = improved_ideas
                    iteration_count += 1

            if not final_good_enough:
                assessment = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
                st.session_state.pipeline_results["assessment"] = assessment
            else:
                st.session_state.pipeline_results["assessment"] = None

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
            st.session_state.pipeline_ran = True
            st.session_state.manual_current_step = 0
            st.session_state.pipeline_results["iteration_data"] = []
            st.rerun()

########################################
# MANUAL MODE UI RENDERING
########################################

if st.session_state.pipeline_ran and st.session_state.manual_mode and st.session_state.final_input_text:
    iteration_data = st.session_state.pipeline_results.get("iteration_data", [])
    current_step = st.session_state.manual_current_step

    # 0: Summarization
    # 1: Ideation
    # 2: Critique
    # 3: Re-Ideate
    # 4: Termination
    # 999: Done

    if current_step == 0:
        st.subheader("Step 1: Summarization")
        if "overview" not in st.session_state.pipeline_results:
            st.info("Click 'Run Summarization' to generate the summary. Then you can chat, rerun, or proceed to Ideation.")
        else:
            with st.expander("Summarization Output", expanded=False):
                st.write(st.session_state.pipeline_results["overview"])
                download_expander_content("overview", st.session_state.pipeline_results["overview"])

            with st.expander("Summarization Chat (Closed by default)", expanded=False):
                chat_interface("overview", st.session_state.pipeline_results["overview"], summarizer_model, verbose)

        if st.button("Run Summarization"):
            st.session_state.pipeline_results["overview"] = do_summarization()
            st.rerun()

        if "overview" in st.session_state.pipeline_results:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Rerun Summarization"):
                    st.session_state.pipeline_results["overview"] = do_summarization()
                    st.rerun()
            with col2:
                if st.button("Proceed to Ideation"):
                    proceed_from_summarization()

    elif current_step == 1:
        st.subheader("Step 2: Ideation")
        if "initial_ideas" not in st.session_state.pipeline_results:
            st.info("We do not have initial ideas yet. Possibly something went wrong.")
        else:
            with st.expander("Initial Alignment Ideas", expanded=False):
                st.write(st.session_state.pipeline_results["initial_ideas"])
                download_expander_content("initial_alignment_ideas", st.session_state.pipeline_results["initial_ideas"])

            with st.expander("Ideation Chat (Closed by default)", expanded=False):
                chat_interface("initial_ideas", st.session_state.pipeline_results["initial_ideas"], ideator_model, verbose)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rerun Ideation"):
                overview_plus_chat = (
                    st.session_state.pipeline_results["overview"]
                    + "\n\n----\nConversation:\n"
                    + get_conversation_transcript("initial_ideas")
                )
                new_ideas = ideate(overview_plus_chat, additional_context, ideator_model, verbose)
                st.session_state.pipeline_results["initial_ideas"] = new_ideas
                if len(iteration_data) == 0:
                    iteration_data.append(("", "", new_ideas, ""))
                else:
                    iteration_data[-1] = (iteration_data[-1][0], iteration_data[-1][1], new_ideas, iteration_data[-1][3])
                st.session_state.pipeline_results["iteration_data"] = iteration_data
                st.rerun()
        with col2:
            if st.button("Proceed to Critique"):
                proceed_from_ideation()

    elif current_step == 2:
        st.subheader("Step 3: Critique")
        if not iteration_data or not iteration_data[-1][0]:
            st.info("No critique has been run or stored yet. Possibly click 'Rerun Critique'.")
        else:
            with st.expander(f"Iteration {len(iteration_data)} Critique", expanded=False):
                st.write(iteration_data[-1][0])
                download_expander_content(f"iteration_{len(iteration_data)}_critique", iteration_data[-1][0])

            with st.expander("Critique Chat (Closed by default)", expanded=False):
                chat_interface(f"iteration_{len(iteration_data)}_critique", iteration_data[-1][0], critic_model, verbose)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rerun Critique"):
                current_ideas = iteration_data[-1][2]
                combined_ideas = (
                    current_ideas
                    + "\n\n----\nConversation:\n"
                    + get_conversation_transcript(f"iteration_{len(iteration_data)}_critique")
                )
                new_critique = run_critique_step(combined_ideas)
                iteration_data[-1] = (new_critique, iteration_data[-1][1], current_ideas, iteration_data[-1][3])
                st.session_state.pipeline_results["iteration_data"] = iteration_data
                st.rerun()
        with col2:
            if st.button("Proceed to Re-Ideate"):
                proceed_from_critique()

    elif current_step == 3:
        st.subheader("Step 4: Re-Ideate")
        idx = len(iteration_data)
        critique_section = iteration_data[-1][1]
        improved_ideas = iteration_data[-1][2]

        if not critique_section and not improved_ideas:
            st.info("No re-ideation has been run or stored yet. Possibly click 'Rerun Re-Ideate'.")
        else:
            # with st.expander(f"Iteration {idx} Internal Reasoning (CRITIQUE_START/END)", expanded=False):
            #     st.write(critique_section)
            #     download_expander_content(f"iteration_{idx}_internal_reasoning", critique_section)

            with st.expander(f"Iteration {idx} Refined Directions", expanded=False):
                st.write(improved_ideas)
                download_expander_content(f"iteration_{idx}_refined_directions", improved_ideas)

            with st.expander("Re-Ideation Chats (Closed by default)", expanded=False):
                chat_interface(f"iteration_{idx}_internal_reasoning", critique_section, ideator_model, verbose)
                st.write("---")
                chat_interface(f"iteration_{idx}_refined_directions", improved_ideas, ideator_model, verbose)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rerun Re-Ideate"):
                old_critique = iteration_data[-1][0]
                old_crit_section = iteration_data[-1][1]
                old_improved = iteration_data[-1][2]
                conv_1 = get_conversation_transcript(f"iteration_{idx}_internal_reasoning")
                conv_2 = get_conversation_transcript(f"iteration_{idx}_refined_directions")
                combined_for_reideate = (
                    old_improved
                    + "\n\n----\n(Existing critique text):\n"
                    + old_critique
                    + "\n\n----\nCRITIQUE_SECTION:\n"
                    + old_crit_section
                    + "\n\n----\nCONV_INTERNAL:\n"
                    + conv_1
                    + "\n\n----\nCONV_REFINED:\n"
                    + conv_2
                )
                refinement_out = run_re_ideate_step(old_improved, combined_for_reideate)
                new_critique_section, new_improved_ideas = parse_refinement_output(refinement_out)
                iteration_data[-1] = (old_critique, new_critique_section, new_improved_ideas, iteration_data[-1][3])
                st.session_state.pipeline_results["iteration_data"] = iteration_data
                st.rerun()
        with col2:
            if st.button("Proceed to Termination Check"):
                proceed_from_reideate()

    elif current_step == 4:
        st.subheader("Step 5: Termination Check")
        idx = len(iteration_data)
        verdict = iteration_data[-1][3]
        if not verdict:
            st.info("No verdict yet. Possibly click 'Rerun Termination Check'.")
        else:
            st.write(f"Termination Check: {verdict}")

        with st.expander("Termination Check Chat (Closed by default)", expanded=False):
            chat_interface(f"iteration_{idx}_termination", iteration_data[-1][2], terminator_model, verbose)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rerun Termination Check"):
                improved_ideas = iteration_data[-1][2]
                conversation_text = get_conversation_transcript(f"iteration_{idx}_termination")
                combined_for_termination = (
                    improved_ideas
                    + "\n\n----\nTermination Chat:\n"
                    + conversation_text
                )
                new_verdict = run_termination_check(combined_for_termination)
                iteration_data[-1] = (
                    iteration_data[-1][0],
                    iteration_data[-1][1],
                    iteration_data[-1][2],
                    new_verdict,
                )
                st.session_state.pipeline_results["iteration_data"] = iteration_data
                st.rerun()
        with col2:
            if st.button("Proceed (Finalize or More Iteration)"):
                proceed_after_termination()

    elif current_step == 999:
        st.subheader("Results")
        final_good_enough = st.session_state.pipeline_results.get("final_good_enough", False)
        final_ideas = st.session_state.pipeline_results.get("final_ideas", "")
        iteration_data = st.session_state.pipeline_results["iteration_data"]

        if final_good_enough:
            with st.expander("Final Accepted Ideas", expanded=True):
                st.write(final_ideas)
                download_expander_content("final_accepted_ideas", final_ideas)
                chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
        else:
            st.warning("Not approved after max iterations. Displaying final output and assessing strengths & weaknesses.")
            with st.expander("Final Non-Approved Ideas", expanded=False):
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

########################################
# Automatic Mode Results Display
########################################

if st.session_state.pipeline_ran and not st.session_state.manual_mode:
    overview = st.session_state.pipeline_results["overview"]
    initial_ideas = st.session_state.pipeline_results["initial_ideas"]
    iteration_data = st.session_state.pipeline_results["iteration_data"]
    final_good_enough = st.session_state.pipeline_results["final_good_enough"]
    final_ideas = st.session_state.pipeline_results["final_ideas"]
    assessment = st.session_state.pipeline_results["assessment"]

    with st.expander("Comprehensive Systematic Overview", expanded=False):
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

        with st.expander(f"Iteration {i} Internal Reasoning (CRITIQUE_START/END)", expanded=False):
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
        st.warning("Not approved after max iterations. Displaying final output and assessing strengths & weaknesses.")
        with st.expander("Final Non-Approved Ideas", expanded=False):
            st.write(final_ideas)
            download_expander_content("final_non_approved_ideas", final_ideas)
            chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

        if assessment is not None:
            with st.expander("Strengths & Weaknesses Assessment", expanded=True):
                st.write(assessment)
                download_expander_content("strengths_weaknesses_assessment", assessment)
                chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)


# import streamlit as st
# import re
# import anthropic
# import io
# from openai import OpenAI
# from PyPDF2 import PdfReader

# # Initialize instructions
# st.title("Alignment Research Automator")
# st.write("This tool takes a research text, summarizes it, proposes alignment research directions, critiques them, and refines them based on termination criteria. The goal is to produce actionable, high-quality alignment research directions.")
# st.write("You can choose different LLMs for each stage (Summarizer, Ideator, Critic, Terminator) and run multiple iterations. If after N iterations it still isn't good enough, final output and a strengths/weaknesses assessment is displayed.")
# st.write("You can upload a PDF or paste text. If both are provided, PDF content takes precedence.")
# st.write("Within each expander, you can chat with an LLM to further explore the displayed content. You can also download the contents of the expander as markdown.")
# st.write("**New**: Choose 'Manual Mode' to manually intervene at each step in the pipeline. You can go back and forth with each step before proceeding. When you press 'Proceed', the next step will automatically run, incorporating the entire conversation from the current step into the next step's prompt. Previous-step conversations remain available in collapsible sections with download buttons.")

# # Ensure session state keys exist
# if "pipeline_ran" not in st.session_state:
#     st.session_state.pipeline_ran = False
# if "conversation_states" not in st.session_state:
#     st.session_state.conversation_states = {}  # key -> list of messages
# if "pipeline_results" not in st.session_state:
#     st.session_state.pipeline_results = {}
# if "manual_current_step" not in st.session_state:
#     st.session_state.manual_current_step = 0
# if "manual_mode" not in st.session_state:
#     st.session_state.manual_mode = False
# if "final_input_text" not in st.session_state:
#     st.session_state.final_input_text = ""

# # Secrets
# openai_client = OpenAI(api_key=st.secrets['API_KEY'])
# anthropic_client = anthropic.Anthropic(api_key=st.secrets['ANTHROPIC_API_KEY'])

# ########################################
# # MODEL HANDLING
# ########################################

# def is_o1_model(m):
#     return m.startswith("o1-")

# def is_gpt4o_model(m):
#     return m == "gpt-4o"

# def is_anthropic_model(m):
#     return m.startswith("anthropic")

# def adjust_messages_for_o1(messages):
#     adjusted = []
#     for msg in messages:
#         if msg["role"] == "system":
#             # Convert to user
#             content = "INSTRUCTIONS:\n" + msg["content"]
#             adjusted.append({"role": "user", "content": content})
#         else:
#             adjusted.append(msg)
#     return adjusted

# def run_completion(messages, model, verbose=False):
#     if is_o1_model(model):
#         messages = adjust_messages_for_o1(messages)
#         if verbose:
#             st.write("**Using O1 Model:**", model)
#             st.write("**Messages:**", messages)
#         completion = openai_client.chat.completions.create(
#             model=model,
#             messages=messages
#         )
#         response = completion.choices[0].message.content
#         if verbose:
#             st.write("**Response:**", response)
#         return response
#     elif is_gpt4o_model(model):
#         if verbose:
#             st.write("**Using GPT-4o Model:**", model)
#             st.write("**Messages:**", messages)
#         completion = openai_client.chat.completions.create(
#             model=model,
#             messages=messages
#         )
#         response = completion.choices[0].message.content
#         if verbose:
#             st.write("**Response:**", response)
#         return response
#     elif is_anthropic_model(model):
#         if verbose:
#             st.write("**Using Anthropic Model:**", model)
#             st.write("**Messages:**", messages)
#         system_str_list = [m["content"] for m in messages if m["role"] == "system"]
#         user_assistant_msgs = [m for m in messages if m["role"] != "system"]
#         system_str = "\n".join(system_str_list) if system_str_list else None

#         kwargs = {
#             "model": "claude-3-5-sonnet-20241022",  # Example placeholder model name
#             "max_tokens": 1000,
#             "messages": user_assistant_msgs
#         }
#         if system_str:
#             kwargs["system"] = system_str

#         response = anthropic_client.messages.create(**kwargs)
#         response = response.content[0].text
#         if verbose:
#             st.write("**Response:**", response)
#         return response.strip()

# ########################################
# # PROMPTS
# ########################################

# SUMMARIZER_SYSTEM = """You are a specialized research assistant focused on analyzing scholarly research text and identifying AI alignment implications.
# Provide a comprehensive, systematic overview:
# - Domain, key hypotheses.
# - Methods, results, theoretical contributions.
# Be factual, thorough, and substantive no vague language.
# """

# SUMMARIZER_USER = """Below is a research text. Provide the comprehensive overview as requested above. It should be extremely substantive and should serve as a functional stand-in for someone reading the paper. Leave out none of the 'meat.'

# Research Text:
# {user_text}
# """

# IDEATOR_SYSTEM = """You are an AI alignment research ideation agent.
# Steps:
# 1. Given the overview, propose original alignment research directions.
# 2. Then we will critique them (no fix at that step).
# 3. Then you will produce updated research directions incorporating the critique seamlessly so the final output is standalone and improved.
# """

# IDEATOR_IDEATION_USER = """Use the overview and the additional angles below to propose several original AI alignment research directions.

# Systematic Overview:
# {overview}

# Additional angles to consider optionally inputted by the user:
# {additional_context}
# """

# CRITIC_SYSTEM = """You are a critic focused on assessing alignment research directions for clarity, novelty, feasibility. Do not fix them, just critique:
# - Identify weaknesses, if any
# - Suggest what is missing, if anything
# - Indicate what could be improved, if anything

# Just critique, do not provide improved ideas yet. Do not critique things that are already great, and you can ask things like "throw out ideas X,Y,Z and just focus on refining A and B" or "keep all ideas but make them all better in XYZ ways"
# """

# CRITIC_USER = """Critique the following AI alignment research directions for clarity, novelty, feasibility, and actionable value.

# Directions:
# {ideas}
# """

# RE_IDEATE_SYSTEM = """You are an AI alignment ideation agent revisiting the directions after critique.
# Instructions:
# - Consider the critique carefully.
# - Produce a new standalone set of improved directions that address the critique.
# - The new ideas should read as original, not a response to critique. Do not mention the critique explicitly, just integrate improvements.

# Use format:
# CRITIQUE_START
# (Brief reasoning about what needed improvement)
# CRITIQUE_END

# IMPROVED_START
# (Your improved, standalone directions)
# IMPROVED_END
# """

# RE_IDEATE_USER = """Here are the previous directions and the critique:

# Previous Directions:
# {ideas}

# Critique:
# {critique}

# Now produce improved directions as per instructions.
# """

# TERMINATOR_SYSTEM = """You are an evaluator of alignment research directions.
# If good enough to present back to the user, say "Good enough".
# If not, say "Needs more iteration: <reason>" where <reason> is clear and specific.
# No other text.
# """

# TERMINATOR_USER = """Evaluate these improved alignment directions:

# {refined_ideas}

# Are they good enough or need more iteration?
# """

# TERMINATOR_ASSESS_SYSTEM = """You are an evaluator.
# Given the final alignment directions, provide a strengths and weaknesses assessment since they were not approved after multiple iterations. Be factual and concise.
# """

# TERMINATOR_ASSESS_USER = """These are the final directions after all attempts:

# {final_ideas}

# Please provide their strengths and weaknesses.
# """

# ########################################
# # WORKFLOW FUNCTIONS
# ########################################

# def summarize_text(user_text, model, verbose):
#     messages = [
#         {"role": "system", "content": SUMMARIZER_SYSTEM},
#         {"role": "user", "content": SUMMARIZER_USER.format(user_text=user_text)}
#     ]
#     return run_completion(messages, model, verbose=verbose)

# def ideate(overview, additional_context, model, verbose):
#     messages = [
#         {"role": "system", "content": IDEATOR_SYSTEM},
#         {"role": "user", "content": IDEATOR_IDEATION_USER.format(overview=overview, additional_context=additional_context)}
#     ]
#     return run_completion(messages, model, verbose=verbose)

# def critique(ideas, model, verbose):
#     messages = [
#         {"role": "system", "content": CRITIC_SYSTEM},
#         {"role": "user", "content": CRITIC_USER.format(ideas=ideas)}
#     ]
#     return run_completion(messages, model, verbose=verbose)

# def re_ideate(ideas, critique_text, model, verbose):
#     messages = [
#         {"role": "system", "content": RE_IDEATE_SYSTEM},
#         {"role": "user", "content": RE_IDEATE_USER.format(ideas=ideas, critique=critique_text)}
#     ]
#     return run_completion(messages, model, verbose=verbose)

# def check_termination(refined_ideas, model, verbose):
#     messages = [
#         {"role": "system", "content": TERMINATOR_SYSTEM},
#         {"role": "user", "content": TERMINATOR_USER.format(refined_ideas=refined_ideas)}
#     ]
#     return run_completion(messages, model, verbose=verbose)

# def assess_strengths_weaknesses(final_ideas, model, verbose):
#     messages = [
#         {"role": "system", "content": TERMINATOR_ASSESS_SYSTEM},
#         {"role": "user", "content": TERMINATOR_ASSESS_USER.format(final_ideas=final_ideas)}
#     ]
#     return run_completion(messages, model, verbose=verbose)

# def parse_refinement_output(refinement_output):
#     critique_pattern = r"CRITIQUE_START\s*(.*?)\s*CRITIQUE_END"
#     improved_pattern = r"IMPROVED_START\s*(.*?)\s*IMPROVED_END"

#     critique_match = re.search(critique_pattern, refinement_output, re.DOTALL)
#     improved_match = re.search(improved_pattern, refinement_output, re.DOTALL)

#     if critique_match and improved_match:
#         critique_section = critique_match.group(1).strip()
#         final_refined_ideas = improved_match.group(1).strip()
#     else:
#         critique_section = "Could not parse critique."
#         final_refined_ideas = refinement_output.strip()

#     return critique_section, final_refined_ideas

# ########################################
# # CHAT AND UTILS
# ########################################

# def ensure_conversation_state(key):
#     if key not in st.session_state.conversation_states:
#         st.session_state.conversation_states[key] = []

# def chat_interface(key, base_content, model, verbose):
#     """
#     Displays a chat interface for the given 'key' with the base_content.
#     The entire conversation is stored in st.session_state.conversation_states[key].
#     """
#     ensure_conversation_state(key)
#     # Show existing chat messages
#     for msg in st.session_state.conversation_states[key]:
#         if msg["role"] == "user":
#             with st.chat_message("user"):
#                 st.write(msg["content"])
#         else:
#             with st.chat_message("assistant"):
#                 st.write(msg["content"])

#     # New user input
#     prompt = st.chat_input("Ask a question or refine this content further", key=f"chat_input_{key}")
#     if prompt:
#         st.session_state.conversation_states[key].append({"role":"user","content":prompt})
#         # We'll pass the entire conversation to the model as context, plus the base_content
#         # so it knows what it's referencing
#         conv_messages = [
#             {
#                 "role":"system",
#                 "content":"You are a helpful assistant. You have the following content:\n\n" 
#                           + base_content 
#                           + "\n\nUser and assistant messages follow. Answer user queries helpfully."
#             }
#         ]
#         for m in st.session_state.conversation_states[key]:
#             conv_messages.append(m)

#         response = run_completion(conv_messages, model, verbose)
#         st.session_state.conversation_states[key].append({"role":"assistant","content":response})
#         st.rerun()

# def download_expander_content(label, content):
#     return st.download_button(
#         label="Download contents as markdown",
#         data=content.encode("utf-8"),
#         file_name=f"{label.replace(' ','_')}.md",
#         mime="text/markdown",
#     )

# def get_conversation_transcript(key):
#     """Combine all user/assistant messages into a single text block."""
#     ensure_conversation_state(key)
#     lines = []
#     for m in st.session_state.conversation_states[key]:
#         role = m["role"].upper()
#         text = m["content"]
#         lines.append(f"{role}: {text}")
#     return "\n".join(lines)

# ########################################
# # PDF UPLOAD
# ########################################

# pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
# pdf_text = ""
# if pdf_file is not None:
#     reader = PdfReader(pdf_file)
#     for page in reader.pages:
#         pdf_text += page.extract_text() + "\n"

# llm_options = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]

# summarizer_model = st.selectbox("Summarizer Model:", llm_options, index=0)
# ideator_model = st.selectbox("Ideator Model:", llm_options, index=0)
# critic_model = st.selectbox("Critic Model:", llm_options, index=0)
# terminator_model = st.selectbox("Terminator Model:", llm_options, index=0)

# verbose = st.checkbox("Show verbose debug info")

# max_iterations = st.number_input("Max Refinement Iterations:", min_value=1, value=3)
# enable_different_angle = st.checkbox("Attempt Different Angle if stuck?", value=False)

# manual_mode = st.checkbox("Manual Mode", value=st.session_state.manual_mode)
# st.session_state.manual_mode = manual_mode

# user_text = st.text_area("Paste your research text (abstract or section):", height=300)
# additional_context = st.text_area("Optional additional angles or considerations:", height=100)

# run_pipeline = st.button("Run Pipeline")

# ########################################
# # Helper: run the Summarizer step
# ########################################

# def do_summarization():
#     # Combine the raw text with any conversation from a "pre_summarizer" key if needed
#     # (In this example we do not have a pre-summarizer chat, but code is flexible.)
#     return summarize_text(
#         st.session_state.final_input_text,
#         summarizer_model,
#         verbose
#     )

# ########################################
# # Manual Mode Step-by-Step Execution
# ########################################

# def run_critique_step(current_ideas):
#     return critique(current_ideas, critic_model, verbose)

# def run_re_ideate_step(current_ideas, critique_text):
#     return re_ideate(current_ideas, critique_text, ideator_model, verbose)

# def run_termination_check(refined_ideas):
#     return check_termination(refined_ideas, terminator_model, verbose)

# def run_strengths_weaknesses_assessment(current_ideas):
#     return assess_strengths_weaknesses(current_ideas, terminator_model, verbose)

# def proceed_from_summarization():
#     """
#     Gathers the conversation from 'overview' key, appends it to the summary,
#     then runs ideation immediately, sets step to 1.
#     """
#     summary = st.session_state.pipeline_results["overview"]
#     conversation_text = get_conversation_transcript("overview")  # entire chat with summary
#     combined_text = summary + "\n\n----\nConversation:\n" + conversation_text
#     # Now do ideation
#     output = ideate(combined_text, additional_context, ideator_model, verbose)
#     st.session_state.pipeline_results["initial_ideas"] = output
#     # Initialize iteration_data if needed
#     if "iteration_data" not in st.session_state.pipeline_results:
#         st.session_state.pipeline_results["iteration_data"] = []
#     if len(st.session_state.pipeline_results["iteration_data"]) == 0:
#         st.session_state.pipeline_results["iteration_data"].append(("", "", output, ""))
#     # Next step
#     st.session_state.manual_current_step = 1
#     st.rerun()

# def proceed_from_ideation():
#     """
#     Gathers conversation from 'initial_ideas' key, appends it to the existing ideas,
#     then does a critique. Step -> 2
#     """
#     iteration_data = st.session_state.pipeline_results["iteration_data"]
#     current_ideas = iteration_data[-1][2]  # last improved ideas
#     conversation_text = get_conversation_transcript("initial_ideas")
#     combined_text = current_ideas + "\n\n----\nConversation:\n" + conversation_text
#     new_critique = run_critique_step(combined_text)
#     iteration_data[-1] = (new_critique, "", current_ideas, "")
#     st.session_state.pipeline_results["iteration_data"] = iteration_data
#     st.session_state.manual_current_step = 2
#     st.rerun()

# def proceed_from_critique():
#     """
#     Gathers conversation from iteration_{len(iteration_data)}_critique,
#     merges with the existing critique text, does re-ideation. Step -> 3
#     """
#     iteration_data = st.session_state.pipeline_results["iteration_data"]
#     # iteration_data[-1] = (critique_text, critique_section, improved_ideas, verdict)
#     critique_key = f"iteration_{len(iteration_data)}_critique"
#     old_critique = iteration_data[-1][0]
#     conversation_text = get_conversation_transcript(critique_key)
#     combined_critique = old_critique + "\n\n----\nConversation:\n" + conversation_text
#     current_ideas = iteration_data[-1][2]
#     refinement_out = run_re_ideate_step(current_ideas, combined_critique)
#     critique_section, improved_ideas = parse_refinement_output(refinement_out)
#     iteration_data[-1] = (old_critique, critique_section, improved_ideas, "")
#     st.session_state.pipeline_results["iteration_data"] = iteration_data
#     st.session_state.manual_current_step = 3
#     st.rerun()

# def proceed_from_reideate():
#     """
#     Gathers conversation from iteration_{len(iteration_data)}_internal_reasoning and
#     iteration_{len(iteration_data)}_refined_directions, merges them with improved_ideas,
#     then runs termination check. Step -> 4
#     """
#     iteration_data = st.session_state.pipeline_results["iteration_data"]
#     idx = len(iteration_data)
#     critique_section_key = f"iteration_{idx}_internal_reasoning"
#     refined_dir_key = f"iteration_{idx}_refined_directions"

#     # We won't actually combine them with the improved ideas text for the termination check,
#     # but let's do it anyway for completeness. It's all just extra context.
#     # The only required input for termination is the improved_ideas themselves,
#     # but let's pass the conversation in if we want to keep it consistent.
#     improved_ideas = iteration_data[-1][2]
#     critique_section = iteration_data[-1][1]

#     conv_critique_section = get_conversation_transcript(critique_section_key)
#     conv_refined = get_conversation_transcript(refined_dir_key)
#     combined_for_termination = (
#         improved_ideas
#         + "\n\n----\nCRITIQUE_SECTION_CONV:\n"
#         + critique_section
#         + "\n\n----\nCRITIQUE_SECTION_CHAT:\n"
#         + conv_critique_section
#         + "\n\n----\nREFINED_DIR_CHAT:\n"
#         + conv_refined
#     )
#     new_verdict = run_termination_check(combined_for_termination)
#     iteration_data[-1] = (
#         iteration_data[-1][0],
#         iteration_data[-1][1],
#         iteration_data[-1][2],
#         new_verdict,
#     )
#     st.session_state.pipeline_results["iteration_data"] = iteration_data
#     st.session_state.manual_current_step = 4
#     st.rerun()

# def proceed_after_termination():
#     """
#     If verdict says 'needs more iteration' but we still have capacity,
#     or if it's good enough, or if we've exhausted tries.
#     """
#     iteration_data = st.session_state.pipeline_results["iteration_data"]
#     verdict = iteration_data[-1][3]
#     final_ideas = iteration_data[-1][2]

#     if verdict.startswith("Good enough"):
#         st.session_state.pipeline_results["final_good_enough"] = True
#         st.session_state.pipeline_results["final_ideas"] = final_ideas
#         st.session_state.manual_current_step = 999
#         st.rerun()
#     else:
#         if len(iteration_data) < max_iterations:
#             # proceed to another iteration
#             iteration_data.append(("", "", final_ideas, ""))
#             st.session_state.pipeline_results["iteration_data"] = iteration_data
#             st.session_state.manual_current_step = 2  # go to Critique again
#             st.rerun()
#         else:
#             # final not good enough
#             st.session_state.pipeline_results["final_good_enough"] = False
#             st.session_state.pipeline_results["final_ideas"] = final_ideas
#             st.session_state.manual_current_step = 999
#             st.rerun()

# ########################################
# # Main Execution Logic
# ########################################

# if run_pipeline:
#     # Reset everything when pipeline is run
#     st.session_state.conversation_states.clear()
#     st.session_state.pipeline_results.clear()
#     st.session_state.pipeline_ran = False
#     st.session_state.manual_current_step = 0
#     st.session_state.final_input_text = pdf_text.strip() if pdf_text.strip() else user_text.strip()

#     if not st.session_state.final_input_text:
#         st.warning("Please provide text or upload a PDF before running the pipeline.")
#     else:
#         if not manual_mode:
#             # AUTOMATIC PIPELINE
#             overview = summarize_text(st.session_state.final_input_text, summarizer_model, verbose)
#             initial_ideas = ideate(overview, additional_context, ideator_model, verbose)

#             iteration_count = 0
#             final_good_enough = False
#             current_ideas = initial_ideas

#             iteration_data = []

#             while iteration_count < max_iterations:
#                 critique_text = critique(current_ideas, critic_model, verbose)
#                 refinement_output = re_ideate(current_ideas, critique_text, ideator_model, verbose)
#                 critique_section, improved_ideas = parse_refinement_output(refinement_output)

#                 verdict = check_termination(improved_ideas, terminator_model, verbose)
#                 iteration_data.append((critique_text, critique_section, improved_ideas, verdict))

#                 if verdict.startswith("Good enough"):
#                     final_good_enough = True
#                     current_ideas = improved_ideas
#                     break
#                 else:
#                     current_ideas = improved_ideas
#                     iteration_count += 1

#             if not final_good_enough:
#                 assessment = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
#                 st.session_state.pipeline_results["assessment"] = assessment
#             else:
#                 st.session_state.pipeline_results["assessment"] = None

#             # Store pipeline results
#             st.session_state.pipeline_results["overview"] = overview
#             st.session_state.pipeline_results["initial_ideas"] = initial_ideas
#             st.session_state.pipeline_results["iteration_data"] = iteration_data
#             st.session_state.pipeline_results["final_good_enough"] = final_good_enough
#             st.session_state.pipeline_results["final_ideas"] = current_ideas
#             st.session_state.pipeline_results["additional_context"] = additional_context

#             st.session_state.pipeline_ran = True
#             st.rerun()

#         else:
#             # MANUAL MODE
#             # Start step 0 (summarization)
#             st.session_state.pipeline_ran = True
#             st.session_state.manual_current_step = 0
#             st.session_state.pipeline_results["iteration_data"] = []
#             st.rerun()

# ########################################
# # MANUAL MODE UI RENDERING
# ########################################

# if st.session_state.pipeline_ran and st.session_state.manual_mode and st.session_state.final_input_text:
#     iteration_data = st.session_state.pipeline_results.get("iteration_data", [])
#     current_step = st.session_state.manual_current_step

#     # Step definitions in manual mode:
#     # 0: Summarization
#     #   after we get a summary, we can chat with it, rerun, or proceed -> auto-run Ideation
#     # 1: Ideation
#     #   chat, rerun if desired, or proceed -> auto-run Critique
#     # 2: Critique
#     #   chat, rerun if desired, or proceed -> auto-run Re-Ideate
#     # 3: Re-Ideate
#     #   chat, rerun if desired, or proceed -> auto-run Termination
#     # 4: Termination
#     #   chat, rerun if desired, or proceed -> either more iteration or finalize
#     # 999: Done

#     if current_step == 0:
#         st.subheader("Step 1: Summarization")
#         if "overview" not in st.session_state.pipeline_results:
#             st.info("Click 'Run Summarization' to generate the summary. Then you can chat, rerun, or proceed to Ideation.")
#         else:
#             # Show existing summary
#             with st.expander("Summarization Output", expanded=False):
#                 st.write(st.session_state.pipeline_results["overview"])
#                 download_expander_content("overview", st.session_state.pipeline_results["overview"])

#             with st.expander("Summarization Chat (Closed by default)", expanded=False):
#                 chat_interface("overview", st.session_state.pipeline_results["overview"], summarizer_model, verbose)

#         if st.button("Run Summarization"):
#             st.session_state.pipeline_results["overview"] = do_summarization()
#             st.rerun()

#         if "overview" in st.session_state.pipeline_results:
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("Rerun Summarization"):
#                     st.session_state.pipeline_results["overview"] = do_summarization()
#                     st.rerun()
#             with col2:
#                 if st.button("Proceed to Ideation"):
#                     proceed_from_summarization()

#     elif current_step == 1:
#         st.subheader("Step 2: Ideation")
#         if "initial_ideas" not in st.session_state.pipeline_results:
#             st.info("We do not have initial ideas yet. Possibly something went wrong.")
#         else:
#             with st.expander("Initial Alignment Ideas", expanded=False):
#                 st.write(st.session_state.pipeline_results["initial_ideas"])
#                 download_expander_content("initial_alignment_ideas", st.session_state.pipeline_results["initial_ideas"])

#             with st.expander("Ideation Chat (Closed by default)", expanded=False):
#                 chat_interface("initial_ideas", st.session_state.pipeline_results["initial_ideas"], ideator_model, verbose)

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Rerun Ideation"):
#                 # We'll combine the conversation with the current summary to produce new ideas
#                 # That means: combine st.session_state.pipeline_results["overview"] with
#                 # the entire conversation from "initial_ideas" (if any).
#                 # Then do ideate again.
#                 overview_plus_chat = (
#                     st.session_state.pipeline_results["overview"]
#                     + "\n\n----\nConversation:\n"
#                     + get_conversation_transcript("initial_ideas")
#                 )
#                 new_ideas = ideate(overview_plus_chat, additional_context, ideator_model, verbose)
#                 st.session_state.pipeline_results["initial_ideas"] = new_ideas
#                 # Update iteration_data[-1] if it exists
#                 if len(iteration_data) == 0:
#                     iteration_data.append(("", "", new_ideas, ""))
#                 else:
#                     iteration_data[-1] = (iteration_data[-1][0], iteration_data[-1][1], new_ideas, iteration_data[-1][3])
#                 st.session_state.pipeline_results["iteration_data"] = iteration_data
#                 st.rerun()
#         with col2:
#             if st.button("Proceed to Critique"):
#                 proceed_from_ideation()

#     elif current_step == 2:
#         st.subheader("Step 3: Critique")
#         # iteration_data[-1] = (critique_text, critique_section, improved_ideas, verdict)
#         if not iteration_data or not iteration_data[-1][0]:
#             st.info("No critique has been run or stored yet. Possibly click 'Rerun Critique'.")
#         else:
#             with st.expander(f"Iteration {len(iteration_data)} Critique", expanded=False):
#                 st.write(iteration_data[-1][0])
#                 download_expander_content(f"iteration_{len(iteration_data)}_critique", iteration_data[-1][0])

#             with st.expander("Critique Chat (Closed by default)", expanded=False):
#                 chat_interface(f"iteration_{len(iteration_data)}_critique", iteration_data[-1][0], critic_model, verbose)

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Rerun Critique"):
#                 current_ideas = iteration_data[-1][2]
#                 combined_ideas = (
#                     current_ideas
#                     + "\n\n----\nConversation:\n"
#                     + get_conversation_transcript(f"iteration_{len(iteration_data)}_critique")
#                 )
#                 new_critique = run_critique_step(combined_ideas)
#                 iteration_data[-1] = (new_critique, iteration_data[-1][1], current_ideas, iteration_data[-1][3])
#                 st.session_state.pipeline_results["iteration_data"] = iteration_data
#                 st.rerun()
#         with col2:
#             if st.button("Proceed to Re-Ideate"):
#                 proceed_from_critique()

#     elif current_step == 3:
#         st.subheader("Step 4: Re-Ideate")
#         # iteration_data[-1] = (critique_text, critique_section, improved_ideas, verdict)
#         idx = len(iteration_data)
#         critique_section = iteration_data[-1][1]
#         improved_ideas = iteration_data[-1][2]

#         if not critique_section and not improved_ideas:
#             st.info("No re-ideation has been run or stored yet. Possibly click 'Rerun Re-Ideate'.")
#         else:
#             with st.expander(f"Iteration {idx} Internal Reasoning (CRITIQUE_START/END)", expanded=False):
#                 st.write(critique_section)
#                 download_expander_content(f"iteration_{idx}_internal_reasoning", critique_section)

#             with st.expander(f"Iteration {idx} Refined Directions", expanded=False):
#                 st.write(improved_ideas)
#                 download_expander_content(f"iteration_{idx}_refined_directions", improved_ideas)

#             with st.expander("Re-Ideation Chats (Closed by default)", expanded=False):
#                 # Chat for the CRITIQUE_START/END portion
#                 chat_interface(f"iteration_{idx}_internal_reasoning", critique_section, ideator_model, verbose)
#                 st.write("---")
#                 # Chat for the refined directions portion
#                 chat_interface(f"iteration_{idx}_refined_directions", improved_ideas, ideator_model, verbose)

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Rerun Re-Ideate"):
#                 old_critique = iteration_data[-1][0]
#                 old_crit_section = iteration_data[-1][1]
#                 old_improved = iteration_data[-1][2]
#                 # Combine conversation from the two expanders
#                 conv_1 = get_conversation_transcript(f"iteration_{idx}_internal_reasoning")
#                 conv_2 = get_conversation_transcript(f"iteration_{idx}_refined_directions")
#                 combined_for_reideate = (
#                     old_improved
#                     + "\n\n----\n(Existing critique text):\n"
#                     + old_critique
#                     + "\n\n----\nCRITIQUE_SECTION:\n"
#                     + old_crit_section
#                     + "\n\n----\nCONV_INTERNAL:\n"
#                     + conv_1
#                     + "\n\n----\nCONV_REFINED:\n"
#                     + conv_2
#                 )
#                 refinement_out = run_re_ideate_step(old_improved, combined_for_reideate)
#                 new_critique_section, new_improved_ideas = parse_refinement_output(refinement_out)
#                 iteration_data[-1] = (old_critique, new_critique_section, new_improved_ideas, iteration_data[-1][3])
#                 st.session_state.pipeline_results["iteration_data"] = iteration_data
#                 st.rerun()
#         with col2:
#             if st.button("Proceed to Termination Check"):
#                 proceed_from_reideate()

#     elif current_step == 4:
#         st.subheader("Step 5: Termination Check")
#         idx = len(iteration_data)
#         verdict = iteration_data[-1][3]
#         if not verdict:
#             st.info("No verdict yet. Possibly click 'Rerun Termination Check'.")
#         else:
#             st.write(f"Termination Check: {verdict}")

#         with st.expander("Termination Check Chat (Closed by default)", expanded=False):
#             chat_interface(f"iteration_{idx}_termination", iteration_data[-1][2], terminator_model, verbose)

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("Rerun Termination Check"):
#                 # Combine existing refined ideas with any chat under iteration_{idx}_termination
#                 improved_ideas = iteration_data[-1][2]
#                 conversation_text = get_conversation_transcript(f"iteration_{idx}_termination")
#                 combined_for_termination = (
#                     improved_ideas
#                     + "\n\n----\nTermination Chat:\n"
#                     + conversation_text
#                 )
#                 new_verdict = run_termination_check(combined_for_termination)
#                 iteration_data[-1] = (
#                     iteration_data[-1][0],
#                     iteration_data[-1][1],
#                     iteration_data[-1][2],
#                     new_verdict,
#                 )
#                 st.session_state.pipeline_results["iteration_data"] = iteration_data
#                 st.rerun()
#         with col2:
#             if st.button("Proceed (Finalize or More Iteration)"):
#                 proceed_after_termination()

#     elif current_step == 999:
#         st.subheader("Results")
#         final_good_enough = st.session_state.pipeline_results.get("final_good_enough", False)
#         final_ideas = st.session_state.pipeline_results.get("final_ideas", "")
#         iteration_data = st.session_state.pipeline_results["iteration_data"]

#         if final_good_enough:
#             with st.expander("Final Accepted Ideas", expanded=True):
#                 st.write(final_ideas)
#                 download_expander_content("final_accepted_ideas", final_ideas)
#                 chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
#         else:
#             st.warning("Not approved after max iterations. Displaying final output and assessing strengths & weaknesses.")
#             with st.expander("Final Non-Approved Ideas", expanded=False):
#                 st.write(final_ideas)
#                 download_expander_content("final_non_approved_ideas", final_ideas)
#                 chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

#             if "assessment" not in st.session_state.pipeline_results:
#                 assessment = run_strengths_weaknesses_assessment(final_ideas)
#                 st.session_state.pipeline_results["assessment"] = assessment
#             else:
#                 assessment = st.session_state.pipeline_results["assessment"]

#             with st.expander("Strengths & Weaknesses Assessment", expanded=True):
#                 st.write(assessment)
#                 download_expander_content("strengths_weaknesses_assessment", assessment)
#                 chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)

#         st.success("Manual pipeline run complete.")

# ########################################
# # Automatic Mode Results Display
# ########################################

# if st.session_state.pipeline_ran and not st.session_state.manual_mode:
#     overview = st.session_state.pipeline_results["overview"]
#     initial_ideas = st.session_state.pipeline_results["initial_ideas"]
#     iteration_data = st.session_state.pipeline_results["iteration_data"]
#     final_good_enough = st.session_state.pipeline_results["final_good_enough"]
#     final_ideas = st.session_state.pipeline_results["final_ideas"]
#     assessment = st.session_state.pipeline_results["assessment"]

#     with st.expander("Comprehensive Systematic Overview", expanded=False):
#         st.write(overview)
#         download_expander_content("overview", overview)
#         chat_interface("overview", overview, ideator_model, verbose)

#     with st.expander("Initial Alignment Ideas", expanded=False):
#         st.write(initial_ideas)
#         download_expander_content("initial_alignment_ideas", initial_ideas)
#         chat_interface("initial_ideas", initial_ideas, ideator_model, verbose)

#     for i, (critique_text, critique_section, improved_ideas, verdict) in enumerate(iteration_data, start=1):
#         with st.expander(f"Iteration {i} Critique", expanded=False):
#             st.write(critique_text)
#             download_expander_content(f"iteration_{i}_critique", critique_text)
#             chat_interface(f"iteration_{i}_critique", critique_text, ideator_model, verbose)

#         with st.expander(f"Iteration {i} Internal Reasoning (CRITIQUE_START/END)", expanded=False):
#             st.write(critique_section)
#             download_expander_content(f"iteration_{i}_internal_reasoning", critique_section)
#             chat_interface(f"iteration_{i}_internal_reasoning", critique_section, ideator_model, verbose)

#         with st.expander(f"Iteration {i} Refined Directions", expanded=False):
#             st.write(improved_ideas)
#             download_expander_content(f"iteration_{i}_refined_directions", improved_ideas)
#             chat_interface(f"iteration_{i}_refined_directions", improved_ideas, ideator_model, verbose)

#         st.write(f"Termination Check: {verdict}")

#     if final_good_enough:
#         with st.expander("Final Accepted Ideas", expanded=True):
#             st.write(final_ideas)
#             download_expander_content("final_accepted_ideas", final_ideas)
#             chat_interface("final_accepted_ideas", final_ideas, ideator_model, verbose)
#     else:
#         st.warning("Not approved after max iterations. Displaying final output and assessing strengths & weaknesses.")
#         with st.expander("Final Non-Approved Ideas", expanded=False):
#             st.write(final_ideas)
#             download_expander_content("final_non_approved_ideas", final_ideas)
#             chat_interface("final_non_approved_ideas", final_ideas, ideator_model, verbose)

#         if assessment is not None:
#             with st.expander("Strengths & Weaknesses Assessment", expanded=True):
#                 st.write(assessment)
#                 download_expander_content("strengths_weaknesses_assessment", assessment)
#                 chat_interface("strengths_weaknesses_assessment", assessment, ideator_model, verbose)
