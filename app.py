import streamlit as st
from openai import OpenAI

# Replace 'key' with your actual OpenAI API key.
client = OpenAI(api_key=st.secrets['API_KEY'])

def run_completion(messages):
    completion = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages
    )
    return completion.choices[0].message.content

########################################
# PROMPT TEMPLATES
########################################

# System prompt for Summarization stage
SUMMARIZER_SYSTEM = """You are a research assistant specializing in analyzing scholarly text and identifying potential AI alignment implications.
Your goals:
1. Ingest the provided research text (like a paper abstract or section).
2. Extract key findings, core concepts, and any methods or hypotheses relevant to advanced AI.
3. Identify potential links or implications for AI alignment, control, safety, or interpretability.
4. Provide a structured summary focusing on aspects that could be further developed into alignment research directions.

You are meticulous, concise, and strive to present information faithfully without hallucination.
If information is not present, be clear about that. If you speculate, label it as speculation.
"""

SUMMARIZER_USER = """Summarize the following research text and highlight aspects that could inform or inspire new AI alignment research directions.

Text:
{user_text}
"""

# System prompt for Ideation/Refinement stage
# This prompt guides the LLM to:
# - Consider the summary and generate novel alignment ideas.
# - Critique and refine them upon request.
IDEATOR_SYSTEM = """You are an AI alignment research ideation agent.
Your role:
1. Given a structured summary of a research paper, propose several innovative and potentially neglected approaches or directions that advance AI alignment.
2. Think creatively but remain grounded in scientific plausibility. Incorporate relevant alignment concepts like interpretability, corrigibility, safe exploration, etc.
3. When asked to refine, first critique the ideas you produced, identify weaknesses, then improve them. Make them more concrete, feasible, and scientifically grounded.
4. Be aware of repetitive patterns or “basins” of thought. If you find yourself circling the same concepts, consciously try a radically different angle or analogy to break free.
5. Always maintain a scholarly tone and cite known alignment concepts or research directions where possible (even if at a high level).

Your output should be structured and clear, and after critique, provide improved versions that address identified shortcomings.
"""

IDEATOR_IDEATION_USER = """Below is the summary of the research text. Based on this summary, propose several original and forward-looking AI alignment research directions. Focus on novelty, feasibility, and potential for significant impact.

Summary:
{summary}
"""

IDEATOR_REFINE_USER = """Here are the current alignment ideas you produced:

{ideas}

Step 1: Critique these ideas. Identify their weaknesses, gaps, or unrealistic assumptions.
Step 2: After critiquing them, rewrite or refine these ideas to be more robust, implementable, and well-grounded in known alignment literature or methodologies.
If you notice repetitive patterns, try introducing a fresh perspective or approach from a different domain.
"""

# System prompt for Termination Check
TERMINATION_SYSTEM = """You are an evaluator judging the quality of proposed alignment research directions.
Your role:
1. Assess whether the final set of ideas is sufficiently novel, clear, and feasible.
2. If these ideas meet a reasonable threshold for alignment research quality (e.g., they offer at least one new angle, are not obviously repetitive, and have some actionable element), respond with "Good enough".
3. If they are still vague, repetitive, or unconvincing, respond with "Needs more iteration".
4. Do not provide explanations beyond these statements. Just choose one response.
"""

TERMINATION_USER = """Evaluate these refined alignment ideas and determine if they are "Good enough" or "Needs more iteration":

{refined_ideas}
"""

# A helper for a different angle attempt if stuck
DIFFERENT_ANGLE_SYSTEM = """You are a resourceful alignment ideation assistant trying to break out of repetitive patterns.
If asked, you will propose a fundamentally different set of approaches, leveraging analogies from unrelated fields, new theoretical frameworks, or fresh metrics, to breathe new life into the alignment directions.
"""

DIFFERENT_ANGLE_USER = """Current refined ideas seem repetitive or stuck. Provide a fundamentally different set of alignment directions, potentially inspired by fields or analogies not previously considered. Aim for genuine novelty and break from earlier patterns.

Current ideas:
{refined_ideas}
"""

########################################
# STREAMLIT UI
########################################

st.title("Alignment Research Automator (MVP)")

if 'summary' not in st.session_state:
    st.session_state.summary = None

if 'ideas' not in st.session_state:
    st.session_state.ideas = None

if 'refined_ideas' not in st.session_state:
    st.session_state.refined_ideas = None

if 'iteration_count' not in st.session_state:
    st.session_state.iteration_count = 0

user_text = st.text_area("Paste your research text here (e.g., abstract or section of a paper):", height=300)

if st.button("Summarize & Extract Alignment Angles"):
    if user_text.strip():
        messages = [
            {"role": "system", "content": SUMMARIZER_SYSTEM},
            {"role": "user", "content": SUMMARIZER_USER.format(user_text=user_text)}
        ]
        summary = run_completion(messages)
        st.session_state.summary = summary
        st.session_state.ideas = None
        st.session_state.refined_ideas = None
        st.session_state.iteration_count = 0
        st.success("Summary generated!")
    else:
        st.warning("Please provide some text before summarizing.")

if st.session_state.summary:
    st.subheader("Summary & Initial Context")
    st.write(st.session_state.summary)

    if st.button("Generate Alignment Ideas"):
        ideation_messages = [
            {"role": "system", "content": IDEATOR_SYSTEM},
            {"role": "user", "content": IDEATOR_IDEATION_USER.format(summary=st.session_state.summary)}
        ]
        ideas = run_completion(ideation_messages)
        st.session_state.ideas = ideas
        st.session_state.refined_ideas = None
        st.session_state.iteration_count = 0
        st.success("Initial alignment ideas generated!")

if st.session_state.ideas:
    st.subheader("Initial Ideas")
    st.write(st.session_state.ideas)

    if st.button("Refine Ideas"):
        refine_messages = [
            {"role": "system", "content": IDEATOR_SYSTEM},
            {"role": "user", "content": IDEATOR_REFINE_USER.format(ideas=st.session_state.refined_ideas if st.session_state.refined_ideas else st.session_state.ideas)}
        ]
        refined = run_completion(refine_messages)
        st.session_state.refined_ideas = refined
        st.session_state.iteration_count += 1
        st.success("Ideas refined and improved!")

if st.session_state.refined_ideas:
    st.subheader("Refined Ideas")
    st.write(st.session_state.refined_ideas)

    if st.button("Check if Good Enough"):
        termination_messages = [
            {"role": "system", "content": TERMINATION_SYSTEM},
            {"role": "user", "content": TERMINATION_USER.format(refined_ideas=st.session_state.refined_ideas)}
        ]
        verdict = run_completion(termination_messages)
        st.write("Termination Check:", verdict)

        if "Needs more iteration" in verdict:
            st.info("The evaluator suggests more refinement. Consider refining again or trying a different angle.")
            if st.session_state.iteration_count >= 2:
                st.warning("Ideas may be stuck in a repetitive pattern. Try a different angle.")
        else:
            st.success("The ideas seem good enough!")

    # Give the option to refine again if needed
    if st.session_state.iteration_count >= 1:
        if st.button("Try a Different Angle"):
            angle_messages = [
                {"role": "system", "content": DIFFERENT_ANGLE_SYSTEM},
                {"role": "user", "content": DIFFERENT_ANGLE_USER.format(refined_ideas=st.session_state.refined_ideas)}
            ]
            diff_angle_ideas = run_completion(angle_messages)
            st.session_state.refined_ideas = diff_angle_ideas
            st.success("Proposed a fundamentally different angle!")

st.write("---")
st.write("This MVP attempts a more role- and context-rich approach to AI alignment ideation from research text. Adjust prompts, iterate, and improve as needed.")
