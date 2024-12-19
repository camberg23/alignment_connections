import streamlit as st
import re
import anthropic
from openai import OpenAI

# Initialize instructions
st.title("Alignment Research Automator")
st.write("This tool takes a research text, summarizes it, proposes alignment research directions, critiques them, and refines them based on termination criteria. The goal is to produce actionable, high-quality alignment research directions.")
st.write("You can choose different LLMs for each stage (Summarizer, Ideator, Critic, Terminator) and run multiple iterations. If after N iterations it still isn't good enough, final output and a strengths/weaknesses assessment is displayed.")

# Secrets
openai_client = OpenAI(api_key=st.secrets['API_KEY'])
anthropic_client = anthropic.Client(api_key=st.secrets['ANTHROPIC_API_KEY'])

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

def convert_messages_for_anthropic(messages):
    # Anthropics expects a single prompt string. We'll follow a simple heuristic:
    # - Combine all system & user as human, assistant as assistant.
    # - For system messages, we'll prepend instructions at the start of the conversation.
    # We assume system and user are effectively from the human side, assistant from the assistant side.
    # We'll form a conversation string: 
    # Human: {system_msg + user_msg}, Assistant: {assistant_msg}, etc.
    
    # We'll accumulate all instructions and user content into one human turn at the start.
    system_instructions = []
    conversation = []
    human_turn = ""
    
    for msg in messages:
        if msg["role"] == "system":
            system_instructions.append(msg["content"])
        elif msg["role"] == "user":
            # Each user message is a human turn. If we have system instructions, prepend once at the start.
            # We'll just combine all system instructions into the first human turn.
            if system_instructions:
                human_turn += "\n".join(system_instructions) + "\n"
                system_instructions = []
            human_turn += msg["content"]
            conversation.append((human_turn, "human"))
            human_turn = ""
        elif msg["role"] == "assistant":
            # Assistant turn
            conversation.append((msg["content"], "assistant"))
    
    # If there were system instructions left with no user turn afterward, put them as a last human turn
    # (though typically we'd always have a user message after system)
    if system_instructions:
        conversation.insert(0, ("\n".join(system_instructions), "human"))
        
    # Construct the full prompt
    # Anthropics format: HUMAN_PROMPT = "<|HUMAN|>" AI_PROMPT="<|ASSISTANT|>" 
    # We'll use anthropic.HUMAN_PROMPT and anthropic.AI_PROMPT
    full_prompt = ""
    # We alternate between human and assistant. The first message should be human (if any)
    for (content, role) in conversation:
        if role == "human":
            full_prompt += f"{anthropic.HUMAN_PROMPT} {content}"
        else:
            full_prompt += f"{anthropic.AI_PROMPT} {content}"
    # End with an AI_PROMPT to get assistant completion
    full_prompt += anthropic.AI_PROMPT
    return full_prompt

def run_completion(messages, model, verbose=False):
    # Handle model type and transform messages as needed
    if is_o1_model(model):
        messages = adjust_messages_for_o1(messages)
        # Use openai with the given messages (o1 model)
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
        # Standard openai model with system/user/assistant
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
        # Convert messages to a single prompt
        prompt = convert_messages_for_anthropic(messages)
        if verbose:
            st.write("**Using Anthropic Model:**", model)
            st.write("**Prompt:**", prompt)
        max_tokens = 600
        response = anthropic_client.completions.create(
            model="claude-v1", # or parse model name if needed, here we assume anthropic:claude means claude-v1
            max_tokens=max_tokens,
            prompt=prompt,
            stop=None,
            temperature=1
        )
        if verbose:
            st.write("**Response:**", response.completion)
        return response.completion.strip()
    else:
        raise ValueError("Unknown model type")

########################################
# PROMPTS
########################################

SUMMARIZER_SYSTEM = """You are a specialized research assistant focused on analyzing scholarly research text and identifying AI alignment implications.
Provide a comprehensive, systematic overview:
- Domain, key hypotheses.
- Methods, results, theoretical contributions.
- Potential AI alignment relevance (interpretability, safety, robustness, etc.)
- Integrate additional user angles if any.
Be factual, thorough, no vague language.
"""

SUMMARIZER_USER = """Below is a research text. Provide the comprehensive overview as requested above.

Research Text:
{user_text}

Additional Angles:
{additional_context}
"""

IDEATOR_SYSTEM = """You are an AI alignment research ideation agent.
Steps:
1. Given the overview, propose original alignment research directions.
2. Then we will critique them (no fix at that step).
3. Then you will produce updated directions incorporating the critique seamlessly so the final output is standalone and improved.

Use formatting instructions as provided in refinement steps.
"""

IDEATOR_IDEATION_USER = """Use the overview and additional angles to propose several original AI alignment research directions.

Systematic Overview:
{overview}

Additional Angles:
{additional_context}
"""

CRITIC_SYSTEM = """You are a critic focused on assessing alignment research directions for clarity, novelty, feasibility. Do not fix them, just critique:
- Identify weaknesses
- Suggest what is missing
- Indicate what could be improved

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
If good enough, say "Good enough".
If not, say "Needs more iteration: <reason>" where <reason> is short and specific.
No other text.
"""

TERMINATOR_USER = """Evaluate these improved alignment directions:

{refined_ideas}

Are they good enough or need more iteration?
"""

# If after N iterations still not good enough, ask terminator for strengths & weaknesses.
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

def summarize_text(user_text, additional_context, model, verbose):
    messages = [
        {"role": "system", "content": SUMMARIZER_SYSTEM},
        {"role": "user", "content": SUMMARIZER_USER.format(user_text=user_text, additional_context=additional_context)}
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
        # fallback if parsing fails
        critique_section = "Could not parse critique."
        final_refined_ideas = refinement_output.strip()

    return critique_section, final_refined_ideas

########################################
# UI
########################################

llm_options = ["gpt-4o", "o1-mini", "o1-2024-12-17", "anthropic:claude"]

summarizer_model = st.selectbox("Summarizer Model:", llm_options, index=0)
ideator_model = st.selectbox("Ideator Model:", llm_options, index=0)
critic_model = st.selectbox("Critic Model:", llm_options, index=0)
terminator_model = st.selectbox("Terminator Model:", llm_options, index=0)

verbose = st.checkbox("Show verbose debug info")

max_iterations = st.number_input("Max Refinement Iterations:", min_value=1, value=3)
enable_different_angle = st.checkbox("Attempt Different Angle if stuck?", value=False)
# Note: Different angle step not clearly requested now. The instructions no longer mention different angle step in final architecture. We can remove it or keep it disabled by default.

user_text = st.text_area("Paste your research text (abstract or section):", height=300)
additional_context = st.text_area("Optional additional angles or considerations:", height=100)

if st.button("Run Pipeline"):
    if not user_text.strip():
        st.warning("Please provide some text before running the pipeline.")
    else:
        # 1. Summarize
        overview = summarize_text(user_text, additional_context, summarizer_model, verbose)
        with st.expander("Comprehensive Systematic Overview"):
            st.write(overview)

        # 2. Ideate initial directions
        initial_ideas = ideate(overview, additional_context, ideator_model, verbose)
        with st.expander("Initial Alignment Ideas"):
            st.write(initial_ideas)

        iteration_count = 0
        final_good_enough = False
        current_ideas = initial_ideas

        while iteration_count < max_iterations:
            # 3. Critique (no fix)
            critique_text = critique(current_ideas, critic_model, verbose)
            with st.expander(f"Iteration {iteration_count+1} Critique"):
                st.write(critique_text)

            # 4. Re-ideate with critique
            refinement_output = re_ideate(current_ideas, critique_text, ideator_model, verbose)
            critique_section, improved_ideas = parse_refinement_output(refinement_output)
            with st.expander(f"Iteration {iteration_count+1} Internal Reasoning (CRITIQUE_START/END)"):
                st.write(critique_section)
            with st.expander(f"Iteration {iteration_count+1} Refined Directions"):
                st.write(improved_ideas)

            # 5. Termination check
            verdict = check_termination(improved_ideas, terminator_model, verbose)
            st.write("Termination Check:", verdict)

            if verdict.startswith("Good enough"):
                final_good_enough = True
                current_ideas = improved_ideas
                break
            else:
                # Not good enough, extract reason and iterate again
                reason_match = re.match(r"Needs more iteration:\s*(.*)", verdict)
                if reason_match:
                    termination_reason = reason_match.group(1).strip()
                else:
                    termination_reason = "No specific feedback provided."

                # Incorporate termination feedback as well?
                # The instructions say we show not looping in circles:
                # Actually, the instructions said we feed first rejection criteria on next iteration. 
                # We'll just continue the loop. The next iteration will produce new critique and new re-ideation from scratch anyway.

                # Set current_ideas to improved_ideas for the next iteration
                current_ideas = improved_ideas
                iteration_count += 1

        if final_good_enough:
            with st.expander("Final Accepted Ideas", expanded=True):
                st.write(current_ideas)
        else:
            # After max iterations not good enough
            st.warning("Not approved after max iterations. Displaying final output and assessing strengths & weaknesses.")
            with st.expander("Final Non-Approved Ideas"):
                st.write(current_ideas)
            assessment = assess_strengths_weaknesses(current_ideas, terminator_model, verbose)
            with st.expander("Strengths & Weaknesses Assessment", expanded=True):
                st.write(assessment)
