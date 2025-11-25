import gradio as gr
import os
from dotenv import load_dotenv
from orchestrator import run_pipeline
from rich import print

# Load environment variables from .env file
load_dotenv()

# -------------------------------------------------------
#  Core Function Called By UI
# -------------------------------------------------------

def run_agent(topic, persona, tone, length, technicality, max_items, api_key_state):
    """
    This function handles the logic when the user clicks "Generate Summary".
    """

    # Validate API key - use state value or fall back to env
    api_key = api_key_state if api_key_state and api_key_state.strip() else os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        return (
            "❌ **Error:** Please provide your OpenAI API key in Dev Options, "
            "or set it as an environment variable (OPENAI_API_KEY).",
            "",
            ""
        )

    ui_preferences = {
        "persona": persona,
        "tone": tone,
        "length": length,
        "technicality": technicality,
        "max_items": max_items  # Add max_items to preferences
    }

    try:
        result = run_pipeline(topic, ui_preferences, api_key=api_key.strip())
    except Exception as e:
        return f"❌ **Error during pipeline run:** {e}", "", ""

    brief = result["brief"]
    ctx = result["context"]
    saved_path = result["saved_path"]

    debug_info = f"""
Persona: {ctx.persona}
Preferences: {ctx.preferences}
Implicit Traits: {ctx.implicit_traits}
Patterns: {ctx.patterns}
Recent Topics: {ctx.recent_topics}
Saved File: {saved_path}

📊 Statistics:
- Total items collected: {result.get('total_items_collected', 0)}
- Relevant items processed: {result['relevant_items_count']}
- RSS feeds: {result.get('sources_discovered', {}).get('rss_feeds', 0)}
- Twitter sources: {result.get('sources_discovered', {}).get('twitter_accounts', 0)}
- Websites explored: {result.get('sources_discovered', {}).get('websites_explored', 0)}
- Subreddits: {result.get('sources_discovered', {}).get('subreddits', 0)}
"""

    return brief, saved_path, debug_info



# -------------------------------------------------------
#  Dropdown Options
# -------------------------------------------------------

persona_options = [
    "default",
    "product_manager",
    "researcher",
    "stay_at_home_parent",
    "developer",
    "executive",
]

tone_options = ["neutral", "friendly", "professional", "direct", "simple"]

length_options = ["short", "medium", "long"]

technicality_options = ["simple", "medium", "advanced"]



# -------------------------------------------------------
#  UI / Layout (Gradio 4 compatible)
# -------------------------------------------------------

with gr.Blocks(title="Personal Research Butler") as demo:

    # -------------------- HEADER --------------------
    with gr.Row():
        gr.Markdown("# 🧠 Personal Research Butler")
        dev_options_btn = gr.Button("⚙️ Dev Options", variant="secondary", size="sm", scale=0)
    
    gr.Markdown("Your personalized, AI-powered research concierge.")
    
    # Dev Options - Popup-style interface
    dev_options_visible = gr.State(value=False)
    
    with gr.Column(visible=False) as dev_options_panel:
        gr.Markdown("### ⚙️ Dev Options")
        env_key_status = "✅ Using API key from .env file" if os.getenv("OPENAI_API_KEY") else "⚠️ No API key found in .env"
        gr.Markdown(f"**Status:** {env_key_status}")
        
        api_key_input = gr.Textbox(
            label="OpenAI API Key",
            placeholder="sk-... (enter your API key and click Save)",
            type="password",
            info="Save your API key to the .env file for persistent use."
        )
        
        with gr.Row():
            save_btn = gr.Button("💾 Save API Key", variant="primary", scale=1)
            close_btn = gr.Button("✕ Close", variant="secondary", scale=1)
        
        save_status = gr.Markdown("", visible=False)
    
    # Store the API key in session state
    api_key_state = gr.State(value="")
    
    def toggle_dev_options(visible):
        return not visible, gr.update(visible=not visible)
    
    def save_api_key(key, current_state):
        if not key or not key.strip():
            return (
                gr.update(value=current_state),
                gr.update(value="⚠️ Please enter an API key", visible=True),
                gr.update(visible=True)
            )
        
        try:
            # Read existing .env file
            env_path = ".env"
            env_lines = []
            key_found = False
            
            if os.path.exists(env_path):
                with open(env_path, "r") as f:
                    env_lines = f.readlines()
            
            # Update or add the API key
            new_lines = []
            for line in env_lines:
                if line.startswith("OPENAI_API_KEY="):
                    new_lines.append(f"OPENAI_API_KEY={key.strip()}\n")
                    key_found = True
                else:
                    new_lines.append(line)
            
            if not key_found:
                new_lines.append(f"OPENAI_API_KEY={key.strip()}\n")
            
            # Write back to .env
            with open(env_path, "w") as f:
                f.writelines(new_lines)
            
            # Reload environment
            load_dotenv(override=True)
            
            return (
                gr.update(value=key.strip()),
                gr.update(value="✅ API key saved successfully! The panel will close automatically.", visible=True),
                gr.update(visible=False)  # Hide the panel
            )
        except Exception as e:
            return (
                gr.update(value=current_state),
                gr.update(value=f"❌ Error saving API key: {str(e)}", visible=True),
                gr.update(visible=True)
            )
    
    def close_panel():
        return gr.update(visible=False), gr.update(value="", visible=False)
    
    # Event handlers
    dev_options_btn.click(
        fn=toggle_dev_options,
        inputs=[dev_options_visible],
        outputs=[dev_options_visible, dev_options_panel]
    )
    
    save_btn.click(
        fn=save_api_key,
        inputs=[api_key_input, api_key_state],
        outputs=[api_key_state, save_status, dev_options_panel]
    )
    
    close_btn.click(
        fn=close_panel,
        outputs=[dev_options_panel, save_status]
    )
    
    # Use the state value for the API key
    api_key = api_key_state

    # -------------------- INPUTS --------------------
    topic = gr.Textbox(
        label="Topic",
        placeholder="e.g. AI model evaluation in finance"
    )

    with gr.Accordion("Personalization Settings", open=True):
        persona = gr.Dropdown(persona_options, label="Persona", value="default")
        tone = gr.Dropdown(tone_options, label="Tone", value="neutral")
        length = gr.Dropdown(length_options, label="Length", value="medium")
        technicality = gr.Dropdown(technicality_options, label="Technicality", value="medium")
    
    with gr.Accordion("Research Settings", open=True):
        max_items = gr.Slider(
            label="Maximum Items to Fetch",
            minimum=5,
            maximum=1000,
            value=100,
            step=5,
            info="💡 5-50: Quick (1-2 min) | 100-200: Standard (3-5 min) | 500+: Deep dive (10+ min) | 1000: Unlimited"
        )
        
        gr.Markdown("""
        **How many items should I choose?**
        - **5-25**: Lightning fast preview, good for quick checks
        - **50-100**: Balanced research, covers most major sources (⭐ recommended)
        - **200-500**: Comprehensive analysis, great for deep research
        - **1000**: No limits, fetches everything available (may be slow)
        """)

    run_btn = gr.Button("🚀 Generate Summary", variant="primary")

    # -------------------- OUTPUTS --------------------
    brief_output = gr.Markdown(label="Summary Output")

    saved_path_output = gr.Textbox(
        label="Saved File",
        interactive=False
    )

    with gr.Accordion("Debug Info & Statistics", open=False):
        debug_output = gr.Textbox(
            label="Debug Information",
            lines=15,
            interactive=False
        )

    # Button click handler
    run_btn.click(
        fn=run_agent,
        inputs=[topic, persona, tone, length, technicality, max_items, api_key_state],
        outputs=[brief_output, saved_path_output, debug_output]
    )



# -------------------------------------------------------
#  Launch
# -------------------------------------------------------

if __name__ == "__main__":
    demo.launch()