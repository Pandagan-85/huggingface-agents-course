import os
from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool, FinalAnswerTool, Tool, tool, LiteLLMModel
import gradio as gr


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the party.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.
    Args:
        query: A search term for finding catering services.
    """
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }
    best_service = max(services, key=services.get)
    return best_service


class SuperheroPartyThemeTool(Tool):
    name = "superhero_party_theme_generator"
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic gotham').",
        }
    }
    output_type = "string"

    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }
        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic gotham'.")


# Crea l'agente
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu,
        catering_service_tool,
        SuperheroPartyThemeTool(),
        FinalAnswerTool()
    ],
    model=LiteLLMModel(
        model_id="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY")
    ),
    max_steps=10,
    verbosity_level=1  # Riduci per il push
)

# Interfaccia Gradio per lo Space


def run_agent(query):
    try:
        result = agent.run(query)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# Per l'esecuzione su Spaces
demo = gr.Interface(
    fn=run_agent,
    inputs=gr.Textbox(label="Ask Alfred",
                      placeholder="What can I help you with for your party?"),
    outputs=gr.Textbox(label="Alfred's Response"),
    title="Alfred - Batman's Butler AI",
    description="Ask Alfred to help plan your superhero party!"
)

if __name__ == "__main__":
    # METODO PIÙ SEMPLICE: usa il push originale ma senza eseguire prima l'agente
    try:
        agent.push_to_hub(
            'pandagan/AlfredAgent',
            commit_message="Add Alfred Butler Agent",
            # Assicurati che non provi a eseguire l'agente durante il push
        )
        print("Successfully pushed to Hub!")
    except Exception as e:
        print(f"Push failed: {e}")

    # Lancia l'interfaccia solo in locale
    if os.getenv("SPACE_ID") is None:
        demo.launch()
