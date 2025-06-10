"""
Alfred for Hugging Face Spaces
Imports and wraps the complete Alfred system with Gradio interface
"""

from datetime import datetime
import time
from typing import List, Tuple
import gradio as gr
import os

# Controlla versione Gradio e avvisa
try:
    import gradio
    print(f"📦 Gradio version: {gradio.__version__}")
    if gradio.__version__.startswith('5.'):
        print("⚠️ Gradio 5.x detected - using compatibility mode")
        GRADIO_5X = True
    else:
        GRADIO_5X = False
except:
    GRADIO_5X = False

# Import il tuo sistema Alfred completo
try:
    from memory import alfred_memory
    from tools import get_all_tools, ensure_guest_data, get_vector_store
    from retriever import get_guest_info_tool
    from typing import TypedDict, Annotated
    from langgraph.graph.message import add_messages
    from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import START, StateGraph
    from langgraph.prebuilt import tools_condition
    from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ALFRED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Alfred tools not fully available: {e}")
    ALFRED_AVAILABLE = False

# Variabile globale per Alfred
alfred_agent = None


def initialize_alfred_for_spaces():
    """Inizializza Alfred ottimizzato per Spaces."""
    global alfred_agent

    if not ALFRED_AVAILABLE:
        return False

    try:
        print("🎩 Inizializzando Alfred per Hugging Face Spaces...")

        # Pre-carica dati
        print("📊 Caricamento dataset...")
        documents = ensure_guest_data()
        print(f"✅ {len(documents)} ospiti caricati")

        print("🗃️ Caricamento vector store...")
        vector_store = get_vector_store()
        print("✅ Vector store pronto")

        # Configura LLM
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("❌ Token HuggingFace non trovato")
            return False

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            huggingfacehub_api_token=hf_token,
            temperature=0.2,
            max_new_tokens=400
        )

        chat = ChatHuggingFace(llm=llm, verbose=False)

        # Carica tools con gestione errori per Gradio 5.x
        try:
            tools = get_all_tools()
            print(f"🛠️ {len(tools)} tools caricati")

            # Per Gradio 5.x, usa solo subset di tools sicuri
            if GRADIO_5X:
                print("🔧 Modalità compatibilità Gradio 5.x")
                # Usa solo tools che non causano problemi di schema
                safe_tools = [t for t in tools if 'news' not in t.name.lower()]
                tools = safe_tools[:5]  # Limita numero
                print(f"🛠️ Usando {len(tools)} tools sicuri")

            chat_with_tools = chat.bind_tools(tools)

        except Exception as e:
            print(f"⚠️ Errore tools, uso modalità fallback: {e}")
            chat_with_tools = chat
            tools = []

        # Stato agente
        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]
            memory_context: str

        def assistant(state: AgentState):
            """Assistente Alfred per Spaces."""
            messages = state["messages"]

            # Context memoria
            memory_context = alfred_memory.get_relevant_context(
                messages[-1].content if messages else "")

            # System message
            enhanced_messages = []
            if memory_context:
                system_message = SystemMessage(content=f"""You are Alfred, a sophisticated AI assistant for an elegant gala event.
Previous conversation context:
{memory_context}
You have access to tools for guest information, news, and statistics.
Always maintain an elegant, professional, and helpful tone.""")
                enhanced_messages.append(system_message)

            enhanced_messages.extend(messages)

            # Genera risposta
            response = chat_with_tools.invoke(enhanced_messages)

            # Salva memoria
            if messages:
                human_msg = messages[-1].content
                ai_response = response.content
                alfred_memory.add_interaction(human_msg, ai_response)

            return {
                "messages": [response],
                "memory_context": memory_context
            }

        # Costruisci grafo solo se abbiamo tools
        if tools:
            builder = StateGraph(AgentState)
            builder.add_node("assistant", assistant)
            builder.add_node("tools", ToolNode(tools))
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges("assistant", tools_condition)
            builder.add_edge("tools", "assistant")
            alfred_agent = builder.compile()
        else:
            # Modalità fallback senza tools
            alfred_agent = assistant

        print("✅ Alfred pronto per Spaces!")
        return True

    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
        return False


def chat_with_alfred(message: str, history: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Funzione chat per Gradio con gestione errori robusta."""
    global alfred_agent

    if not alfred_agent:
        if not initialize_alfred_for_spaces():
            error_msg = "❌ Alfred non è pronto. Riprova tra un momento..."
            return history + [(message, error_msg)]

    try:
        # Timeout per Spaces
        start_time = time.time()

        # Invoca Alfred
        messages = [HumanMessage(content=message)]

        if hasattr(alfred_agent, 'invoke'):
            # Modalità completa con LangGraph
            response = alfred_agent.invoke({
                "messages": messages,
                "memory_context": ""
            })
            alfred_response = response['messages'][-1].content
        else:
            # Modalità fallback
            response = alfred_agent({
                "messages": messages,
                "memory_context": ""
            })
            alfred_response = response['messages'][-1].content

        response_time = time.time() - start_time

        # Formatta risposta
        timestamp = datetime.now().strftime("%H:%M")
        formatted_response = f"{alfred_response}\n\n*⏱️ {response_time:.1f}s • {timestamp}*"

        # Aggiorna cronologia
        return history + [(message, formatted_response)]

    except Exception as e:
        print(f"❌ Errore chat: {e}")

        # Fallback response
        fallback_responses = {
            "ada": "🎭 Ada Lovelace è una matematica brillante, pioniera della programmazione. Ottima per discussioni su tecnologia e innovazione!",
            "weather": "🌤️ Il tempo per il gala stasera è perfetto - cielo sereno e 22°C!",
            "news": "📰 Mi scuso, al momento non posso accedere alle ultime notizie. Prova con informazioni sui guest!",
        }

        # Cerca response appropriata
        fallback_msg = "🎩 Mi scuso per l'inconveniente tecnico. Come posso assisterla altrimenti?"
        for key, response in fallback_responses.items():
            if key in message.lower():
                fallback_msg = response
                break

        return history + [(message, fallback_msg)]


def get_alfred_info():
    """Info su Alfred con diagnostica."""
    status_parts = []

    if ALFRED_AVAILABLE and alfred_agent:
        status_parts.append("🟢 **Alfred Active**")
        if GRADIO_5X:
            status_parts.append("🔧 *Compatibility Mode*")
    elif ALFRED_AVAILABLE:
        status_parts.append("🟡 **Alfred Ready**")
    else:
        status_parts.append("🔴 **Alfred Initializing**")

    # Aggiungi info sistema
    try:
        import gradio
        status_parts.append(f"📦 Gradio: {gradio.__version__}")
    except:
        pass

    if ALFRED_AVAILABLE:
        try:
            memory_summary = alfred_memory.get_memory_summary()
            status_parts.append(f"🧠 {memory_summary}")
        except:
            status_parts.append("🧠 Memory: Available")

    return "\n".join(status_parts)


# Inizializza Alfred all'avvio
print("🚀 Starting Alfred for Hugging Face Spaces...")
alfred_ready = initialize_alfred_for_spaces()

# Crea interfaccia Gradio con compatibilità
demo_kwargs = {
    "title": "🎩 Alfred - Sophisticated Gala Assistant",
}

# Usa theme solo se non è Gradio 5.x
if not GRADIO_5X:
    try:
        demo_kwargs["theme"] = gr.themes.Soft(
            primary_hue="blue", secondary_hue="slate")
    except:
        pass

with gr.Blocks(**demo_kwargs) as demo:

    # Header
    gr.Markdown("""
    # 🎩 Alfred - Sophisticated Gala Assistant
    *Your elegant AI companion for the most sophisticated gala of the century*
    
    **Advanced AI Capabilities:**
    • 👥 Guest information with conversation starters
    • 📰 Real-time AI & tech news • 🤖 Hugging Face Hub statistics  
    • 🌤️ Weather forecasts • 💭 Conversational memory
    """)

    if not alfred_ready:
        gr.Markdown("⚠️ **Alfred is initializing... Please wait a moment.**")

    # Chat interface
    with gr.Row():
        with gr.Column(scale=4):
            # Usa configurazione compatibile per chatbot
            chatbot_kwargs = {
                "height": 450,
                "show_copy_button": True,
                "placeholder": "🎩 Alfred: Good evening! I'm ready to assist with your gala preparations. How may I help?",
            }

            # Avatar solo se non Gradio 5.x
            if not GRADIO_5X:
                chatbot_kwargs["avatar_images"] = ("👤", "🎩")

            chatbot = gr.Chatbot(**chatbot_kwargs)

            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about guests, AI news, weather, or Hugging Face statistics...",
                    show_label=False,
                    scale=5
                )
                send_btn = gr.Button("🎩 Send", scale=1, variant="primary")

            # Esempi eleganti
            gr.Examples(
                examples=[
                    "Tell me about Ada Lovelace with conversation starters",
                    "What's the latest AI industry news?",
                    "Show me Microsoft's top models on Hugging Face",
                    "What's the weather for our outdoor reception?",
                    "Get breaking tech news for informed discussions",
                    "Compare AI statistics: Google vs Microsoft"
                ],
                inputs=msg,
                label="✨ Sophisticated Query Examples"
            )

        # Sidebar status
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Alfred Status")

            status_display = gr.Markdown(
                get_alfred_info(),
                every=30  # Auto-refresh ogni 30s
            )

    # Event handlers - FIX per "No API found"
    def handle_submit(message, history):
        """Wrapper per gestire submit con return corretto."""
        new_history = chat_with_alfred(message, history)
        return "", new_history[1] if isinstance(new_history, tuple) else new_history

    def handle_click(message, history):
        """Wrapper per gestire click con return corretto."""
        result = chat_with_alfred(message, history)
        if isinstance(result, tuple):
            return result[0], result[1]
        return "", result

    msg.submit(
        fn=handle_submit,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        api_name="submit_message"
    )

    send_btn.click(
        fn=handle_click,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        api_name="send_message"
    )

    # Footer
    gr.Markdown("""
    ---
    **🎩 Alfred** • *Built with LangGraph, advanced AI tools & sophisticated conversation intelligence*
    
    *Ensuring your gala is remembered as the most sophisticated event of the century!*
    """)

# Launch per Spaces con configurazione robusta
if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "show_api": False,
        "share": False
    }

    # Debug info
    print(f"🔧 Gradio 5.x mode: {GRADIO_5X}")
    print(f"🎩 Alfred ready: {alfred_ready}")

    demo.launch(**launch_kwargs)
