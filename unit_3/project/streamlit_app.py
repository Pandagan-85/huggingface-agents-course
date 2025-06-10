"""
Alfred for Hugging Face Spaces - STREAMLIT VERSION
Mantiene tutto il sistema LangGraph originale
"""

import streamlit as st
import os
from typing import List, Tuple
import time
from datetime import datetime

# Import il tuo sistema Alfred COMPLETO
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
    st.error(f"⚠️ Alfred tools not available: {e}")
    ALFRED_AVAILABLE = False

# Configurazione pagina
st.set_page_config(
    page_title="🎩 Alfred - Gala Assistant",
    page_icon="🎩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
st.markdown("""
<style>
.main-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
}

.chat-message {
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 10px;
    border-left: 4px solid #667eea;
}

.user-message {
    background-color: #f0f2f6;
    border-left-color: #667eea;
}

.alfred-message {
    background-color: #e8f4f8;
    border-left-color: #764ba2;
}

.status-box {
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}

.status-active { background-color: #d4edda; border-left: 4px solid #28a745; }
.status-loading { background-color: #fff3cd; border-left: 4px solid #ffc107; }
.status-error { background-color: #f8d7da; border-left: 4px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

# Inizializzazione Alfred


@st.cache_resource
def initialize_alfred():
    """Inizializza Alfred con caching."""
    if not ALFRED_AVAILABLE:
        return None, "Alfred components not available"

    try:
        # Pre-carica dati
        documents = ensure_guest_data()
        vector_store = get_vector_store()

        # Configura LLM
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            return None, "HuggingFace token not found"

        llm = HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            huggingfacehub_api_token=hf_token,
            temperature=0.2,
            max_new_tokens=400
        )

        chat = ChatHuggingFace(llm=llm, verbose=False)

        # Carica tools
        tools = get_all_tools()
        chat_with_tools = chat.bind_tools(tools)

        # Stato agente
        class AgentState(TypedDict):
            messages: Annotated[list[AnyMessage], add_messages]
            memory_context: str

        def assistant(state: AgentState):
            """Assistente Alfred."""
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

        # Costruisci grafo
        builder = StateGraph(AgentState)
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(tools))
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        alfred_agent = builder.compile()

        return alfred_agent, f"Alfred ready with {len(tools)} tools and {len(documents)} guests"

    except Exception as e:
        return None, f"Initialization error: {str(e)}"


# Inizializza stato sessione
if "messages" not in st.session_state:
    st.session_state.messages = []

if "alfred_agent" not in st.session_state:
    with st.spinner("🎩 Initializing Alfred..."):
        agent, status = initialize_alfred()
        st.session_state.alfred_agent = agent
        st.session_state.alfred_status = status

# Header principale
st.markdown("""
<div class="main-header">
    <h1>🎩 Alfred - Sophisticated Gala Assistant</h1>
    <p><em>Your elegant AI companion for the most sophisticated gala of the century</em></p>
</div>
""", unsafe_allow_html=True)

# Layout principale
col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("### 📊 Alfred Status")

    if st.session_state.alfred_agent:
        st.markdown(f"""
        <div class="status-box status-active">
            <strong>🟢 Alfred Active</strong><br>
            {st.session_state.alfred_status}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="status-box status-error">
            <strong>🔴 Alfred Error</strong><br>
            {st.session_state.alfred_status}
        </div>
        """, unsafe_allow_html=True)

    # Memoria
    try:
        memory_info = alfred_memory.get_memory_summary()
        st.markdown(f"""
        **🧠 Memory Status:**
        ```
        {memory_info}
        ```
        """)
    except:
        st.markdown("🧠 Memory: Ready")

    # Esempi
    st.markdown("### ✨ Quick Examples")
    examples = [
        "Tell me about Ada Lovelace with conversation starters",
        "What's the latest AI industry news?",
        "Show me Microsoft's top models on Hugging Face",
        "What's the weather for our outdoor reception?",
        "Get breaking tech news for discussions"
    ]

    for example in examples:
        if st.button(example, key=f"ex_{hash(example)}", use_container_width=True):
            st.session_state.messages.append(
                {"role": "user", "content": example})
            st.rerun()

with col1:
    st.markdown("### 💬 Conversation")

    # Mostra messaggi
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message alfred-message">
                <strong>🎩 Alfred:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)

    # Input utente
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask Alfred anything about the gala...",
            placeholder="Ask about guests, AI news, weather, or Hugging Face statistics...",
            height=100
        )

        col_a, col_b = st.columns([1, 4])
        with col_a:
            submit = st.form_submit_button("🎩 Send", use_container_width=True)

    # Processa input
    if submit and user_input and st.session_state.alfred_agent:
        # Aggiungi messaggio utente
        st.session_state.messages.append(
            {"role": "user", "content": user_input})

        with st.spinner("🎩 Alfred is thinking..."):
            try:
                # Invoca Alfred
                messages = [HumanMessage(content=user_input)]

                start_time = time.time()
                response = st.session_state.alfred_agent.invoke({
                    "messages": messages,
                    "memory_context": ""
                })
                response_time = time.time() - start_time

                alfred_response = response['messages'][-1].content

                # Formatta risposta
                timestamp = datetime.now().strftime("%H:%M")
                formatted_response = f"{alfred_response}\n\n*⏱️ {response_time:.1f}s • {timestamp}*"

                # Aggiungi risposta
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": formatted_response
                })

            except Exception as e:
                error_msg = f"❌ I apologize for the technical difficulty: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

        st.rerun()

    elif submit and user_input and not st.session_state.alfred_agent:
        st.error("Alfred is not available. Please check the system status.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <strong>🎩 Alfred</strong> • <em>Built with LangGraph, advanced AI tools & sophisticated conversation intelligence</em><br>
    <em>Ensuring your gala is remembered as the most sophisticated event of the century!</em>
</div>
""", unsafe_allow_html=True)

# Auto-refresh status (opzionale)
if st.button("🔄 Refresh Status", key="refresh"):
    st.rerun()
