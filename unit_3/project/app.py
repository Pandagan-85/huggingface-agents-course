from memory import alfred_memory
from retriever import get_guest_info_tool, ensure_guest_data, get_vector_store
from tools import get_all_tools  # ← IMPORTA TUTTI I TOOLS INCLUSO WEATHER
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
import warnings

# Carica le variabili d'ambiente
load_dotenv()

# Sopprimi warnings deprecation di LangChain
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain")


def preload_all_systems():
    """
    Pre-carica tutti i sistemi all'avvio per evitare ritardi durante l'uso.
    """
    print("🎩 Pre-caricamento Sistemi Alfred...")
    print("=" * 50)

    try:
        # 1. Pre-carica dataset
        print("📊 Pre-caricamento dataset ospiti...")
        documents = ensure_guest_data()
        print(f"✅ Dataset caricato: {len(documents)} ospiti")

        # 2. Pre-carica vector store
        print("🗃️ Pre-caricamento vector store...")
        vector_store = get_vector_store()
        print("✅ Vector store pronto")

        # 3. Pre-carica tools
        print("🔧 Pre-caricamento tools...")
        all_tools = get_all_tools()  # ← CARICA TUTTI I TOOLS
        print(f"✅ Tools pronti: {len(all_tools)} tools disponibili")

        # Mostra quali tools sono disponibili
        tool_names = [tool.name for tool in all_tools]
        print(f"🛠️ Tools attivi: {', '.join(tool_names)}")

        # 4. Test rapido del sistema
        print("🧪 Test sistema...")
        guest_tool = get_guest_info_tool(use_semantic=True)
        test_result = guest_tool.invoke({"query": "Ada"})
        if "❌" not in test_result and "Nessun" not in test_result:
            print("✅ Sistema funzionante")
        else:
            print("⚠️ Sistema ha problemi")

        print("🎉 Pre-caricamento completato!")
        print("=" * 50)
        return True

    except Exception as e:
        print(f"❌ Errore pre-caricamento: {e}")
        return False


def create_agent_with_memory(use_semantic=True):
    """
    Crea l'agente Alfred con memoria conversazionale e TUTTI i tools.
    """
    print("🎩 Inizializzazione Alfred - Assistente Sofisticato")

    # Configura l'LLM
    print("🔗 Connessione a HuggingFace...")
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.2,
        max_new_tokens=512
    )

    chat = ChatHuggingFace(llm=llm, verbose=False)

    # ← QUESTO È IL PUNTO CHIAVE: USA TUTTI I TOOLS
    tools = get_all_tools()  # Ora include: guest_info, web_search, combined_search, weather
    print(f"🛠️ Tools configurati: {[tool.name for tool in tools]}")

    chat_with_tools = chat.bind_tools(tools)
    print("✅ LLM configurato!")

    # Definisci lo stato dell'agente con memoria
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        memory_context: str

    def assistant(state: AgentState):
        """
        Assistente migliorato che usa la memoria conversazionale.
        """
        messages = state["messages"]

        # Ottieni il contesto dalla memoria se disponibile
        memory_context = alfred_memory.get_relevant_context(
            messages[-1].content)

        # Crea un messaggio di sistema con il contesto
        enhanced_messages = []
        if memory_context:
            system_message = SystemMessage(content=f"""Sei Alfred, un assistente sofisticato per un gala di beneficenza. 
            
Contesto della conversazione precedente:
{memory_context}

Hai accesso a diversi strumenti:
- Informazioni sugli ospiti del gala
- Ricerca web per informazioni aggiornate  
- Informazioni meteorologiche per pianificare l'evento

Usa questi strumenti per fornire risposte complete e utili.
Mantieni sempre un tono elegante e professionale.""")
            enhanced_messages.append(system_message)

        enhanced_messages.extend(messages)

        # Genera la risposta
        response = chat_with_tools.invoke(enhanced_messages)

        # Salva l'interazione nella memoria
        human_msg = messages[-1].content
        ai_response = response.content
        alfred_memory.add_interaction(human_msg, ai_response)

        # Estrai e salva informazioni sui guest menzionati
        guest_names = alfred_memory.extract_guest_names(
            human_msg + " " + ai_response)
        for guest_name in guest_names:
            alfred_memory.update_guest_context(
                guest_name, {"discussed_at": "now"})

        return {
            "messages": [response],
            "memory_context": memory_context
        }

    # Costruisci il grafo
    print("🔨 Costruzione del grafo dell'agente...")
    builder = StateGraph(AgentState)

    # Definisci i nodi
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))  # ← USA TUTTI I TOOLS

    # Definisci gli archi
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    alfred = builder.compile()
    print("✅ Alfred pronto con tutti i tools!")

    return alfred


def interactive_session():
    """
    Sessione interattiva con Alfred ottimizzata.
    """
    # Pre-carica tutto prima di iniziare
    if not preload_all_systems():
        print("❌ Errore nel pre-caricamento. Uscita.")
        return

    # Ora crea l'agente (veloce perché tutto è pre-caricato)
    alfred = create_agent_with_memory(use_semantic=True)

    print("\n🎩 Alfred è pronto! Comandi disponibili:")
    print("• 'quit' / 'exit' - Esci")
    print("• 'memory' - Mostra riassunto memoria")
    print("• 'stats' - Statistiche sessione")
    print("• 'tools' - Mostra tools disponibili")
    print("\n💡 Prova queste domande:")
    print("  - 'Tell me about Ada Lovelace'")
    print("  - 'What's the weather like in London?'")
    print("  - 'Who is good at mathematics?'")
    print("  - 'Should we have the reception outdoors?'")
    print("=" * 60)

    interaction_count = 0

    while True:
        user_input = input("\n👤 Tu: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print(
                f"👋 Sessione completata! Interazioni totali: {interaction_count}")
            print("È stato un piacere assisterti! Arrivederci!")
            break

        if user_input.lower() == 'memory':
            print(f"🧠 {alfred_memory.get_memory_summary()}")
            continue

        if user_input.lower() == 'stats':
            print(f"📊 Statistiche sessione:")
            print(f"  • Interazioni: {interaction_count}")
            print(f"  • Guest discussi: {len(alfred_memory.guest_context)}")
            print(
                f"  • Conversazioni in memoria: {len(alfred_memory.conversation_history)}")
            continue

        if user_input.lower() == 'tools':
            tools = get_all_tools()
            print(f"🛠️ Tools disponibili ({len(tools)}):")
            for i, tool in enumerate(tools, 1):
                print(f"  {i}. {tool.name}: {tool.description}")
            continue

        try:
            interaction_count += 1
            messages = [HumanMessage(content=user_input)]

            # Misura tempo di risposta
            import time
            start_time = time.time()

            response = alfred.invoke({
                "messages": messages,
                "memory_context": ""
            })

            response_time = time.time() - start_time

            print(f"\n🎩 Alfred: {response['messages'][-1].content}")
            print(f"⏱️ Tempo risposta: {response_time:.2f}s")

        except Exception as e:
            print(f"❌ Errore: {e}")


def test_all_tools():
    """
    Test di tutti i tools incluso weather.
    """
    # Pre-carica sistemi
    if not preload_all_systems():
        print("❌ Errore nel pre-caricamento per test.")
        return

    alfred = create_agent_with_memory(use_semantic=True)

    print("🧪 Test di tutti i tools:")
    print("=" * 60)

    # Sequenza di test per dimostrare tutti i tools
    test_sequence = [
        "Tell me about Ada Lovelace.",
        "What's the weather like in London?",
        "Who else is good at mathematics?",
        "Should we plan an outdoor reception based on the weather?",
        "Search the web for information about Marie Curie's latest work."
    ]

    for i, query in enumerate(test_sequence, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)

        try:
            import time
            start_time = time.time()

            messages = [HumanMessage(content=query)]
            response = alfred.invoke({
                "messages": messages,
                "memory_context": ""
            })

            response_time = time.time() - start_time

            print(f"🎩 Alfred: {response['messages'][-1].content}")
            print(f"⏱️ Tempo: {response_time:.2f}s")

        except Exception as e:
            print(f"❌ Errore: {e}")

        print("\n" + "=" * 60)

    print("✅ Test di tutti i tools completati!")
    print(f"📊 {alfred_memory.get_memory_summary()}")


def main():
    """
    Funzione principale ottimizzata con tutti i tools.
    """
    print("🎩 Alfred - Assistente Gala Completo")
    print("=" * 60)
    print("Scegli un'opzione:")
    print("1. Sessione interattiva (consigliato)")
    print("2. Test di tutti i tools")
    print("3. Solo pre-caricamento sistemi")
    print("4. Export dataset CSV")

    choice = input("\nScelta (1-4): ")

    if choice == "1":
        interactive_session()
    elif choice == "2":
        test_all_tools()
    elif choice == "3":
        preload_all_systems()
        print("✅ Pre-caricamento completato!")
    elif choice == "4":
        print("📊 Avvio export CSV...")
        os.system("python export_dataset.py")
    else:
        print("❌ Scelta non valida!")


if __name__ == "__main__":
    main()
