from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
import os
from retriever import load_guest_dataset_semantic, load_guest_dataset_bm25, get_guest_info_tool
from memory import alfred_memory

# Carica le variabili d'ambiente
load_dotenv()


def create_agent_with_memory(use_semantic=True):
    """
    Crea l'agente Alfred con memoria conversazionale.
    """
    print("🎩 Creazione di Alfred con Memoria - Assistente Sofisticato")
    print("=" * 60)

    # Carica il tool appropriato
    guest_info_tool = get_guest_info_tool(use_semantic)

    if use_semantic:
        print("🧠 Usando retriever semantico con memoria...")
    else:
        print("📝 Usando retriever BM25 con memoria...")

    # Configura l'LLM
    print("🔗 Connessione a HuggingFace...")
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature=0.2,
        max_new_tokens=512
    )

    chat = ChatHuggingFace(llm=llm, verbose=False)
    tools = [guest_info_tool]
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

Usa questo contesto per fornire risposte più personalizzate e coerenti. 
Se l'utente si riferisce a qualcosa menzionato prima, puoi fare riferimento a quella conversazione.
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
    print("🔨 Costruzione del grafo dell'agente con memoria...")
    builder = StateGraph(AgentState)

    # Definisci i nodi
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # Definisci gli archi
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    alfred = builder.compile()
    print("✅ Alfred con memoria è pronto!")
    print("=" * 60)

    return alfred


def interactive_session():
    """
    Sessione interattiva con Alfred che dimostra la memoria.
    """
    alfred = create_agent_with_memory(use_semantic=True)

    print("🎩 Alfred è pronto! Digita 'quit' per uscire, 'memory' per vedere il riassunto.")
    print("💡 Prova a fare domande sui guest e poi a riferirti a conversazioni precedenti!")
    print("=" * 60)

    while True:
        user_input = input("\n👤 Tu: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 È stato un piacere assisterti! Arrivederci!")
            break

        if user_input.lower() == 'memory':
            print(
                f"🧠 Riassunto memoria di Alfred:\n{alfred_memory.get_memory_summary()}")
            continue

        try:
            messages = [HumanMessage(content=user_input)]
            response = alfred.invoke({
                "messages": messages,
                "memory_context": ""
            })

            print(f"\n🎩 Alfred: {response['messages'][-1].content}")

        except Exception as e:
            print(f"❌ Errore: {e}")


def test_memory_features():
    """
    Test specifici per le funzionalità di memoria.
    """
    alfred = create_agent_with_memory(use_semantic=True)

    print("🧪 Test delle funzionalità di memoria:")
    print("=" * 60)

    # Sequenza di test per dimostrare la memoria
    test_sequence = [
        "Dimmi informazioni su Lady Ada Lovelace.",
        "Chi altro potrebbe essere interessato al suo lavoro?",
        "Ricordi di chi stavamo parlando prima?",
        "Parlami di un altro guest matematico."
    ]

    for i, query in enumerate(test_sequence, 1):
        print(f"\n🔍 Test {i}: '{query}'")
        print("-" * 40)

        try:
            messages = [HumanMessage(content=query)]
            response = alfred.invoke({
                "messages": messages,
                "memory_context": ""
            })

            print(f"🎩 Alfred: {response['messages'][-1].content}")

            # Mostra lo stato della memoria dopo ogni interazione
            if i > 1:  # Solo dopo la prima interazione
                context = alfred_memory.get_relevant_context(query)
                if context:
                    print(f"\n💭 Contesto usato: {context[:100]}...")

        except Exception as e:
            print(f"❌ Errore: {e}")

        print("\n" + "=" * 60)

    print("✅ Test memoria completati!")
    print(f"📊 {alfred_memory.get_memory_summary()}")


def main():
    """
    Funzione principale con opzioni per testare diverse funzionalità.
    """
    print("🎩 Alfred - Assistente Gala con Memoria Avanzata")
    print("=" * 60)
    print("Scegli un'opzione:")
    print("1. Sessione interattiva")
    print("2. Test funzionalità memoria")
    print("3. Test base (senza memoria)")

    choice = input("\nScelta (1-3): ")

    if choice == "1":
        interactive_session()
    elif choice == "2":
        test_memory_features()
    elif choice == "3":
        # Test originale senza memoria
        alfred = create_agent_with_memory(use_semantic=True)
        test_queries = [
            "Tell me about Lady Ada Lovelace.",
            "Who is the mathematician among our guests?"
        ]
        for query in test_queries:
            messages = [HumanMessage(content=query)]
            response = alfred.invoke(
                {"messages": messages, "memory_context": ""})
            print(f"\n🔍 Query: {query}")
            print(f"🎩 Alfred: {response['messages'][-1].content}")
    else:
        print("Scelta non valida!")


if __name__ == "__main__":
    main()
