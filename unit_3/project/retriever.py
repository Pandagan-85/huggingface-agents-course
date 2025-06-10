"""
Enhanced retriever system for Alfred - Versione Ottimizzata
Fixes deprecation warnings e implementa pre-caricamento
"""

from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS

# Fix per deprecation warning - usa la nuova importazione
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ Usando HuggingFaceEmbeddings aggiornato")
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("⚠️ Usando HuggingFaceEmbeddings legacy")

import datasets
from langchain_core.documents import Document
import os
import pickle
import re
import warnings
from typing import List, Dict, Tuple, Optional

# Sopprimi warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configurazione
DATA_DIR = "./data"
FAISS_DIR = f"{DATA_DIR}/faiss_index"
ENHANCED_DATA_DIR = f"{DATA_DIR}/enhanced_guests"

# Cache globale
_vector_store = None
_guest_documents = None
_search_tool = None
_embeddings = None
_system_ready = False

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ENHANCED_DATA_DIR, exist_ok=True)


def get_embeddings():
    """Ottieni embeddings sentence-transformers ottimizzati."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _embeddings


def ensure_guest_data():
    """Carica e arricchisce i dati dei guest."""
    global _guest_documents

    enhanced_path = f"{ENHANCED_DATA_DIR}/enhanced_guests.pkl"

    # Controlla se abbiamo dati arricchiti
    if os.path.exists(enhanced_path) and _guest_documents is None:
        with open(enhanced_path, 'rb') as f:
            _guest_documents = pickle.load(f)
        return _guest_documents

    # Altrimenti carica e arricchisci
    if _guest_documents is None:
        guest_dataset = datasets.load_dataset(
            "agents-course/unit3-invitees", split="train")

        _guest_documents = []
        for guest in guest_dataset:
            # Arricchimento con analisi intelligente
            interests, topics = extract_interests_and_topics(
                guest['description'])
            conversation_starters = generate_conversation_starters(
                guest, interests, topics)

            # Documento arricchito
            enhanced_content = "\n".join([
                f"Nome: {guest['name']}",
                f"Relazione: {guest['relation']}",
                f"Descrizione: {guest['description']}",
                f"Email: {guest['email']}",
                f"Interessi principali: {', '.join(interests)}",
                f"Argomenti di conversazione: {', '.join(topics)}",
                f"Conversation starters: {'; '.join(conversation_starters)}"
            ])

            doc = Document(
                page_content=enhanced_content,
                metadata={
                    "name": guest["name"],
                    "relation": guest["relation"],
                    "email": guest["email"],
                    "interests": interests,
                    "topics": topics,
                    "conversation_starters": conversation_starters
                }
            )
            _guest_documents.append(doc)

        # Salva dati arricchiti
        with open(enhanced_path, 'wb') as f:
            pickle.dump(_guest_documents, f)

    return _guest_documents


def extract_interests_and_topics(description: str) -> Tuple[List[str], List[str]]:
    """Estrae interessi e argomenti dalla descrizione del guest."""

    # Pattern per identificare interessi
    interest_patterns = [
        r'interest in (\w+)',
        r'passionate about (\w+)',
        r'expertise in (\w+)',
        r'specializes in (\w+)',
        r'works in (\w+)',
        r'studies (\w+)',
        r'focuses on (\w+)',
        r'pioneer in (\w+)'
    ]

    # Termini chiave per argomenti
    topic_keywords = {
        'technology': ['computer', 'programming', 'algorithm', 'software', 'digital', 'AI', 'machine'],
        'science': ['physics', 'chemistry', 'biology', 'research', 'experiment', 'discovery'],
        'mathematics': ['mathematical', 'calculation', 'theorem', 'equation', 'geometry'],
        'literature': ['writing', 'poetry', 'novel', 'book', 'author', 'literary'],
        'arts': ['painting', 'music', 'sculpture', 'artistic', 'creative', 'design'],
        'philosophy': ['philosophical', 'ethics', 'logic', 'thinking', 'wisdom'],
        'business': ['entrepreneurship', 'company', 'business', 'industry', 'commerce'],
        'education': ['teaching', 'university', 'academic', 'student', 'learning'],
        'innovation': ['invention', 'innovation', 'revolutionary', 'breakthrough']
    }

    interests = []
    topics = []

    description_lower = description.lower()

    # Estrai interessi con pattern
    for pattern in interest_patterns:
        matches = re.findall(pattern, description_lower)
        interests.extend(matches)

    # Identifica argomenti per parole chiave
    for topic, keywords in topic_keywords.items():
        if any(keyword in description_lower for keyword in keywords):
            topics.append(topic)

    # Pulizia e deduplica
    interests = list(set([i.capitalize() for i in interests if len(i) > 2]))
    topics = list(set(topics))

    # Se non troviamo nulla, aggiungi topics generici
    if not interests:
        interests = ["General topics", "Current events"]
    if not topics:
        topics = ["conversation", "networking"]

    return interests[:5], topics[:3]


def generate_conversation_starters(guest: Dict, interests: List[str], topics: List[str]) -> List[str]:
    """Genera conversation starters personalizzati per il guest."""

    name = guest['name']
    relation = guest['relation']
    description = guest['description']

    starters = []

    # Starters basati su interessi specifici
    if 'technology' in topics or 'computer' in description.lower():
        starters.extend([
            f"Ho sentito che {name} ha esperienza in tecnologia. Quali innovazioni la entusiasmano di più?",
            f"Come vede l'evoluzione tecnologica nel suo campo di expertise?",
            f"Quali sfide tecnologiche ritiene più interessanti da risolvere?"
        ])

    if 'science' in topics or 'research' in description.lower():
        starters.extend([
            f"Le ricerche di {name} sono affascinanti. Su cosa sta lavorando attualmente?",
            f"Quale scoperta scientifica recente trova più promettente?",
            f"Come vede il futuro della ricerca nel suo settore?"
        ])

    if 'mathematics' in topics or 'mathematical' in description.lower():
        starters.extend([
            f"La matematica dietro il lavoro di {name} è impressionante. Come applica questi concetti?",
            f"Quali sono le applicazioni pratiche più entusiasmanti della matematica oggi?",
            f"Come spiega concetti matematici complessi a un pubblico generale?"
        ])

    if 'literature' in topics or 'writing' in description.lower():
        starters.extend([
            f"Le opere di {name} sono molto apprezzate. Cosa la ispira di più nella scrittura?",
            f"Quale autore contemporaneo trova più influente?",
            f"Come vede l'evoluzione della letteratura nell'era digitale?"
        ])

    # Starters generici basati sulla relazione
    relation_starters = {
        'friend': [
            f"Come avete iniziato la vostra amicizia con l'organizzatore?",
            f"Quali sono i ricordi più belli che condividete?"
        ],
        'colleague': [
            f"In che progetti avete collaborato insieme?",
            f"Come descrivereste il vostro ambiente di lavoro?"
        ],
        'university': [
            f"Che ricordi avete dell'università?",
            f"Quali erano le vostre materie preferite?"
        ]
    }

    # Aggiungi starters basati sulla relazione
    for key, rel_starters in relation_starters.items():
        if key in relation.lower():
            starters.extend(rel_starters)

    # Starters universali di backup
    universal_starters = [
        f"Cosa la ha portata a interessarsi al lavoro che fa?",
        f"Quali progetti la entusiasmano di più in questo momento?",
        f"Come bilancia vita professionale e personale?",
        f"Quali consigli darebbe a giovani interessati al suo campo?",
        f"Quale libro o risorsa consiglierebbe per capire meglio il suo lavoro?"
    ]

    starters.extend(universal_starters)

    # Rimuovi duplicati e limita
    starters = list(dict.fromkeys(starters))
    return starters[:8]


def get_vector_store():
    """Crea vector store FAISS ottimizzato."""
    global _vector_store

    if _vector_store is None:
        if os.path.exists(FAISS_DIR):
            try:
                _vector_store = FAISS.load_local(
                    FAISS_DIR,
                    get_embeddings(),
                    allow_dangerous_deserialization=True
                )
            except Exception:
                _vector_store = None

        if _vector_store is None:
            documents = ensure_guest_data()
            _vector_store = FAISS.from_documents(
                documents=documents,
                embedding=get_embeddings()
            )
            _vector_store.save_local(FAISS_DIR)

    return _vector_store


@tool
def guest_info_retriever(query: str) -> str:
    """
    Recupera informazioni complete sui guest con conversation starters.

    Args:
        query: Nome, interesse o caratteristica del guest

    Returns:
        Informazioni dettagliate e conversation starters personalizzati
    """
    try:
        vector_store = get_vector_store()

        # Ricerca semantica multi-livello
        results = vector_store.similarity_search_with_score(query, k=3)

        if not results:
            return f"❌ Nessuna informazione trovata per: '{query}'"

        formatted_results = []

        for i, (doc, score) in enumerate(results):
            similarity = 1 / (1 + score)

            if similarity >= 0.3:  # Soglia di rilevanza
                # Estrai metadati arricchiti
                name = doc.metadata.get('name', 'Guest sconosciuto')
                relation = doc.metadata.get('relation', 'N/A')
                interests = doc.metadata.get('interests', [])
                starters = doc.metadata.get('conversation_starters', [])

                result_text = f"""
🎭 **{name}** (Rilevanza: {similarity:.3f})
📧 Email: {doc.metadata.get('email', 'N/A')}
🤝 Relazione: {relation}

📝 **Informazioni:**
{doc.page_content.split('Conversation starters:')[0].strip()}

💡 **Conversation Starters:**"""

                # Aggiungi conversation starters numerati
                for j, starter in enumerate(starters[:5], 1):
                    result_text += f"\n   {j}. {starter}"

                if interests:
                    result_text += f"\n\n🎯 **Argomenti consigliati:** {', '.join(interests)}"

                formatted_results.append(result_text)

        if formatted_results:
            header = f"🔍 **Risultati per '{query}':**\n" + "="*50
            return header + "\n" + "\n\n".join(formatted_results)
        else:
            return f"❌ Nessun guest rilevante trovato per: '{query}'"

    except Exception as e:
        return f"❌ Errore nel retrieval avanzato: {str(e)}"


@tool
def web_search_guests(query: str) -> str:
    """Cerca informazioni aggiuntive sui guest sul web."""
    global _search_tool

    if _search_tool is None:
        try:
            wrapper = DuckDuckGoSearchAPIWrapper(
                max_results=3,
                region="en-us",
                safesearch="moderate"
            )
            _search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        except Exception as e:
            return f"❌ Ricerca web non disponibile: {e}"

    try:
        results = _search_tool.run(f'"{query}" biography achievements')
        if len(results) > 800:
            results = results[:800] + "..."

        return f"🌐 **Informazioni aggiuntive dal web:**\n{results}"

    except Exception as e:
        return f"❌ Errore ricerca web: {str(e)}"


@tool
def combined_guest_search(query: str) -> str:
    """
    Ricerca combinata: database locale + web + conversation starters.

    Args:
        query: Nome o caratteristica del guest

    Returns:
        Informazioni complete da tutte le fonti disponibili
    """
    results = []

    # 1. Ricerca nel database locale - USA .invoke() invece di ()
    local_results = guest_info_retriever.invoke({"query": query})
    results.append(f"📚 **Database Locale:**\n{local_results}")

    # 2. Se necessario, cerca sul web
    if "Nessun" in local_results or "❌" in local_results:
        web_results = web_search_guests.invoke({"query": query})
        results.append(f"\n{web_results}")

        # 3. Genera conversation starters generici se non trovati
        generic_starters = [
            f"Ho sentito parlare molto bene di lei. Può raccontarmi del suo lavoro?",
            f"Quale aspetto del suo campo trova più entusiasmante?",
            f"Come è arrivata a interessarsi a questo settore?",
            f"Quali sono i progetti che la appassionano di più?"
        ]

        starter_text = "\n💡 **Conversation Starters Generici:**"
        for i, starter in enumerate(generic_starters, 1):
            starter_text += f"\n   {i}. {starter}"

        results.append(starter_text)

    return "\n\n".join(results)

# Funzioni di compatibilità per app.py esistente


def load_guest_dataset_semantic():
    """Compatibilità: restituisce il tool avanzato."""
    return guest_info_retriever


def load_guest_dataset_bm25():
    """Compatibilità: usa sistema avanzato invece di BM25."""
    return guest_info_retriever


def get_guest_info_tool(use_semantic=True):
    """Compatibilità: restituisce il tool appropriato."""
    return combined_guest_search if use_semantic else guest_info_retriever

# API per ottenere tutti i tools


def get_all_tools():
    """Tutti i tools disponibili per Alfred."""
    return [guest_info_retriever, web_search_guests, combined_guest_search]


def get_basic_tools():
    """Tools di base senza ricerca web."""
    return [guest_info_retriever]


def get_enhanced_tools():
    """Tools avanzati con tutte le funzionalità."""
    return [combined_guest_search]


def initialize_system():
    """Inizializza tutto il sistema in una volta."""
    global _system_ready

    if not _system_ready:
        print("🎩 Inizializzazione sistema Alfred...")

        # Pre-carica tutto
        ensure_guest_data()
        get_vector_store()

        _system_ready = True
        print("✅ Sistema Alfred pronto!")

    return _system_ready

# Funzione di test


def test_enhanced_retrieval():
    """Test del sistema di retrieval avanzato."""
    print("🧪 Test sistema di retrieval avanzato...")

    # Inizializza sistema
    initialize_system()

    test_queries = [
        "Ada Lovelace",
        "mathematician",
        "technology expert",
        "writer"
    ]

    for query in test_queries:
        print(f"\n🔍 Test: '{query}'")
        result = guest_info_retriever.invoke({"query": query})
        print(f"Risultato: {result[:200]}...")


if __name__ == "__main__":
    test_enhanced_retrieval()
