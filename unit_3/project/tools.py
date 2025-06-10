from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import datasets
from langchain_core.documents import Document
import os
import pickle
from typing import List, Optional

# Configurazione percorsi per persistenza
DATA_DIR = "./data"
FAISS_DIR = f"{DATA_DIR}/faiss_index"

# Assicurati che le directory esistano
os.makedirs(DATA_DIR, exist_ok=True)

# Variabili globali per caching
_vector_store = None
_guest_documents = None
_search_tool = None
_embeddings = None

def get_embeddings():
    """Ottieni l'oggetto embeddings (inizializzato una sola volta)."""
    global _embeddings
    if _embeddings is None:
        print("🧠 Inizializzazione embeddings...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embeddings pronti")
    return _embeddings

def ensure_guest_data():
    """
    Assicura che i dati dei guest siano caricati e processati.
    """
    global _guest_documents
    
    dataset_path = f"{DATA_DIR}/guest_dataset.pkl"
    
    # Controlla se abbiamo già il dataset salvato
    if os.path.exists(dataset_path) and _guest_documents is None:
        print("📂 Caricamento dataset salvato...")
        with open(dataset_path, 'rb') as f:
            _guest_documents = pickle.load(f)
        print(f"✅ Dataset caricato: {len(_guest_documents)} documenti")
    
    # Se non esiste, scaricalo e salvalo
    elif _guest_documents is None:
        print("📥 Download dataset ospiti da HuggingFace...")
        guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")
        
        _guest_documents = [
            Document(
                page_content="\n".join([
                    f"Nome: {guest['name']}",
                    f"Relazione: {guest['relation']}",
                    f"Descrizione: {guest['description']}",
                    f"Email: {guest['email']}"
                ]),
                metadata={
                    "name": guest["name"],
                    "relation": guest["relation"],
                    "email": guest["email"]
                }
            )
            for guest in guest_dataset
        ]
        
        # Salva il dataset processato
        with open(dataset_path, 'wb') as f:
            pickle.dump(_guest_documents, f)
        
        print(f"💾 Dataset salvato: {len(_guest_documents)} documenti")
    
    return _guest_documents

def get_vector_store():
    """
    Ottiene il vector store FAISS, creandolo se necessario.
    """
    global _vector_store
    
    if _vector_store is None:
        print("🗄️ Inizializzazione vector store...")
        
        # Controlla se esiste un indice salvato
        if os.path.exists(FAISS_DIR):
            print("📂 Caricamento indice FAISS esistente...")
            try:
                _vector_store = FAISS.load_local(
                    FAISS_DIR, 
                    get_embeddings(),
                    allow_dangerous_deserialization=True
                )
                print("✅ Indice FAISS caricato")
            except Exception as e:
                print(f"⚠️ Errore caricamento indice: {e}")
                print("🔄 Creazione nuovo indice...")
                _vector_store = None
        
        # Se non esiste o caricamento fallito, crea nuovo indice
        if _vector_store is None:
            print("🆕 Creazione nuovo indice FAISS...")
            
            # Assicurati che i documenti siano caricati
            documents = ensure_guest_data()
            
            # Crea vector store da documenti
            _vector_store = FAISS.from_documents(
                documents=documents,
                embedding=get_embeddings()
            )
            
            # Salva l'indice
            _vector_store.save_local(FAISS_DIR)
            print(f"💾 Indice FAISS creato e salvato con {len(documents)} documenti")
    
    return _vector_store

@tool
def guest_info_retriever(query: str) -> str:
    """
    Recupera informazioni dettagliate sugli ospiti del gala usando ricerca vettoriale.
    
    Args:
        query: Il nome, relazione o descrizione dell'ospite che stai cercando
        
    Returns:
        Informazioni dettagliate sui guest trovati
    """
    vector_store = get_vector_store()
    
    # Esegui ricerca semantica con score
    results = vector_store.similarity_search_with_score(query, k=3)
    
    if results:
        formatted_results = []
        
        for i, (doc, score) in enumerate(results):
            # FAISS usa distanza, convertiamo in similarità
            similarity = 1 / (1 + score)
            
            if similarity >= 0.3:  # Soglia di rilevanza
                formatted_results.append(
                    f"Risultato {i+1} (rilevanza: {similarity:.3f}):\n{doc.page_content}"
                )
        
        if formatted_results:
            return "\n\n".join(formatted_results)
    
    return f"Nessuna informazione trovata per: '{query}'"

def get_web_search_tool():
    """
    Ottiene il tool di ricerca web, creandolo se necessario.
    """
    global _search_tool
    
    if _search_tool is None:
        print("🔍 Inizializzazione tool ricerca web...")
        try:
            wrapper = DuckDuckGoSearchAPIWrapper(
                max_results=3,
                region="it-it",
                safesearch="moderate"
            )
            _search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
            print("✅ Tool ricerca web pronto")
        except Exception as e:
            print(f"⚠️ Errore inizializzazione ricerca web: {e}")
            _search_tool = None
    
    return _search_tool

@tool
def web_search(query: str) -> str:
    """
    Cerca informazioni aggiornate sul web per guest non presenti nel database.
    
    Args:
        query: La query di ricerca (nome del guest, evento, etc.)
        
    Returns:
        Informazioni trovate sul web
    """
    search_tool = get_web_search_tool()
    
    if search_tool is None:
        return "❌ Ricerca web non disponibile al momento"
    
    try:
        # Esegui ricerca
        results = search_tool.run(query)
        
        # Aggiungi contesto e limita lunghezza
        if len(results) > 1000:
            results = results[:1000] + "..."
        
        formatted_result = f"🌐 Risultati ricerca web per '{query}':\n\n{results}"
        
        return formatted_result
        
    except Exception as e:
        return f"❌ Errore nella ricerca web: {str(e)}"

@tool
def combined_guest_search(query: str) -> str:
    """
    Combina ricerca nel database locale e ricerca web per informazioni complete sui guest.
    
    Args:
        query: Il nome o informazioni sul guest che stai cercando
        
    Returns:
        Informazioni combinate da database locale e web
    """
    # Prima cerca nel database locale
    local_results = guest_info_retriever(query)
    
    results_parts = [f"📚 Database locale:\n{local_results}"]
    
    # Se non troviamo risultati soddisfacenti localmente, cerca sul web
    if "Nessuna informazione trovata" in local_results:
        print(f"🔍 Nessun risultato locale per '{query}', provo ricerca web...")
        web_results = web_search(f'"{query}" biography gala event')
        results_parts.append(f"\n{web_results}")
    
    return "\n".join(results_parts)

# Lista di tutti i tools disponibili
def get_all_tools():
    """Restituisce tutti i tools disponibili per Alfred."""
    return [
        guest_info_retriever,
        web_search, 
        combined_guest_search
    ]

def get_basic_tools():
    """Restituisce solo i tools di base (senza ricerca web)."""
    return [guest_info_retriever]

def get_enhanced_tools():
    """Restituisce tutti i tools inclusa la ricerca web combinata."""
    return [combined_guest_search]

@tool
def guest_info_retriever(query: str) -> str:
    """
    Recupera informazioni dettagliate sugli ospiti del gala usando Chroma vector database.
    
    Args:
        query: Il nome, relazione o descrizione dell'ospite che stai cercando
        
    Returns:
        Informazioni dettagliate sui guest trovati
    """
    collection = get_chroma_collection()
    
    # Esegui ricerca semantica
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    
    if results['documents'][0]:  # Se ci sono risultati
        formatted_results = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0], 
            results['distances'][0]
        )):
            similarity = 1 - distance  # Converti distanza in similarità
            if similarity >= 0.3:  # Soglia di rilevanza
                formatted_results.append(
                    f"Risultato {i+1} (rilevanza: {similarity:.3f}):\n{doc}"
                )
        
        if formatted_results:
            return "\n\n".join(formatted_results)
    
    return f"Nessuna informazione trovata per: '{query}'"

def get_web_search_tool():
    """
    Ottiene il tool di ricerca web, creandolo se necessario.
    """
    global _search_tool
    
    if _search_tool is None:
        print("🔍 Inizializzazione tool ricerca web...")
        wrapper = DuckDuckGoSearchAPIWrapper(
            max_results=5,
            region="it-it",
            safesearch="moderate"
        )
        _search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        print("✅ Tool ricerca web pronto")
    
    return _search_tool

@tool
def web_search(query: str) -> str:
    """
    Cerca informazioni aggiornate sul web per guest non presenti nel database.
    
    Args:
        query: La query di ricerca (nome del guest, evento, etc.)
        
    Returns:
        Informazioni trovate sul web
    """
    search_tool = get_web_search_tool()
    
    try:
        # Esegui ricerca
        results = search_tool.run(query)
        
        # Aggiungi contesto
        formatted_result = f"🌐 Risultati ricerca web per '{query}':\n\n{results}"
        
        return formatted_result
        
    except Exception as e:
        return f"❌ Errore nella ricerca web: {str(e)}"

@tool
def combined_guest_search(query: str) -> str:
    """
    Combina ricerca nel database locale e ricerca web per informazioni complete sui guest.
    
    Args:
        query: Il nome o informazioni sul guest che stai cercando
        
    Returns:
        Informazioni combinate da database locale e web
    """
    # Prima cerca nel database locale
    local_results = guest_info_retriever(query)
    
    results_parts = [f"📚 Database locale:\n{local_results}"]
    
    # Se non troviamo risultati soddisfacenti localmente, cerca sul web
    if "Nessuna informazione trovata" in local_results:
        web_results = web_search(f"{query} biography information")
        results_parts.append(f"\n{web_results}")
    
    return "\n".join(results_parts)

# Lista di tutti i tools disponibili
def get_all_tools():
    """
    Restituisce tutti i tools disponibili per Alfred.
    """
    return [
        guest_info_retriever,
        web_search, 
        combined_guest_search
    ]

def get_basic_tools():
    """
    Restituisce solo i tools di base (senza ricerca web).
    """
    return [guest_info_retriever]

def get_enhanced_tools():
    """
    Restituisce tutti i tools inclusa la ricerca web combinata.
    """
    return [combined_guest_search]