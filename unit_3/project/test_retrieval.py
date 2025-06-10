#!/usr/bin/env python3
"""
Test rapido per verificare se il retrieval funziona.
"""

import os
from tools import guest_info_retriever, get_vector_store, ensure_guest_data


def test_data_loading():
    """Testa il caricamento dei dati step by step."""
    print("🔍 Test caricamento dati...")

    # 1. Controlla se esistono i file
    print("\n📁 Controllo file:")
    data_dir = "./data"
    dataset_file = f"{data_dir}/guest_dataset.pkl"
    faiss_dir = f"{data_dir}/faiss_index"

    print(f"  Directory data: {'✅' if os.path.exists(data_dir) else '❌'}")
    print(f"  File dataset: {'✅' if os.path.exists(dataset_file) else '❌'}")
    print(f"  Directory FAISS: {'✅' if os.path.exists(faiss_dir) else '❌'}")

    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"  Contenuti data/: {files}")

    # 2. Testa caricamento documenti
    print("\n📄 Test caricamento documenti...")
    try:
        documents = ensure_guest_data()
        print(f"  ✅ Documenti caricati: {len(documents)}")

        # Mostra primi documenti
        for i, doc in enumerate(documents[:2]):
            print(f"  Documento {i+1}:")
            print(f"    Nome: {doc.metadata.get('name', 'N/A')}")
            print(f"    Content: {doc.page_content[:100]}...")

    except Exception as e:
        print(f"  ❌ Errore caricamento documenti: {e}")
        return False

    # 3. Testa vector store
    print("\n🗃️ Test vector store...")
    try:
        vector_store = get_vector_store()
        print(f"  ✅ Vector store caricato")

        # Test di ricerca diretta
        results = vector_store.similarity_search("Ada Lovelace", k=1)
        if results:
            print(
                f"  ✅ Test ricerca OK: {results[0].metadata.get('name', 'N/A')}")
        else:
            print(f"  ❌ Nessun risultato per Ada Lovelace")

    except Exception as e:
        print(f"  ❌ Errore vector store: {e}")
        return False

    # 4. Testa il tool direttamente
    print("\n🔧 Test tool guest_info_retriever...")
    try:
        result = guest_info_retriever.invoke({"query": "Ada Lovelace"})
        print(f"  Risultato tool: {result[:200]}...")

        if "Nessuna informazione trovata" in result:
            print("  ❌ Tool non trova informazioni")
            return False
        else:
            print("  ✅ Tool funziona correttamente")

    except Exception as e:
        print(f"  ❌ Errore tool: {e}")
        return False

    return True


def force_recreate_data():
    """Forza la ricreazione dei dati."""
    print("\n🔄 Ricreazione forzata dei dati...")

    import shutil
    data_dir = "./data"

    # Rimuovi directory esistente
    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)
        print("  🗑️ Directory data rimossa")

    # Ricrea tutto
    os.makedirs(data_dir, exist_ok=True)
    print("  📁 Directory data ricreata")

    # Forza ricaricamento
    from tools import _vector_store, _guest_documents
    globals()['_vector_store'] = None
    globals()['_guest_documents'] = None

    print("  🔄 Cache pulita")

    # Ricarica dati
    try:
        documents = ensure_guest_data()
        vector_store = get_vector_store()
        print(f"  ✅ Dati ricreati: {len(documents)} documenti")
        return True
    except Exception as e:
        print(f"  ❌ Errore ricreazione: {e}")
        return False


def main():
    """Test principale."""
    print("🎩 Alfred Retrieval Test")
    print("=" * 40)

    # Prima prova a testare lo stato attuale
    success = test_data_loading()

    if not success:
        print("\n🔧 Tentativo di riparazione...")
        success = force_recreate_data()

        if success:
            print("\n✅ Riparazione completata! Ritesta...")
            success = test_data_loading()

    if success:
        print("\n🎉 Tutto funziona! Alfred dovrebbe ora rispondere correttamente.")
        print("💡 Prova di nuovo 'Tell me about ada' in Alfred")
    else:
        print("\n❌ Problemi persistenti. Verifica:")
        print("  1. Connessione internet per scaricare il dataset")
        print("  2. Permessi di scrittura nella directory")
        print("  3. Spazio su disco disponibile")
        print("  4. Dipendenze installate: datasets, sentence-transformers, faiss-cpu")


if __name__ == "__main__":
    main()
