#!/usr/bin/env python3
"""
Script di debug per verificare e generare i dati mancanti per Alfred.
"""

import os
import pickle
import datasets
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def check_data_directory():
    """Verifica lo stato della directory dei dati."""
    print("🔍 Controllo directory dati...")

    data_dir = "./data"
    faiss_dir = f"{data_dir}/faiss_index"
    dataset_file = f"{data_dir}/guest_dataset.pkl"

    print(
        f"📁 Directory data: {'✅ Esiste' if os.path.exists(data_dir) else '❌ Non esiste'}")
    print(
        f"📁 Directory FAISS: {'✅ Esiste' if os.path.exists(faiss_dir) else '❌ Non esiste'}")
    print(
        f"📄 File dataset: {'✅ Esiste' if os.path.exists(dataset_file) else '❌ Non esiste'}")

    if os.path.exists(data_dir):
        files = os.listdir(data_dir)
        print(f"📋 Contenuti data/: {files if files else 'Vuota'}")

    return data_dir, faiss_dir, dataset_file


def download_and_prepare_dataset():
    """Scarica e prepara il dataset degli ospiti."""
    print("\n📥 Download dataset ospiti...")

    try:
        # Scarica il dataset
        guest_dataset = datasets.load_dataset(
            "agents-course/unit3-invitees", split="train")
        print(f"✅ Dataset scaricato: {len(guest_dataset)} ospiti")

        # Converti in documenti
        documents = []
        for guest in guest_dataset:
            doc = Document(
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
            documents.append(doc)

        print(f"📄 Creati {len(documents)} documenti")

        # Mostra alcuni esempi
        print("\n👥 Primi 3 ospiti:")
        for i, doc in enumerate(documents[:3]):
            print(f"\n{i+1}. {doc.metadata['name']}")
            print(f"   Relazione: {doc.metadata['relation']}")
            print(f"   Content preview: {doc.page_content[:100]}...")

        return documents

    except Exception as e:
        print(f"❌ Errore nel download: {e}")
        return None


def create_directories_and_save_dataset(documents):
    """Crea le directory e salva il dataset."""
    print("\n📁 Creazione directory e salvataggio...")

    data_dir = "./data"
    dataset_file = f"{data_dir}/guest_dataset.pkl"

    # Crea directory
    os.makedirs(data_dir, exist_ok=True)
    print(f"✅ Directory {data_dir} creata")

    # Salva il dataset processato
    with open(dataset_file, 'wb') as f:
        pickle.dump(documents, f)

    print(f"💾 Dataset salvato in {dataset_file}")
    print(f"📊 Dimensione file: {os.path.getsize(dataset_file)} bytes")


def create_faiss_index(documents):
    """Crea l'indice FAISS."""
    print("\n🧠 Creazione indice FAISS...")

    try:
        # Inizializza embeddings
        print("⚙️ Inizializzo embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print("✅ Embeddings pronti")

        # Crea vector store
        print("🗃️ Creazione vector store...")
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        print("✅ Vector store creato")

        # Salva l'indice
        faiss_dir = "./data/faiss_index"
        vector_store.save_local(faiss_dir)
        print(f"💾 Indice FAISS salvato in {faiss_dir}")

        # Verifica che sia stato salvato
        faiss_files = os.listdir(faiss_dir)
        print(f"📋 File FAISS creati: {faiss_files}")

        return vector_store

    except Exception as e:
        print(f"❌ Errore nella creazione FAISS: {e}")
        return None


def test_retrieval(vector_store):
    """Testa il sistema di retrieval."""
    print("\n🧪 Test del sistema di retrieval...")

    test_queries = [
        "Ada Lovelace",
        "mathematician",
        "Lord Byron",
        "scientist"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            results = vector_store.similarity_search_with_score(query, k=2)

            if results:
                for i, (doc, score) in enumerate(results):
                    similarity = 1 / (1 + score)
                    print(f"  Risultato {i+1} (sim: {similarity:.3f}):")
                    print(f"    Nome: {doc.metadata.get('name', 'N/A')}")
                    print(
                        f"    Relazione: {doc.metadata.get('relation', 'N/A')}")
            else:
                print("  ❌ Nessun risultato")

        except Exception as e:
            print(f"  ❌ Errore: {e}")


def main():
    """Funzione principale di debug."""
    print("🎩 Alfred Data Debug Tool")
    print("=" * 50)

    # 1. Controlla stato attuale
    data_dir, faiss_dir, dataset_file = check_data_directory()

    # 2. Se mancano dati, scaricali
    if not os.path.exists(dataset_file):
        print("\n📥 Dataset mancante, procedo al download...")
        documents = download_and_prepare_dataset()

        if documents:
            create_directories_and_save_dataset(documents)
        else:
            print("❌ Impossibile scaricare il dataset")
            return
    else:
        print("\n📂 Dataset già presente, lo carico...")
        with open(dataset_file, 'rb') as f:
            documents = pickle.load(f)
        print(f"✅ Dataset caricato: {len(documents)} documenti")

    # 3. Se manca FAISS, crealo
    if not os.path.exists(faiss_dir):
        print("\n🗃️ Indice FAISS mancante, lo creo...")
        vector_store = create_faiss_index(documents)
    else:
        print("\n📂 Indice FAISS già presente, lo carico...")
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vector_store = FAISS.load_local(
                faiss_dir,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ Indice FAISS caricato")
        except Exception as e:
            print(f"❌ Errore caricamento FAISS: {e}")
            vector_store = None

    # 4. Testa il sistema
    if vector_store:
        test_retrieval(vector_store)

        print("\n✅ Setup completato con successo!")
        print("🎩 Alfred dovrebbe ora funzionare correttamente")
    else:
        print("\n❌ Setup fallito")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
