#!/usr/bin/env python3
"""
Esporta il dataset in CSV per visualizzazione rapida.
"""

import pandas as pd
import datasets
import os
from retriever import ensure_guest_data


def export_original_dataset():
    """Esporta il dataset originale in CSV."""
    print("📥 Download dataset originale...")

    try:
        # Carica dataset originale
        guest_dataset = datasets.load_dataset(
            "agents-course/unit3-invitees", split="train")

        # Converti in DataFrame
        df = pd.DataFrame(guest_dataset)

        # Salva CSV
        csv_path = "./data/original_guests.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

        print(f"✅ Dataset originale salvato in: {csv_path}")
        print(f"📊 Righe: {len(df)}, Colonne: {list(df.columns)}")

        # Mostra anteprima
        print("\n👥 Primi 3 ospiti:")
        print(df.head(3).to_string())

        return df

    except Exception as e:
        print(f"❌ Errore export originale: {e}")
        return None


def export_enhanced_dataset():
    """Esporta il dataset arricchito in CSV."""
    print("\n📊 Export dataset arricchito...")

    try:
        # Carica documenti arricchiti
        documents = ensure_guest_data()

        # Estrai dati in formato tabellare
        enhanced_data = []
        for doc in documents:
            metadata = doc.metadata

            # Estrai informazioni dal content
            content_lines = doc.page_content.split('\n')
            content_dict = {}
            for line in content_lines:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    content_dict[key] = value

            enhanced_data.append({
                'name': metadata.get('name', ''),
                'relation': metadata.get('relation', ''),
                'email': metadata.get('email', ''),
                'description': content_dict.get('Descrizione', ''),
                'interests': ', '.join(metadata.get('interests', [])),
                'topics': ', '.join(metadata.get('topics', [])),
                'conversation_starters_count': len(metadata.get('conversation_starters', [])),
                # Prime 3
                'conversation_starters': ' | '.join(metadata.get('conversation_starters', [])[:3])
            })

        # Crea DataFrame
        df_enhanced = pd.DataFrame(enhanced_data)

        # Salva CSV arricchito
        enhanced_csv_path = "./data/enhanced_guests.csv"
        df_enhanced.to_csv(enhanced_csv_path, index=False, encoding='utf-8')

        print(f"✅ Dataset arricchito salvato in: {enhanced_csv_path}")
        print(
            f"📊 Righe: {len(df_enhanced)}, Colonne: {list(df_enhanced.columns)}")

        # Mostra anteprima
        print("\n🎯 Anteprima arricchimenti:")
        for _, row in df_enhanced.head(2).iterrows():
            print(f"\n👤 {row['name']}:")
            print(f"   Interessi: {row['interests']}")
            print(f"   Topics: {row['topics']}")
            print(f"   Starters: {row['conversation_starters_count']}")

        return df_enhanced

    except Exception as e:
        print(f"❌ Errore export arricchito: {e}")
        return None


def create_comparison_report():
    """Crea un report di comparazione."""
    print("\n📋 Creazione report comparativo...")

    # Carica entrambi i dataset
    df_original = export_original_dataset()
    df_enhanced = export_enhanced_dataset()

    if df_original is not None and df_enhanced is not None:
        # Crea report
        report = f"""
📊 REPORT DATASET ALFRED
========================

📈 STATISTICHE:
- Ospiti totali: {len(df_original)}
- Campi originali: {len(df_original.columns)}
- Campi arricchiti: {len(df_enhanced.columns)}

👥 OSPITI NEL DATABASE:
{chr(10).join([f"• {name} ({relation})" for name, relation in zip(df_original['name'], df_original['relation'])])}

🎯 ARRICCHIMENTI AGGIUNTI:
- Interessi estratti automaticamente
- Topics di conversazione identificati  
- Conversation starters personalizzati
- Metadati per ricerca semantica

📁 FILE GENERATI:
- ./data/original_guests.csv - Dataset originale
- ./data/enhanced_guests.csv - Dataset arricchito
- ./data/dataset_report.txt - Questo report
"""

        # Salva report
        with open("./data/dataset_report.txt", "w", encoding='utf-8') as f:
            f.write(report)

        print(report)
        print("✅ Report salvato in: ./data/dataset_report.txt")


def main():
    """Funzione principale di export."""
    print("🎩 Alfred Dataset Export Tool")
    print("=" * 40)

    # Assicurati che la directory esista
    os.makedirs("./data", exist_ok=True)

    # Export completo
    create_comparison_report()

    print("\n🎉 Export completato!")
    print("📁 Controlla la cartella ./data/ per i file CSV")


if __name__ == "__main__":
    main()
