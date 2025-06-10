# Questo file mantiene compatibilità con il codice esistente
# I tools principali sono ora in tools.py

from tools import guest_info_retriever, get_basic_tools, get_enhanced_tools

# Funzioni di compatibilità


def load_guest_dataset_semantic():
    """Compatibilità: restituisce il tool semantico."""
    return guest_info_retriever


def load_guest_dataset_bm25():
    """Compatibilità: restituisce il tool semantico (BM25 deprecato)."""
    return guest_info_retriever


def get_guest_info_tool(use_semantic=True):
    """Compatibilità: restituisce il tool appropriato."""
    return guest_info_retriever
