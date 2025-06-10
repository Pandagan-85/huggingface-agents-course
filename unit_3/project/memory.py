from typing import Dict, List, Any
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import json
from datetime import datetime


class AlfredMemory:
    """
    Sistema di memoria per Alfred che mantiene:
    - Conversazioni recenti complete
    - Riassunti di conversazioni più vecchie
    - Preferenze dell'utente
    - Informazioni contestuali sui guest
    """

    def __init__(self, max_token_limit: int = 1000):
        self.max_token_limit = max_token_limit
        self.conversation_history = []
        self.guest_context = {}  # Informazioni sui guest menzionati
        self.user_preferences = {}  # Preferenze dell'utente
        self.session_start = datetime.now()

    def add_interaction(self, human_message: str, ai_message: str, context: Dict = None):
        """
        Aggiunge una nuova interazione alla memoria.
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "human": human_message,
            "ai": ai_message,
            "context": context or {}
        }

        self.conversation_history.append(interaction)

        # Mantieni solo le ultime N interazioni per evitare overflow
        if len(self.conversation_history) > 20:
            # Mantieni le ultime 15
            self.conversation_history = self.conversation_history[-15:]

    def update_guest_context(self, guest_name: str, info: Dict):
        """
        Aggiorna le informazioni contestuali su un guest.
        """
        if guest_name not in self.guest_context:
            self.guest_context[guest_name] = {}

        self.guest_context[guest_name].update(info)
        self.guest_context[guest_name]["last_mentioned"] = datetime.now(
        ).isoformat()

    def get_relevant_context(self, current_query: str) -> str:
        """
        Recupera il contesto rilevante per la query corrente.
        """
        context_parts = []

        # Aggiungi conversazioni recenti
        if self.conversation_history:
            # Ultime 3 interazioni
            recent_conversations = self.conversation_history[-3:]
            context_parts.append("Conversazioni recenti:")
            for conv in recent_conversations:
                context_parts.append(f"Utente: {conv['human']}")
                # Primi 100 caratteri
                context_parts.append(f"Alfred: {conv['ai'][:100]}...")

        # Aggiungi contesto sui guest se rilevante
        query_lower = current_query.lower()
        relevant_guests = []
        for guest_name, info in self.guest_context.items():
            if guest_name.lower() in query_lower:
                relevant_guests.append(f"- {guest_name}: {info}")

        if relevant_guests:
            context_parts.append(
                "\nInformazioni sui guest precedentemente discussi:")
            context_parts.extend(relevant_guests)

        return "\n".join(context_parts) if context_parts else ""

    def extract_guest_names(self, text: str) -> List[str]:
        """
        Estrae i nomi dei guest dal testo (semplice implementazione).
        """
        # Lista di nomi comuni dai guest (potresti espanderla)
        known_guests = ["Ada Lovelace", "Lord Byron",
                        "Charles Babbage", "Mary Shelley"]

        found_guests = []
        text_lower = text.lower()

        for guest in known_guests:
            if guest.lower() in text_lower:
                found_guests.append(guest)

        return found_guests

    def get_memory_summary(self) -> str:
        """
        Restituisce un riassunto della sessione corrente.
        """
        summary_parts = [
            f"Sessione iniziata: {self.session_start.strftime('%H:%M')}",
            f"Interazioni totali: {len(self.conversation_history)}",
            f"Guest discussi: {len(self.guest_context)}"
        ]

        if self.guest_context:
            guest_names = list(self.guest_context.keys())
            summary_parts.append(f"Guest menzionati: {', '.join(guest_names)}")

        return "\n".join(summary_parts)


# Istanza globale della memoria (per semplicità)
alfred_memory = AlfredMemory()
