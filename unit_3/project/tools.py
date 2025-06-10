# File tools.py COMPLETO con tutti i tools inclusi News Tools

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
import random
from huggingface_hub import list_models, list_datasets, list_spaces
import requests
from datetime import datetime, timedelta
import json
from urllib.parse import quote_plus

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
    """Assicura che i dati dei guest siano caricati e processati."""
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
        guest_dataset = datasets.load_dataset(
            "agents-course/unit3-invitees", split="train")

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
    """Ottiene il vector store FAISS, creandolo se necessario."""
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
            print(
                f"💾 Indice FAISS creato e salvato con {len(documents)} documenti")

    return _vector_store

# =================
# GUEST TOOLS
# =================


@tool
def guest_info_retriever(query: str) -> str:
    """
    Recupera informazioni dettagliate sugli ospiti del gala usando ricerca vettoriale FAISS.

    Args:
        query: Il nome, relazione o descrizione dell'ospite che stai cercando

    Returns:
        Informazioni dettagliate sui guest trovati
    """
    try:
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

    except Exception as e:
        return f"❌ Errore nel retrieval: {str(e)}"

# =================
# WEB SEARCH TOOLS
# =================


def get_web_search_tool():
    """Ottiene il tool di ricerca web, creandolo se necessario."""
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

# =================
# WEATHER TOOL
# =================


@tool
def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20},
        {"condition": "Cloudy", "temp_c": 18},
        {"condition": "Sunny", "temp_c": 28},
        {"condition": "Stormy", "temp_c": 12}
    ]

    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"🌤️ Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# =================
# HUGGING FACE HUB TOOLS
# =================


@tool
def get_hub_stats(author: str) -> str:
    """
    Fetches comprehensive Hugging Face Hub statistics for a specific author.

    Args:
        author: The Hugging Face username/organization name

    Returns:
        Detailed statistics about models, datasets, and spaces
    """
    try:
        # Fetch most downloaded model
        models = list(list_models(
            author=author, sort="downloads", direction=-1, limit=5))

        # Fetch datasets
        try:
            datasets_list = list(list_datasets(
                author=author, sort="downloads", direction=-1, limit=3))
        except:
            datasets_list = []

        # Fetch spaces
        try:
            spaces = list(list_spaces(
                author=author, sort="likes", direction=-1, limit=3))
        except:
            spaces = []

        # Build comprehensive response
        response_parts = [f"🤖 **Hugging Face Hub Statistics for {author}:**"]

        # Models section
        if models:
            response_parts.append(f"\n📊 **Top Models ({len(models)}):**")
            for i, model in enumerate(models, 1):
                downloads = getattr(model, 'downloads', 0) or 0
                likes = getattr(model, 'likes', 0) or 0
                response_parts.append(f"  {i}. **{model.id}**")
                response_parts.append(f"     📥 Downloads: {downloads:,}")
                response_parts.append(f"     ❤️ Likes: {likes:,}")

                # Add model tags if available
                if hasattr(model, 'tags') and model.tags:
                    key_tags = [tag for tag in model.tags[:3]
                                if not tag.startswith('license:')]
                    if key_tags:
                        response_parts.append(
                            f"     🏷️ Tags: {', '.join(key_tags)}")
        else:
            response_parts.append(f"\n📊 **Models:** No models found")

        # Datasets section
        if datasets_list:
            response_parts.append(
                f"\n📚 **Top Datasets ({len(datasets_list)}):**")
            for i, dataset in enumerate(datasets_list, 1):
                downloads = getattr(dataset, 'downloads', 0) or 0
                response_parts.append(
                    f"  {i}. **{dataset.id}** ({downloads:,} downloads)")

        # Spaces section
        if spaces:
            response_parts.append(f"\n🚀 **Top Spaces ({len(spaces)}):**")
            for i, space in enumerate(spaces, 1):
                likes = getattr(space, 'likes', 0) or 0
                response_parts.append(
                    f"  {i}. **{space.id}** ({likes:,} likes)")

        # Summary
        try:
            total_models = len(
                list(list_models(author=author, limit=1000))) if models else 0
            if total_models > 0:
                response_parts.append(f"\n📈 **Summary:**")
                response_parts.append(f"  • Total Models: {total_models}")
                if models:
                    total_downloads = sum(
                        getattr(m, 'downloads', 0) or 0 for m in models)
                    response_parts.append(
                        f"  • Top 5 Downloads: {total_downloads:,}")
        except:
            pass

        return "\n".join(response_parts)

    except Exception as e:
        return f"❌ Error fetching Hub stats for {author}: {str(e)}\n💡 Make sure the username exists on Hugging Face Hub."


@tool
def get_trending_models(task: str = "") -> str:
    """
    Get trending models from Hugging Face Hub, optionally filtered by task.

    Args:
        task: Optional task filter (e.g., 'text-generation', 'image-classification')

    Returns:
        List of trending models with statistics
    """
    try:
        # Get trending models
        if task:
            models = list(list_models(
                task=task, sort="downloads", direction=-1, limit=5))
            header = f"🔥 **Trending {task} Models:**"
        else:
            models = list(list_models(sort="downloads", direction=-1, limit=5))
            header = f"🔥 **Trending Models:**"

        if not models:
            return f"❌ No trending models found for task: {task}"

        response_parts = [header]

        for i, model in enumerate(models, 1):
            downloads = getattr(model, 'downloads', 0) or 0
            likes = getattr(model, 'likes', 0) or 0
            author = model.id.split('/')[0] if '/' in model.id else 'Unknown'

            response_parts.append(f"\n{i}. **{model.id}**")
            response_parts.append(f"   👤 Author: {author}")
            response_parts.append(f"   📥 Downloads: {downloads:,}")
            response_parts.append(f"   ❤️ Likes: {likes:,}")

        return "\n".join(response_parts)

    except Exception as e:
        return f"❌ Error fetching trending models: {str(e)}"


@tool
def compare_ai_builders(authors: str) -> str:
    """
    Compare statistics between multiple AI builders on Hugging Face Hub.

    Args:
        authors: Comma-separated list of usernames (e.g., "microsoft,google,openai")

    Returns:
        Comparison of their Hub statistics
    """
    try:
        author_list = [author.strip()
                       for author in authors.split(',') if author.strip()]

        if len(author_list) < 2:
            return "❌ Please provide at least 2 authors separated by commas"

        if len(author_list) > 5:
            return "❌ Maximum 5 authors for comparison"

        response_parts = ["🏆 **AI Builders Comparison:**"]
        comparison_data = []

        for author in author_list:
            try:
                # Get top model
                models = list(list_models(
                    author=author, sort="downloads", direction=-1, limit=1))
                top_model = models[0] if models else None

                # Count total models
                total_models = len(list(list_models(author=author, limit=100)))

                comparison_data.append({
                    'author': author,
                    'total_models': total_models,
                    'top_model': top_model.id if top_model else 'None',
                    'top_downloads': getattr(top_model, 'downloads', 0) if top_model else 0
                })

            except Exception as e:
                comparison_data.append({
                    'author': author,
                    'total_models': 0,
                    'top_model': f'Error: {str(e)}',
                    'top_downloads': 0
                })

        # Sort by top downloads
        comparison_data.sort(key=lambda x: x['top_downloads'], reverse=True)

        response_parts.append(f"\n📊 **Rankings by Top Model Downloads:**")
        for i, data in enumerate(comparison_data, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
            response_parts.append(f"\n{medal} **{data['author']}**")
            response_parts.append(f"   📈 Total Models: {data['total_models']}")
            response_parts.append(f"   🏆 Top Model: {data['top_model']}")
            response_parts.append(
                f"   📥 Top Downloads: {data['top_downloads']:,}")

        return "\n".join(response_parts)

    except Exception as e:
        return f"❌ Error comparing AI builders: {str(e)}"

# =================
# NEWS TOOLS
# =================


@tool
def get_latest_news(topic: str, max_results: int = 5) -> str:
    """
    Get the latest news about a specific topic using DuckDuckGo News.

    Args:
        topic: The topic to search for (e.g., "artificial intelligence", "Hugging Face", "machine learning")
        max_results: Maximum number of news articles to return (default: 5, max: 10)

    Returns:
        Latest news articles about the topic with summaries and sources
    """
    try:
        # Limit max_results to prevent overwhelming responses
        max_results = min(max_results, 10)

        # Use our existing web search but with news-specific queries
        search_tool = get_web_search_tool()

        if search_tool is None:
            return "❌ News search not available at the moment"

        # Enhanced news search query
        news_query = f"{topic} news latest 2024 OR 2025"

        try:
            results = search_tool.run(news_query)

            # Format the results for news presentation
            response_parts = [f"📰 **Latest News about '{topic}':**"]
            response_parts.append(
                f"🕐 *Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC*\n")

            # Split results into segments (basic parsing)
            if len(results) > 100:
                # Take first portion for latest news
                news_content = results[:1500]
                response_parts.append(news_content)

                if len(results) > 1500:
                    response_parts.append(
                        f"\n... [Results truncated for readability]")
            else:
                response_parts.append(results)

            response_parts.append(
                f"\n💡 *For more detailed news, try specific searches about recent developments in {topic}*")

            return "\n".join(response_parts)

        except Exception as e:
            return f"❌ Error fetching news: {str(e)}"

    except Exception as e:
        return f"❌ Error in news search: {str(e)}"


@tool
def get_ai_industry_news(category: str = "general") -> str:
    """
    Get the latest AI industry news, perfect for gala conversations.

    Args:
        category: News category ("general", "companies", "research", "funding", "events")

    Returns:
        Latest AI industry news relevant for sophisticated gala discussions
    """
    try:
        # Define search queries for different categories
        search_queries = {
            "general": "artificial intelligence AI news latest developments 2024 2025",
            "companies": "AI companies funding startup unicorn acquisitions 2024 2025",
            "research": "AI research breakthrough GPT LLM machine learning 2024 2025",
            "funding": "AI funding venture capital investment round 2024 2025",
            "events": "AI conference summit gala awards tech events 2024 2025"
        }

        query = search_queries.get(category, search_queries["general"])

        search_tool = get_web_search_tool()
        if search_tool is None:
            return "❌ AI news search not available at the moment"

        try:
            results = search_tool.run(query)

            # Format for gala-appropriate conversation
            response_parts = [
                f"🤖 **Latest AI Industry News - {category.title()}:**"]
            response_parts.append(
                f"*Perfect conversation starters for the gala!*\n")

            # Process and format results
            news_content = results[:1200] if len(results) > 1200 else results
            response_parts.append(news_content)

            # Add conversation suggestions
            conversation_tips = {
                "general": [
                    "💬 *Great for: Opening conversations about industry trends*",
                    "🎯 *Mention: Recent AI breakthroughs and their implications*"
                ],
                "companies": [
                    "💬 *Great for: Discussing with entrepreneurs and investors*",
                    "🎯 *Mention: Market dynamics and competitive landscape*"
                ],
                "research": [
                    "💬 *Great for: Conversations with researchers and academics*",
                    "🎯 *Mention: Technical advances and research directions*"
                ],
                "funding": [
                    "💬 *Great for: Discussions with VCs and startup founders*",
                    "🎯 *Mention: Investment trends and market opportunities*"
                ],
                "events": [
                    "💬 *Great for: Networking and event planning discussions*",
                    "🎯 *Mention: Upcoming conferences and industry gatherings*"
                ]
            }

            tips = conversation_tips.get(
                category, conversation_tips["general"])
            response_parts.append(f"\n🎩 **Alfred's Conversation Tips:**")
            for tip in tips:
                response_parts.append(tip)

            return "\n".join(response_parts)

        except Exception as e:
            return f"❌ Error fetching AI industry news: {str(e)}"

    except Exception as e:
        return f"❌ Error in AI news search: {str(e)}"


@tool
def get_tech_company_news(company: str) -> str:
    """
    Get the latest news about a specific tech company, great for gala small talk.

    Args:
        company: Company name (e.g., "OpenAI", "Google", "Microsoft", "Meta", "Apple")

    Returns:
        Recent news about the company formatted for sophisticated conversation
    """
    try:
        search_tool = get_web_search_tool()
        if search_tool is None:
            return f"❌ Company news search for {company} not available"

        # Enhanced search query for company news
        search_query = f"{company} news latest announcements products 2024 2025"

        try:
            results = search_tool.run(search_query)

            response_parts = [f"🏢 **Latest News about {company}:**"]
            response_parts.append(
                f"*Perfect for informed gala conversations!*\n")

            # Format results
            news_content = results[:1000] if len(results) > 1000 else results
            response_parts.append(news_content)

            # Add context for gala conversations
            response_parts.append(f"\n🎭 **Gala Conversation Context:**")

            company_contexts = {
                "openai": "Discuss GPT developments, partnerships, and AI safety initiatives",
                "google": "Mention Bard, DeepMind research, and cloud AI services",
                "microsoft": "Talk about Copilot integration, Azure AI, and OpenAI partnership",
                "meta": "Discuss LLaMA models, VR/AR developments, and AI research",
                "apple": "Mention on-device AI, Siri improvements, and privacy-focused ML",
                "nvidia": "Discuss GPU innovations, AI chip developments, and partnerships",
                "tesla": "Talk about FSD progress, AI training, and autonomous driving",
                "hugging face": "Mention model releases, community growth, and open-source AI"
            }

            context = company_contexts.get(
                company.lower(), f"Discuss {company}'s latest developments and market position")
            response_parts.append(f"💡 *Suggested topics: {context}*")

            return "\n".join(response_parts)

        except Exception as e:
            return f"❌ Error fetching {company} news: {str(e)}"

    except Exception as e:
        return f"❌ Error in company news search: {str(e)}"


@tool
def get_breaking_tech_news() -> str:
    """
    Get breaking technology news that's trending right now.

    Returns:
        Current breaking tech news for immediate gala conversation topics
    """
    try:
        search_tool = get_web_search_tool()
        if search_tool is None:
            return "❌ Breaking news search not available"

        # Search for very recent tech news
        current_date = datetime.now().strftime("%Y-%m-%d")
        search_query = f"breaking tech news today {current_date} AI technology startup"

        try:
            results = search_tool.run(search_query)

            response_parts = [f"🚨 **Breaking Technology News:**"]
            response_parts.append(
                f"📅 *{datetime.now().strftime('%B %d, %Y at %H:%M UTC')}*\n")

            # Format breaking news
            news_content = results[:1200] if len(results) > 1200 else results
            response_parts.append(news_content)

            # Add urgency context for gala
            response_parts.append(f"\n⚡ **Gala Impact:**")
            response_parts.append(
                f"🗣️ *These are today's hottest tech topics - perfect for current conversations!*")
            response_parts.append(
                f"🎯 *Use these for: Demonstrating you're up-to-date with latest developments*")
            response_parts.append(
                f"💼 *Great with: Investors, entrepreneurs, and industry leaders*")

            return "\n".join(response_parts)

        except Exception as e:
            return f"❌ Error fetching breaking news: {str(e)}"

    except Exception as e:
        return f"❌ Error in breaking news search: {str(e)}"


@tool
def get_guest_related_news(guest_name: str) -> str:
    """
    Search for recent news about a specific gala guest to prepare conversation topics.

    Args:
        guest_name: Name of the guest (e.g., "Ada Lovelace", "Marie Curie")

    Returns:
        Recent news and information about the guest for informed conversations
    """
    try:
        # First check if guest is in our database
        guest_info = guest_info_retriever.invoke({"query": guest_name})

        # Then search for recent news
        search_tool = get_web_search_tool()
        if search_tool is None:
            return f"📚 Guest database info for {guest_name}:\n{guest_info}\n\n❌ Recent news search not available"

        # Search for news about the guest
        news_query = f'"{guest_name}" news recent work achievements 2024 2025'

        try:
            news_results = search_tool.run(news_query)

            response_parts = [
                f"👤 **Complete Information about {guest_name}:**"]

            # Add database info first
            if "Nessuna informazione" not in guest_info:
                response_parts.append(f"\n📚 **From Gala Database:**")
                response_parts.append(guest_info)

            # Add recent news
            response_parts.append(f"\n📰 **Recent News & Updates:**")
            news_content = news_results[:800] if len(
                news_results) > 800 else news_results
            response_parts.append(news_content)

            # Conversation preparation
            response_parts.append(f"\n🎩 **Conversation Preparation:**")
            response_parts.append(
                f"✨ *You're now well-prepared to have an informed discussion with {guest_name}*")
            response_parts.append(
                f"💡 *Mention both their background and recent developments*")
            response_parts.append(
                f"🎯 *Show genuine interest in their current work and achievements*")

            return "\n".join(response_parts)

        except Exception as e:
            return f"📚 Guest database info:\n{guest_info}\n\n❌ Error fetching recent news: {str(e)}"

    except Exception as e:
        return f"❌ Error in guest news search: {str(e)}"

# =================
# TOOL COLLECTIONS
# =================


def get_all_tools():
    """Restituisce tutti i tools disponibili per Alfred."""
    return [
        # Guest tools
        guest_info_retriever,
        combined_guest_search,
        get_guest_related_news,  # Combines guest DB + news

        # Web search
        web_search,

        # Weather
        get_weather_info,

        # Hugging Face Hub
        get_hub_stats,
        get_trending_models,
        compare_ai_builders,

        # News tools
        get_latest_news,
        get_ai_industry_news,
        get_tech_company_news,
        get_breaking_tech_news
    ]


def get_basic_tools():
    """Restituisce solo i tools di base."""
    return [guest_info_retriever, get_weather_info]


def get_enhanced_tools():
    """Restituisce tools avanzati."""
    return [combined_guest_search, get_hub_stats, get_latest_news]


def get_news_tools():
    """Restituisce solo i news tools."""
    return [
        get_latest_news,
        get_ai_industry_news,
        get_tech_company_news,
        get_breaking_tech_news,
        get_guest_related_news
    ]


def get_hub_tools():
    """Restituisce solo i tools per Hugging Face Hub."""
    return [get_hub_stats, get_trending_models, compare_ai_builders]

# Test function


def test_all_tools():
    """Test di tutti i tools."""
    print("🧪 Test All Tools")
    print("=" * 40)

    # Test News tools
    print("1. Test Latest News:")
    result1 = get_latest_news.invoke(
        {"topic": "artificial intelligence", "max_results": 3})
    print(f"   {result1[:200]}...")

    print("\n2. Test AI Industry News:")
    result2 = get_ai_industry_news.invoke({"category": "companies"})
    print(f"   {result2[:200]}...")

    print("\n3. Test Company News:")
    result3 = get_tech_company_news.invoke({"company": "OpenAI"})
    print(f"   {result3[:200]}...")

    print("\n4. Test Breaking News:")
    result4 = get_breaking_tech_news.invoke({})
    print(f"   {result4[:200]}...")

    print("\n5. Test Hub Stats:")
    result5 = get_hub_stats.invoke({"author": "microsoft"})
    print(f"   {result5[:200]}...")

    print("\n6. Test Weather:")
    result6 = get_weather_info.invoke({"location": "London"})
    print(f"   {result6}")

    print("\n7. Test Guest Info:")
    result7 = guest_info_retriever.invoke({"query": "Ada"})
    print(f"   {result7[:200]}...")


if __name__ == "__main__":
    test_all_tools()
