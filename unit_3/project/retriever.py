from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool
import datasets
from langchain_core.documents import Document


class GuestInfoRetrieverTool(Tool):
    """
    A tool to retrieve information about gala guests based on their name or relation.
    This tool uses a BM25 retriever to find relevant documents from a dataset of guests.
    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of what the tool does.
        inputs (dict): The expected input format for the tool.
        output_type (str): The type of output the tool returns.
    """
    name = "guest_info_retriever"
    description = "Retrieves detailed information about gala guests based on their name or relation."
    inputs = {
        "query": {
            "type": "string",
            "description": "The name or relation of the guest you want information about."
        }
    }
    output_type = "string"

    def __init__(self, docs):
        self.is_initialized = False
        self.retriever = BM25Retriever.from_documents(docs)

    def forward(self, query: str):
        results = self.retriever.get_relevant_documents(query)
        if results:
            return "\n\n".join([doc.page_content for doc in results[:3]])
        else:
            return "No matching guest information found."


def load_guest_dataset():
    """
    Load the guest dataset and create a retriever tool for it.
    Returns:
        GuestInfoRetrieverTool: A tool that retrieves guest information.
    """
    # Load the dataset
    guest_dataset = datasets.load_dataset(
        "agents-course/unit3-invitees", split="train")

    # Convert dataset entries into Document objects
    docs = [
        Document(
            page_content="\n".join([
                f"Name: {guest['name']}",
                f"Relation: {guest['relation']}",
                f"Description: {guest['description']}",
                f"Email: {guest['email']}"
            ]),
            metadata={"name": guest["name"]}
        )
        for guest in guest_dataset
    ]

    # Return the tool
    return GuestInfoRetrieverTool(docs)
