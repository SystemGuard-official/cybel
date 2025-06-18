from src.llm.openai_llm import OpenAIClient
from src.llm.groq_llm import GroqAIClient

class LLMFactory:
    @staticmethod
    def get_client(provider: str):
        provider = provider.lower()
        if provider == "openai":
            return OpenAIClient()
        elif provider == "groq":
            return GroqAIClient()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
