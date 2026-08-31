from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

class QueryExpander:
    def __init__(self, llm=None):
        # Use Gemini 2.5 Flash by default (fast and free tier available)
        self.llm = llm or ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0.3)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an AI research assistant. Given a user query about an academic paper, "
                "generate 3 distinct search query variations to retrieve relevant context:\n"
                "1. A version using technical academic keywords.\n"
                "2. A version focusing on underlying concepts or methodologies.\n"
                "3. A version focusing on metrics, evaluation, or baseline comparisons.\n\n"
                "Return ONLY the 3 variations separated by newlines, with no numbering, bullets, or extra text."
            )),
            ("human", "{query}")
        ])

    def expand_query(self, original_query: str) -> list[str]:
        # 1. Connect prompt template to the LLM using the pipe operator
        chain = self.prompt | self.llm

        # 2. Run the chain with the original query
        response = chain.invoke({"query": original_query})

        # 3. Extract text content and split by newlines
        raw_lines = response.content.strip().split("\n")

        # 4. Clean up blank lines and whitespace
        queries = [line.strip() for line in raw_lines if line.strip()]

        # 5. Always include the original query so you don't lose the user's raw intent!
        if original_query not in queries:
            queries.insert(0, original_query)

        return queries