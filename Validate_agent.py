import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# MODIFIED: A much stricter prompt template
prompt_template = """
You are a strict Quality Control agent. Analyze the text chunk based on the following criteria and provide a score from 1 (unacceptable) to 10 (excellent). The chunk is only valid if it scores 7 or higher.

CRITERIA:
1.  **Grammar and Spelling**: Must be grammatically correct with no spelling errors.
2.  **Clarity**: The meaning must be clear and easy to understand. Avoid vague language or excessive jargon.
3.  **Redundancy**: Should not contain repetitive phrases or redundant sentences.
4.  **Profanity**: Must not contain offensive words.
5.  **Formatting**: Must not contain excessive capitalization or repeated symbols.

Return a single JSON object with:
- "score": An integer from 1 to 10.
- "is_valid": A boolean (true if score >= 7, otherwise false).
- "issues": A list of specific issues found (empty if valid).

Text chunk:
"{chunk}"
"""

prompt = PromptTemplate(input_variables=["chunk"], template=prompt_template)
validation_chain = LLMChain(llm=llm, prompt=prompt)

# Function to validate large text
def validate_large_text(text: str, chunk_size=450, overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    
    results = []
    for i, chunk in enumerate(chunks, start=1):
        # The LLM returns a string, so we need to parse it into a dictionary
        result_str = validation_chain.run(chunk=chunk)
        try:
            # MODIFIED: Parse the JSON string into a Python dictionary
            result_json = json.loads(result_str)
            results.append({"chunk": i, "result": result_json})
        except json.JSONDecodeError:
            # Handle cases where the LLM might return a malformed response
            results.append({"chunk": i, "result": {"error": "Invalid JSON response", "raw_output": result_str}})
    
    return results

# Example usage with the "invalid" text
if __name__ == "__main__":
    big_text = """
Understanding your audience is a fundamental aspect of effective sales communication. It involves recognizing the needs, preferences, and behaviors of potential customers to tailor your sales strategies accordingly. In today's competitive market, sales professionals must go beyond basic demographic information and delve into psychographics, which include values, interests, and lifestyle choices. This deeper understanding allows salespeople to create personalized messages that resonate with their audience, ultimately leading to higher engagement and conversion rates.

Audience analysis is crucial for several reasons and it is very important. First, it helps in identifying the pain points. These are the problems customers face. By understanding their problems, sales professionals can position their products or services as solutions to their problems. Secondly, audience analysis helps segment the market. For instance, a company who sell fitness equipment may target people who are healthy. This is an example of targeting a segment.

There is many methods to gain insights and get data. Surveys are a tool. Online analytics can also be used, these tools track users and what they do. Additionally, customer interviews are a method for getting insights that are qualitative, which is data that is not quantitative. This allows for a more nuanced understanding of the audience you are trying to understand. Practicing empathy helps to connect and it fosters relationships for the long-term, which is good. Empathy is key.

In conclusion, understanding the audience is the foundation. It is critical. Synergyzing core competencies with next-gen paradigms will allow for the leveraging of impactful frameworks. This proactive approach to customer-centricity is mission-critical for optimizing value-added deliverables and ensuring a holistic and robust sales pipeline going forward. It's about thinking outside the box.
"""
    validation_results = validate_large_text(big_text)
    
    for r in validation_results:
        print(f"Chunk {r['chunk']} ->")
        print(json.dumps(r['result'], indent=2))
        print("-" * 20)