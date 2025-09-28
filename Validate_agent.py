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



"""
FOR THE PROPER CORRECT BIG SENTENCE USE THIS:
Understanding your audience is a fundamental aspect of effective sales communication. It involves recognizing the needs, preferences, and behaviors of potential customers to tailor your sales strategies accordingly. In today's competitive market, sales professionals must go beyond basic demographic information and delve into psychographics, which include values, interests, and lifestyle choices. This deeper understanding allows salespeople to create personalized messages that resonate with their audience, ultimately leading to higher engagement and conversion rates.

Audience analysis is crucial for several reasons. First, it helps in identifying the pain points and motivations of potential customers. By understanding what challenges they face, sales professionals can position their products or services as solutions. Second, audience analysis aids in segmenting the market, allowing for targeted marketing efforts. For instance, a company selling fitness equipment may target health-conscious individuals aged 25-40, tailoring their messaging to highlight the benefits of staying fit and healthy. Third, understanding audience behavior can lead to more effective communication strategies, ensuring that messages are delivered through the right channels at the right times.

There are several methods to gain insights into your audience. Surveys and questionnaires are effective tools for collecting direct feedback from potential customers. Online analytics tools can track user behavior on websites and social media platforms, providing valuable data on preferences and engagement. Social listening tools can also help sales professionals monitor conversations about their brand or industry, revealing customer sentiments and emerging trends. Additionally, customer interviews and focus groups can provide qualitative insights that quantitative data may not capture, allowing for a more nuanced understanding of audience needs.

Creating customer personas is an effective strategy for understanding your audience. A customer persona is a semi-fictional representation of your ideal customer based on market research and real data about your existing customers. To create a persona, gather information such as demographics, interests, pain points, and buying behaviors. For instance, a persona for a luxury skincare brand might include a 35-year-old woman who values high-quality ingredients and sustainability. By developing these personas, sales teams can tailor their messaging and marketing strategies to better align with the needs and preferences of their target audience.

Empathy plays a crucial role in understanding your audience. It involves putting yourself in the customer's shoes and genuinely understanding their feelings and perspectives. When sales professionals practice empathy, they can better connect with their audience, build trust, and foster long-term relationships. This connection can lead to increased customer loyalty and repeat business. For example, a salesperson who empathizes with a customer's frustration over a product issue can address their concerns more effectively, turning a negative experience into a positive one. Empathy not only enhances communication but also helps in identifying solutions that truly meet customer needs.

In conclusion, understanding your audience is a critical foundation of effective sales communication. By analyzing demographics and psychographics, utilizing various tools, and practicing empathy, sales professionals can create meaningful connections with their customers. This understanding not only enhances communication but also drives sales success by aligning products and services with the specific needs of the target audience.
"""