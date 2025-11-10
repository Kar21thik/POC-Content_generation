import json
import re
import os
import logging
from typing import Dict, Any, Union, Optional

# --- Python Built-in ---
from datetime import datetime
import asyncio

# --- Third-Party Imports ---
import uvicorn
import textstat
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from better_profanity import profanity

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults

# ==============================================================================
# 1. INITIAL SETUP (Env, Logging)
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 2. TOOL LOGIC FUNCTIONS
# (These are all fast, local tools)
# ==============================================================================

def check_length_and_structure(text: str) -> str:
    """Analyzes content length and structure (word count, paragraphs)."""
    logging.info(f"Running LengthAndStructureCheck on text (approx {len(text)} chars)")
    words = text.split()
    word_count = len(words)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    issues = []
    score = 10
    
    if word_count < 15:
        issues.append(f"Content is very short with only {word_count} words.")
        score -= 3
    if word_count > 150 and len(paragraphs) == 1:
        issues.append("Long content is not broken into paragraphs, making it hard to read.")
        score -= 2
    if len(sentences) < 2 and word_count > 20:
        issues.append("Content consists of a single long sentence; consider breaking it up.")
        score -= 2
    
    return json.dumps({
        "word_count": word_count, 
        "sentences": len(sentences), 
        "paragraphs": len(paragraphs), 
        "issues": issues,
        "score": max(1, score),
        "score_explanation": f"Structure score: {max(1, score)}/10. Based on {word_count} words, {len(sentences)} sentences, {len(paragraphs)} paragraphs."
    })

# --- EFFICIENT VERSION: Removed check_grammar_with_llm ---
# The main agent will do this task itself in its single call.

def check_readability_with_textstat(text: str) -> str:
    """Uses the 'textstat' library to calculate objective readability scores."""
    logging.info("Running ReadabilityCheck")
    if not text.strip():
        return json.dumps({"score": 0, "level": "unassessable", "readability_score": 1, "score_explanation": "Cannot assess readability of empty text."})
    
    flesch_score = textstat.flesch_reading_ease(text)
    level = "Very Easy"
    readability_score = 10
    
    if flesch_score < 30: 
        level = "Very Confusing (College Graduate)"
        readability_score = 3
    elif flesch_score < 60: 
        level = "Difficult"
        readability_score = 6
    elif flesch_score < 80: 
        level = "Fairly Easy"
        readability_score = 8
    
    return json.dumps({
        "flesch_reading_ease_score": flesch_score, 
        "level": level,
        "readability_score": readability_score,
        "score_explanation": f"Readability score: {readability_score}/10. Flesch score of {flesch_score:.1f} indicates {level.lower()} reading level."
    })

def check_professionalism_with_library(text: str) -> str:
    """Checks for unprofessional content using the 'better-profanity' library."""
    logging.info("Running ProfessionalismCheck")
    issues = []
    score = 10

    if profanity.contains_profanity(text):
        issues.append("Inappropriate or profane language was found.")
        score -= 5
    if re.search(r'\b[A-Z]{4,}\b', text) and sum(1 for c in text if c.isupper()) / max(1, len(text)) > 0.3:
        issues.append("Excessive capitalization is used, which appears unprofessional.")
        score -= 3
    if re.search(r'[!?@#$%^&*()]{4,}', text):
        issues.append("Excessive punctuation or symbols are used.")
        score -= 2

    return json.dumps({
        "issues": issues,
        "score": max(1, score),
        "score_explanation": f"Professionalism score: {max(1, score)}/10. {len(issues)} professional issues detected."
    })

def check_redundancy(text: str) -> str:
    """Analyzes the text for repetitive sentences and overused words."""
    logging.info("Running RedundancyCheck")
    sentences = [s.lower().strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    issues = []
    score = 10
    
    if len(sentences) > 2:
        unique_sentences = len(set(sentences))
        total_sentences = len(sentences)
        redundancy_ratio = unique_sentences / total_sentences
        if redundancy_ratio < 0.5:
            redundancy_percentage = 100 - (redundancy_ratio * 100)
            issues.append(f"High sentence redundancy detected. {redundancy_percentage:.0f}% of sentences are repetitive.")
            score = max(1, int(redundancy_ratio * 10))
        
        return json.dumps({
            "redundancy_issues": issues,
            "score": score,
            "score_explanation": f"Redundancy score: {score}/10. Unique sentence ratio: {redundancy_ratio:.2f}"
        })
    
    return json.dumps({
        "redundancy_issues": [],
        "score": 10,
        "score_explanation": "Redundancy score: 10/10. Not enough sentences to check for repetition."
    })

# ==============================================================================
# 3. AGENT CLASS DEFINITION
# ==============================================================================

class ContentQualityAgent:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.setup_agent()

    def setup_agent(self):
        # --- EFFICIENT VERSION: Tool list no longer includes the grammar tool ---
        tools = [
            Tool(name="ReadabilityCheck", func=check_readability_with_textstat, description="Use to get the readability score of a piece of text."),
            Tool(name="ProfessionalismCheck", func=check_professionalism_with_library, description="Use to check a piece of text for unprofessional language."),
            Tool(name="LengthAndStructureCheck", func=check_length_and_structure, description="Use to get word count and structure of a piece of text."),
            Tool(name="RedundancyCheck", func=check_redundancy, description="Use to check a piece of text for repetitive sentences."),
            TavilySearchResults(name="FactCheckSearch", max_results=3, description="Use to verify factual claims in a piece of text.")
        ]
        
        # --- EFFICIENT VERSION: Updated prompt ---
        # This new prompt tells the LLM to do the grammar check itself.
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Quality Assurance assistant. Your goal is to provide a complete quality report for a given piece of plain text.

            **Your Job:**
            1.  You will be given a piece of plain text.
            2.  You will use your tools (ReadabilityCheck, ProfessionalismCheck, etc.) to get reports on the text.
            3.  You MUST **perform your own grammar and spelling check** on the original text. Look for spelling mistakes, incorrect punctuation, subject-verb agreement, and incorrect word usage.
            4.  You will then synthesize all of this information—the tool outputs AND your own grammar analysis—into the final JSON report.

            **Your Plan (What the agent executor will do):**
            - You will receive a plain text input.
            - You must run these tools:
                  1.  ReadabilityCheck
                  2.  ProfessionalismCheck
                  3.  LengthAndStructureCheck
                  4.  RedundancyCheck
                  5.  FactCheckSearch (use this *only* if the text makes a specific, verifiable factual claim, e.g., "The sun is 100 miles away").
            
            **FINAL ANSWER (Your Synthesis Step):**
            After all tools run, you will receive their outputs. You must then look at the *original text* again, perform your detailed grammar and spelling check, and then generate the final JSON.

            **FINAL ANSWER FORMATTING INSTRUCTIONS:**
            Your final answer MUST be a single JSON object with the specified structure.
            {{
                "overall_score": <An integer score from 1-10, which is the average of all category scores>,
                "category_scores": {{
                    "grammar_and_spelling": <Score 1-10, based on YOUR analysis. Deduct points for errors.>,
                    "readability": <Score 1-10, from the ReadabilityCheck tool output>,
                    "professionalism_and_tone": <Score 1-10, from the ProfessionalismCheck tool output>,
                    "factual_accuracy": <Score 1-10. Default to 10 if no claims to check or if FactCheckSearch finds no errors.>,
                    "redundancy": <Score 1-10, from the RedundancyCheck tool output>
                }},
                "score_explanations": {{
                    "grammar_and_spelling": "<Explanation for grammar score, including a list of errors YOU found (e.g., 'Spelling error: "wrogn" should be "wrong"'). If no errors, say 'No errors found.'>",
                    "readability": "<Explanation for readability score, from the ReadabilityCheck tool output>",
                    "professionalism_and_tone": "<Explanation for professionalism score, from the ProfessionalismCheck tool output>",
                    "factual_accuracy": "<Explanation for factual accuracy score. If claims were checked, summarize findings.>",
                    "redundancy": "<Explanation for redundancy score, from the RedundancyCheck tool output>"
                }},
                "summary": "<A natural language summary of key issues and suggestions for improvement.>"
            }}
            """),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        
        agent = create_openai_functions_agent(self.llm, tools, prompt)
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    async def validate_async(self, input_text: str) -> Dict[str, Any]:
        """Runs the agent asynchronously on a plain text string."""
        
        # The prompt to the agent executor is just the user's text.
        # The detailed instructions are in the system prompt.
        prompt = f"Please provide a comprehensive quality analysis of the following text:\n---\n{input_text}\n---"
        
        result = {} 
        try:
            # Use ainvoke for async execution
            result = await self.agent_executor.ainvoke({"input": prompt})
            
            # Find the JSON object in the output
            json_match = re.search(r'\{.*\}', result['output'], re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                logging.error(f"No JSON object found in agent output: {result['output']}")
                raise json.JSONDecodeError("No JSON object found in agent output.", result.get('output', ''), 0)
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"❌ Error during agent execution or parsing: {e}")
            return {"error": "Failed to generate a valid report.", "raw_output": result.get('output', 'No output was generated.')}
        except Exception as e:
            logging.error(f"❌ Unexpected agent error: {e}")
            return {"error": f"An unexpected error occurred: {str(e)}", "raw_output": "Agent execution failed."}


# ==============================================================================
# 4. FASTAPI APP INITIALIZATION
# ==============================================================================

app = FastAPI(
    title="Content Quality Agent API",
    description="API for running comprehensive quality checks on plain text."
)

# ==============================================================================
# 5. PYDANTIC MODELS
# ==============================================================================

class ValidationRequest(BaseModel):
    """Defines the expected JSON input for the /validate endpoint."""
    plain_text: str

# ==============================================================================
# 6. MONGODB CONFIG & HELPERS
# ==============================================================================

# --- All MongoDB code removed ---

# ==============================================================================
# 7. GLOBAL AGENT INITIALIZATION
# ==============================================================================

logging.info("🚀 Initializing Content Quality Agent for the API...")
validator: Optional[ContentQualityAgent] = None

if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    logging.critical("❌ FATAL ERROR: API keys (OPENAI_API_KEY, TAVILY_API_KEY) must be in .env file.")
    validator = None
else:
    try:
        validator = ContentQualityAgent()
        logging.info("✅ Agent Initialized. API is ready.")
    except Exception as e:
        logging.exception(f"Failed to initialize ContentQualityAgent: {e}")
        validator = None

# ==============================================================================
# 8. FASTAPI EVENTS
# ==============================================================================

# --- MongoDB startup event removed ---

# ==============================================================================
# 9. FASTAPI ENDPOINTS
# ==============================================================================

# --- Endpoint 1: Validate from plain text ---
@app.post("/validate", tags=["Validation"])
async def http_validate_content(request: ValidationRequest):
    """
    Run a comprehensive quality analysis on a piece of text.
    You must provide a `plain_text` field in the request body.
    """
    if validator is None:
        logging.error("Validator not initialized. Check API key .env setup.")
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys.")

    # Get the text directly. Pydantic already ensures 'plain_text' exists.
    input_to_agent = request.plain_text
    
    if not input_to_agent.strip():
        raise HTTPException(status_code=400, detail="Input 'plain_text' cannot be empty or just whitespace.")

    logging.info("Received plain_text for validation.")
    logging.info("Invoking agent for validation...")
    report = await validator.validate_async(input_to_agent)
    
    if "error" in report:
        logging.error(f"Agent failed to produce valid report. Raw output: {report.get('raw_output')}")
        raise HTTPException(status_code=500, detail=report)
        
    logging.info(f"Successfully generated report. Overall score: {report.get('overall_score')}")
    return report

# ==============================================================================
# 10. RUN THE SERVER
# ==============================================================================

if __name__ == "__main__":
    # This block is only for direct execution (e.g., `python api.py`)
    print("--- Starting API server in development mode ---")
    print("API docs available at: http://127.0.0.1:3000/docs")
    print("Press CTRL+C to stop the server")
    # Note: Using "127.0.0.1" is safer for local dev than "0.0.0.0"
    # The `reload=True` is great for development
    uvicorn.run(__name__ + ":app", host="127.0.0.1", port=3000, reload=True)





# import json
# import re
# import os
# import logging
# from typing import Dict, Any, Union, Optional

# # --- Python Built-in ---
# from datetime import datetime
# import asyncio

# # --- Third-Party Imports ---
# import uvicorn
# import pymongo
# import textstat
# from motor.motor_asyncio import AsyncIOMotorClient
# from bson import ObjectId
# from dotenv import load_dotenv
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from better_profanity import profanity

# # --- LangChain Imports ---
# from langchain_openai import ChatOpenAI
# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.tools import Tool
# from langchain_community.tools.tavily_search import TavilySearchResults

# # ==============================================================================
# # 1. INITIAL SETUP (Env, Logging)
# # ==============================================================================

# # Load environment variables from .env file
# load_dotenv()

# # Setup professional logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # ==============================================================================
# # 2. TOOL LOGIC FUNCTIONS
# # (Must be defined *before* the Agent class)
# # ==============================================================================

# def check_length_and_structure(text: str) -> str:
#     """Analyzes content length and structure (word count, paragraphs)."""
#     logging.info(f"Running LengthAndStructureCheck on text (approx {len(text)} chars)")
#     words = text.split()
#     word_count = len(words)
#     sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
#     paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
#     issues = []
#     score = 10
    
#     if word_count < 15:
#         issues.append(f"Content is very short with only {word_count} words.")
#         score -= 3
#     if word_count > 150 and len(paragraphs) == 1:
#         issues.append("Long content is not broken into paragraphs, making it hard to read.")
#         score -= 2
#     if len(sentences) < 2 and word_count > 20:
#         issues.append("Content consists of a single long sentence; consider breaking it up.")
#         score -= 2
    
#     return json.dumps({
#         "word_count": word_count, 
#         "sentences": len(sentences), 
#         "paragraphs": len(paragraphs), 
#         "issues": issues,
#         "score": max(1, score),
#         "score_explanation": f"Structure score: {max(1, score)}/10. Based on {word_count} words, {len(sentences)} sentences, {len(paragraphs)} paragraphs."
#     })

# def check_grammar_with_llm(text: str) -> str:
#     """Uses a powerful LLM to check for grammar and spelling errors."""
#     logging.info("🧠 Using LLM to check grammar...")
#     # Use a faster, cheaper model for this specific task if desired
#     checker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
#     prompt = f"""
#     You are a highly focused grammar and spelling checker. Your *only* job is to find objective, mechanical errors.
#     -   **LOOK FOR:** Spelling mistakes, incorrect punctuation, subject-verb agreement, run-on sentences, and incorrect word usage (e.g., 'their' vs 'they're').
#     -   **YOU MUST IGNORE:** The text's tone, style, opinions, or any profanity. Another tool is responsible for checking professionalism.
#     Analyze the text below *strictly* for mechanical grammar and spelling errors.
#     If there are no errors, return an empty list and a score of 10.
#     Respond with a JSON object with:
#     -   "errors": ["list of concise error descriptions (e.g., 'Spelling error: "wrogn" should be "wrong"')"]
#     -   "score": integer from 1-10 (10 = perfect, deduct points *only* for each *mechanical* error)
#     -   "score_explanation": brief explanation of why this score was given, based *only* on grammar/spelling.
#     Text: "{text}"
#     """
#     try:
#         response = checker_llm.invoke(prompt)
#         return response.content
#     except Exception as e:
#         logging.error(f"Grammar LLM call failed: {e}")
#         return json.dumps({"errors": ["Grammar check failed to run."], "score": 1, "score_explanation": "Tool execution error."})

# def check_readability_with_textstat(text: str) -> str:
#     """Uses the 'textstat' library to calculate objective readability scores."""
#     logging.info("Running ReadabilityCheck")
#     if not text.strip():
#         return json.dumps({"score": 0, "level": "unassessable", "readability_score": 1, "score_explanation": "Cannot assess readability of empty text."})
    
#     flesch_score = textstat.flesch_reading_ease(text)
#     level = "Very Easy"
#     readability_score = 10
    
#     if flesch_score < 30: 
#         level = "Very Confusing (College Graduate)"
#         readability_score = 3
#     elif flesch_score < 60: 
#         level = "Difficult"
#         readability_score = 6
#     elif flesch_score < 80: 
#         level = "Fairly Easy"
#         readability_score = 8
    
#     return json.dumps({
#         "flesch_reading_ease_score": flesch_score, 
#         "level": level,
#         "readability_score": readability_score,
#         "score_explanation": f"Readability score: {readability_score}/10. Flesch score of {flesch_score:.1f} indicates {level.lower()} reading level."
#     })

# def check_professionalism_with_library(text: str) -> str:
#     """Checks for unprofessional content using the 'better-profanity' library."""
#     logging.info("Running ProfessionalismCheck")
#     issues = []
#     score = 10

#     if profanity.contains_profanity(text):
#         issues.append("Inappropriate or profane language was found.")
#         score -= 5
#     if re.search(r'\b[A-Z]{4,}\b', text) and sum(1 for c in text if c.isupper()) / max(1, len(text)) > 0.3:
#         issues.append("Excessive capitalization is used, which appears unprofessional.")
#         score -= 3
#     if re.search(r'[!?@#$%^&*()]{4,}', text):
#         issues.append("Excessive punctuation or symbols are used.")
#         score -= 2

#     return json.dumps({
#         "issues": issues,
#         "score": max(1, score),
#         "score_explanation": f"Professionalism score: {max(1, score)}/10. {len(issues)} professional issues detected."
#     })

# def check_redundancy(text: str) -> str:
#     """Analyzes the text for repetitive sentences and overused words."""
#     logging.info("Running RedundancyCheck")
#     sentences = [s.lower().strip() for s in re.split(r'[.!?]+', text) if s.strip()]
#     issues = []
#     score = 10
    
#     if len(sentences) > 2:
#         unique_sentences = len(set(sentences))
#         total_sentences = len(sentences)
#         redundancy_ratio = unique_sentences / total_sentences
#         if redundancy_ratio < 0.5:
#             redundancy_percentage = 100 - (redundancy_ratio * 100)
#             issues.append(f"High sentence redundancy detected. {redundancy_percentage:.0f}% of sentences are repetitive.")
#             score = max(1, int(redundancy_ratio * 10))
        
#         return json.dumps({
#             "redundancy_issues": issues,
#             "score": score,
#             "score_explanation": f"Redundancy score: {score}/10. Unique sentence ratio: {redundancy_ratio:.2f}"
#         })
    
#     return json.dumps({
#         "redundancy_issues": [],
#         "score": 10,
#         "score_explanation": "Redundancy score: 10/10. Not enough sentences to check for repetition."
#     })

# # ==============================================================================
# # 3. AGENT CLASS DEFINITION
# # ==============================================================================

# class ContentQualityAgent:
#     def __init__(self, model="gpt-4o-mini", temperature=0):
#         self.llm = ChatOpenAI(model=model, temperature=temperature)
#         self.setup_agent()

#     def setup_agent(self):
#         tools = [
#             Tool(name="GrammarAndSpellingCheck", func=check_grammar_with_llm, description="Use to check a piece of text for spelling and grammatical errors."),
#             Tool(name="ReadabilityCheck", func=check_readability_with_textstat, description="Use to get the readability score of a piece of text."),
#             Tool(name="ProfessionalismCheck", func=check_professionalism_with_library, description="Use to check a piece of text for unprofessional language."),
#             Tool(name="LengthAndStructureCheck", func=check_length_and_structure, description="Use to get word count and structure of a piece of text."),
#             Tool(name="RedundancyCheck", func=check_redundancy, description="Use to check a piece of text for repetitive sentences."),
#             TavilySearchResults(name="FactCheckSearch", max_results=3, description="Use to verify factual claims in a piece of text.")
#         ]
        
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", """You are an expert Quality Assurance assistant. Your goal is to provide a complete quality report for a given piece of plain text.

#             **Your Plan:**
#             - You will receive a plain text input.
#             - You must run a comprehensive set of checks on this text:
#                 1.  GrammarAndSpellingCheck
#                 2.  ReadabilityCheck
#                 3.  ProfessionalismCheck
#                 4.  LengthAndStructureCheck
#                 5.  RedundancyCheck
#                 6.  FactCheckSearch (use this *only* if the text makes a specific, verifiable factual claim, e.g., "The sun is 100 miles away").
            
#             After executing your plan, compile all results into the final JSON report.

#             **FINAL ANSWER FORMATTING INSTRUCTIONS:**
#             Your final answer MUST be a single JSON object with the specified structure.
#             {{
#                 "overall_score": <An integer score from 1-10, which is the average of all category scores>,
#                 "category_scores": {{
#                     "grammar_and_spelling": <Score 1-10>,
#                     "readability": <Score 1-10>,
#                     "professionalism_and_tone": <Score 1-10>,
#                     "factual_accuracy": <Score 1-10. Default to 10 if no claims to check.>,
#                     "redundancy": <Score 1-10>
#                 }},
#                 "score_explanations": {{
#                     "grammar_and_spelling": "<Explanation for grammar score>",
#                     "readability": "<Explanation for readability score>",
#                     "professionalism_and_tone": "<Explanation for professionalism score>",
#                     "factual_accuracy": "<Explanation for factual accuracy score>",
#                     "redundancy": "<Explanation for redundancy score>"
#                 }},
#                 "summary": "<A natural language summary of key issues and suggestions for improvement.>"
#             }}
#             """),
#             ("human", "{input}"),
#             ("placeholder", "{agent_scratchpad}"),
#         ])
        
#         agent = create_openai_functions_agent(self.llm, tools, prompt)
#         self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#     async def validate_async(self, input_text: str) -> Dict[str, Any]:
#         """Runs the agent asynchronously on a plain text string."""
        
#         prompt = f"Please provide a comprehensive quality analysis of the following text:\n---\n{input_text}\n---"
        
#         result = {} 
#         try:
#             # Use ainvoke for async execution
#             result = await self.agent_executor.ainvoke({"input": prompt})
            
#             # Find the JSON object in the output
#             json_match = re.search(r'\{.*\}', result['output'], re.DOTALL)
#             if json_match:
#                 return json.loads(json_match.group(0))
#             else:
#                 logging.error(f"No JSON object found in agent output: {result['output']}")
#                 raise json.JSONDecodeError("No JSON object found in agent output.", result.get('output', ''), 0)
#         except (json.JSONDecodeError, KeyError) as e:
#             logging.error(f"❌ Error during agent execution or parsing: {e}")
#             return {"error": "Failed to generate a valid report.", "raw_output": result.get('output', 'No output was generated.')}
#         except Exception as e:
#             logging.error(f"❌ Unexpected agent error: {e}")
#             return {"error": f"An unexpected error occurred: {str(e)}", "raw_output": "Agent execution failed."}


# # ==============================================================================
# # 4. FASTAPI APP INITIALIZATION
# # ==============================================================================

# app = FastAPI(
#     title="Content Quality Agent API",
#     description="API for running comprehensive quality checks on text from plain text or MongoDB."
# )

# # ==============================================================================
# # 5. PYDANTIC MODELS
# # ==============================================================================

# class ValidationRequest(BaseModel):
#     """Defines the expected JSON input for the /validate endpoint."""
#     plain_text: Optional[str] = None
#     itext_payload: Optional[Dict[str, Any]] = None

# class MongoValidationRequest(BaseModel):
#     """Request model for MongoDB-based validation."""
#     document_id: str
#     content_field: str = "content"  # supports nested like "data.description"
#     save_results: bool = True  # whether to save results back to MongoDB

# class BulkValidationRequest(BaseModel):
#     """Request model for bulk validation."""
#     query_filter: Dict[str, Any] = {}  # raw MongoDB query
#     content_field: str = "content"
#     limit: int = 10
#     save_results: bool = True

# # ==============================================================================
# # 6. MONGODB CONFIG & HELPERS
# # ==============================================================================

# MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
# MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "content_db")
# MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "documents")

# motor_client: Optional[AsyncIOMotorClient] = None

# async def init_mongodb() -> bool:
#     """Initialize MongoDB connection (call on startup)."""
#     global motor_client
#     if motor_client is not None:
#         return True
#     try:
#         motor_client = AsyncIOMotorClient(MONGODB_URL)
#         # simple connectivity check
#         await motor_client.admin.command("ping")
#         logging.info("✅ MongoDB connected successfully")
#         return True
#     except Exception as e:
#         logging.error(f"❌ MongoDB connection failed: {e}")
#         motor_client = None
#         return False

# def get_db():
#     """Get the configured database instance. Raises HTTPException if not ready."""
#     if motor_client is None:
#         logging.error("MongoDB not initialized — call init_mongodb() on startup")
#         raise HTTPException(status_code=500, detail="MongoDB not initialized")

#     # ADD THIS PRINT STATEMENT TO SEE THE TRUTH
#     print(f"--- DEBUG: CONNECTING TO DB: {MONGODB_DB_NAME}, COLLECTION: {MONGODB_COLLECTION} ---")

#     return motor_client[MONGODB_DB_NAME]
# async def fetch_document_by_id(document_id: str, content_field: str = "content"):
#     """
#     Fetch a single document by id.
#     Returns tuple: (document_dict, content_text)
#     """
#     try:
#         db = get_db()
#         collection = db[MONGODB_COLLECTION]
        
#         # This query correctly finds the ID as a string OR an ObjectId
#         query_filters = []
#         query_filters.append({"_id": document_id})
#         if ObjectId.is_valid(document_id):
#             query_filters.append({"_id": ObjectId(document_id)})
#         query = {"$or": query_filters}
        
#         document = await collection.find_one(query)
        
#         if not document:
#             raise HTTPException(status_code=404, detail=f"Document with ID {document_id} not found")

#         # This logic correctly finds fields in nested objects AND arrays
#         if "." in content_field:
#             parts = content_field.split(".")
#             value = document
#             try:
#                 for p in parts:
#                     if isinstance(value, dict):
#                         value = value[p]
#                     elif isinstance(value, list) and p.isdigit():
#                         value = value[int(p)]
#                     else:
#                         raise KeyError
#             except (KeyError, IndexError, TypeError):
#                 value = None
#             content = value
#         else:
#             content = document.get(content_field)

#         if not content or not isinstance(content, str):
#             raise HTTPException(status_code=400, detail=f"Field '{content_field}' not found or is not a string in document")

#         return document, content

#     except HTTPException:
#         raise
#     except Exception as e:
#         logging.exception(f"Error fetching document {document_id}: {e}")
#         raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

# async def save_validation_results(document_id: str, validation_results: Dict[str, Any]) -> bool:
#     """Save validation results back into the document under `quality_analysis`."""
#     try:
#         db = get_db()
#         collection = db[MONGODB_COLLECTION]

#         results_to_save = {
#             "quality_analysis": {
#                 **validation_results,
#                 "analyzed_at": datetime.utcnow().isoformat(),
#                 "analyzer_version": "1.0"
#             }
#         }
        
#         # USE THE ROBUST QUERY HERE AS WELL
#         query_filters = []
#         query_filters.append({"_id": document_id})
#         if ObjectId.is_valid(document_id):
#             query_filters.append({"_id": ObjectId(document_id)})
#         query = {"$or": query_filters}

#         update_result = await collection.update_one(query, {"$set": results_to_save})
#         return update_result.modified_count > 0
#     except Exception as e:
#         logging.exception(f"Error saving validation results for {document_id}: {e}")
#         return False

# # ==============================================================================
# # 7. GLOBAL AGENT INITIALIZATION
# # ==============================================================================

# logging.info("🚀 Initializing Content Quality Agent for the API...")
# validator: Optional[ContentQualityAgent] = None

# if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
#     logging.critical("❌ FATAL ERROR: API keys (OPENAI_API_KEY, TAVILY_API_KEY) must be in .env file.")
#     validator = None
# else:
#     try:
#         validator = ContentQualityAgent()
#         logging.info("✅ Agent Initialized. API is ready.")
#     except Exception as e:
#         logging.exception(f"Failed to initialize ContentQualityAgent: {e}")
#         validator = None

# # ==============================================================================
# # 8. FASTAPI EVENTS
# # ==============================================================================

# @app.on_event("startup")
# async def startup_event():
#     """On app startup, initialize MongoDB connection."""
#     success = await init_mongodb()
#     if not success:
#         logging.error("MongoDB failed to initialize on startup. MongoDB endpoints will fail.")

# # ==============================================================================
# # 9. FASTAPI ENDPOINTS
# # ==============================================================================

# # --- Endpoint 1: Validate from plain text ---
# @app.post("/validate", tags=["Validation"])
# async def http_validate_content(request: ValidationRequest):
#     """
#     Run a comprehensive quality analysis on a piece of text.
#     You must provide *either* a `plain_text` field or an `itext_payload` field.
#     """
#     if validator is None:
#         logging.error("Validator not initialized. Check API key .env setup.")
#         raise HTTPException(status_code=500, detail="Server not configured. Missing API keys.")

#     input_to_agent = ""
    
#     if request.itext_payload:
#         logging.info("Received iText payload for validation.")
#         try:
#             # Simple extractor for a specific iText format
#             input_to_agent = request.itext_payload["data"][0]["data"]["content"]
#         except (KeyError, IndexError, TypeError):
#             logging.error("Invalid iText payload structure received.")
#             raise HTTPException(status_code=400, detail="Invalid iText payload structure. Could not find 'data[0].data.content' field.")
    
#     elif request.plain_text:
#         logging.info("Received plain_text for validation.")
#         input_to_agent = request.plain_text
    
#     else:
#         logging.warning("Validation request received without plain_text or itext_payload.")
#         raise HTTPException(status_code=400, detail="You must provide either 'plain_text' or 'itext_payload' in the request body.")
    
#     if not input_to_agent:
#         raise HTTPException(status_code=400, detail="Input content is empty.")

#     logging.info("Invoking agent for validation...")
#     report = await validator.validate_async(input_to_agent)
    
#     if "error" in report:
#         logging.error(f"Agent failed to produce valid report. Raw output: {report.get('raw_output')}")
#         raise HTTPException(status_code=500, detail=report)
        
#     logging.info(f"Successfully generated report. Overall score: {report.get('overall_score')}")
#     return report

# # --- Endpoint 2: Validate from MongoDB ID ---
# @app.post("/validate-from-mongodb", tags=["MongoDB Validation"])
# async def validate_from_mongodb(request: MongoValidationRequest):
#     """
#     Fetch content from MongoDB, run quality analysis,
#     optionally save results back to MongoDB, and return the report.
#     """
#     if validator is None:
#         raise HTTPException(status_code=500, detail="Server not configured. Missing API keys or validator initialization failed.")

#     try:
#         logging.info(f"Fetching document {request.document_id} from MongoDB (field={request.content_field})...")
#         document, content = await fetch_document_by_id(request.document_id, request.content_field)
#         logging.info(f"Found content length: {len(content)} characters")

#         logging.info("Running quality analysis...")
#         report = await validator.validate_async(content)

#         if "error" in report:
#              logging.error(f"Agent failed to produce valid report for doc {request.document_id}. Raw output: {report.get('raw_output')}")
#              raise HTTPException(status_code=500, detail=report)

#         # Attach document metadata to report
#         report_meta = report.copy()
#         report_meta["source"] = {
#             "document_id": request.document_id,
#             "content_field": request.content_field,
#             "analyzed_at": datetime.utcnow().isoformat(),
#             "content_length": len(content),
#             "content_preview": (content[:200] + "...") if len(content) > 200 else content
#         }

#         # Optionally save results back to MongoDB
#         if request.save_results:
#             saved = await save_validation_results(request.document_id, report_meta)
#             report_meta["results_saved"] = saved
#             if saved:
#                 logging.info(f"✅ Validation results saved to MongoDB for {request.document_id}")
#             else:
#                 logging.warning(f"⚠️ Validation results were NOT saved to MongoDB for {request.document_id}")
        
#         logging.info(f"✅ Analysis complete for {request.document_id}. Overall score: {report.get('overall_score')}")
#         return report_meta

#     except HTTPException:
#         raise  # Re-raise known HTTP exceptions
#     except Exception as e:
#         logging.exception(f"Unexpected error during MongoDB validation: {e}")
#         raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

# # --- Endpoint 3: Bulk Validate from MongoDB Query ---
# @app.post("/validate-bulk-from-mongodb", tags=["MongoDB Validation"])
# async def validate_bulk_from_mongodb(request: BulkValidationRequest):
#     """
#     Bulk validate documents matching query_filter. Processes up to `limit` documents.
#     """
#     if validator is None:
#         raise HTTPException(status_code=500, detail="Server not configured. Missing API keys or validator initialization failed.")

#     try:
#         db = get_db()
#         collection = db[MONGODB_COLLECTION]
        
#         logging.info(f"Starting bulk validation for query: {request.query_filter}, limit: {request.limit}")
#         cursor = collection.find(request.query_filter).limit(int(request.limit))
#         processed = []
        
#         async for doc in cursor:
#             doc_id = str(doc.get("_id"))
#             content = None
            
#             try:
#                 # Use helper to get content and handle nested fields/errors
#                 _, content = await fetch_document_by_id(doc_id, request.content_field)
#             except HTTPException as e:
#                 logging.warning(f"Skipping {doc_id} (fetch failed): {e.detail}")
#                 processed.append({"document_id": doc_id, "error": e.detail})
#                 continue
            
#             if not content:
#                 logging.warning(f"Skipping {doc_id}: No content found.")
#                 processed.append({"document_id": doc_id, "error": "Content field was empty or null."})
#                 continue

#             # Run validation for this document
#             report = await validator.validate_async(content)
#             report_meta = report.copy()
#             report_meta["source"] = {
#                 "document_id": doc_id,
#                 "content_field": request.content_field,
#                 "analyzed_at": datetime.utcnow().isoformat(),
#                 "content_length": len(content)
#             }
            
#             if "error" in report:
#                 logging.warning(f"Skipping {doc_id} (analysis failed): {report.get('error')}")
#                 processed.append({"document_id": doc_id, "error": report.get('error', 'Analysis failed')})
#                 continue

#             if request.save_results:
#                 saved = await save_validation_results(doc_id, report_meta)
#                 report_meta["results_saved"] = saved

#             processed.append({"document_id": doc_id, "result": report_meta})

#         logging.info(f"Bulk validation complete. Processed {len(processed)} documents.")
#         return {"processed_count": len(processed), "items": processed}

#     except Exception as e:
#         logging.exception("Bulk validation failed: %s", e)
#         raise HTTPException(status_code=500, detail=str(e))

# # ==============================================================================
# # 10. RUN THE SERVER
# # ==============================================================================

# if __name__ == "__main__":
#     # This block is only for direct execution (e.g., `python api.py`)
#     print("--- Starting API server in development mode ---")
#     print("API docs available at: http://127.0.0.1:3000/docs")
#     print("Press CTRL+C to stop the server")
#     # Note: Using "127.0.0.1" is safer for local dev than "0.0.0.0"
#     # The `reload=True` is great for development
#     uvicorn.run(__name__ + ":app", host="127.0.0.1", port=3000, reload=True)