import json
import re
import os
import logging
from typing import Dict, Any, Union, Optional

# --- Python Built-in ---
import asyncio

# --- Third-Party Imports ---
import uvicorn
import textstat
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from better_profanity import profanity

# --- Google ADK Imports ---
import google.ai.adk as adk
from google.ai.adk.models import Gemini
from google.ai.adk.tools import google_search
from google.ai.adk.config import load_config_from_env

# ==============================================================================
# 1. INITIAL SETUP (Env, Logging)
# ==============================================================================

# Load environment variables (GOOGLE_API_KEY, GOOGLE_CSE_ID)
try:
    load_config_from_env()
    logging.info("Loaded configuration from environment variables.")
except Exception as e:
    logging.warning(f"Could not load .env: {e}. Relying on system environment.")

# Setup professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 2. ADK TOOL DEFINITIONS
# (These are your local functions, now decorated as ADK tools)
# ==============================================================================

@adk.tool
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

@adk.tool
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

@adk.tool
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

@adk.tool
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
# 3. FASTAPI APP INITIALIZATION
# ==============================================================================

app = FastAPI(
    title="Google ADK Content Quality Agent API",
    description="API for running comprehensive quality checks on plain text using Google ADK."
)

# ==============================================================================
# 4. PYDANTIC MODELS
# ==============================================================================

class ValidationRequest(BaseModel):
    """Defines the expected JSON input for the /validate endpoint."""
    plain_text: str

# ==============================================================================
# 5. GLOBAL AGENT INITIALIZATION
# ==============================================================================

logging.info("🚀 Initializing Google ADK Content Quality Agent...")
executor: Optional[adk.AgentExecutor] = None

# This is the system prompt, adapted from your LangChain example.
# It's now instructed to use 'google_search' instead of Tavily.
ADK_SYSTEM_PROMPT = """You are an expert Quality Assurance assistant. Your goal is to provide a complete quality report for a given piece of plain text.

**Your Job:**
1.  You will be given a piece of plain text.
2.  You will use your tools (check_readability_with_textstat, check_professionalism_with_library, etc.) to get reports on the text.
3.  You MUST **perform your own grammar and spelling check** on the original text. Look for spelling mistakes, incorrect punctuation, subject-verb agreement, and incorrect word usage.
4.  You will then synthesize all of this information—the tool outputs AND your own grammar analysis—into the final JSON report.

**Your Plan:**
- You will receive a plain text input.
- You must run these tools:
    1.  check_readability_with_textstat
    2.  check_professionalism_with_library
    3.  check_length_and_structure
    4.  check_redundancy
    5.  google_search (use this *only* if the text makes a specific, verifiable factual claim, e.g., "The sun is 100 miles away").

**FINAL ANSWER (Your Synthesis Step):**
After all tools run, you will receive their outputs. You must then look at the *original text* again, perform your detailed grammar and spelling check, and then generate the final JSON.

**FINAL ANSWER FORMATTING INSTRUCTIONS:**
Your final answer MUST be a single JSON object with the specified structure.
{{
    "overall_score": <An integer score from 1-10, which is the average of all category scores>,
    "category_scores": {{
        "grammar_and_spelling": <Score 1-10, based on YOUR analysis. Deduct points for errors.>,
        "readability": <Score 1-10, from the check_readability_with_textstat tool output>,
        "professionalism_and_tone": <Score 1-10, from the check_professionalism_with_library tool output>,
        "factual_accuracy": <Score 1-10. Default to 10 if no claims to check or if google_search finds no errors.>,
        "redundancy": <Score 1-10, from the check_redundancy tool output>
    }},
    "score_explanations": {{
        "grammar_and_spelling": "<Explanation for grammar score, including a list of errors YOU found (e.g., 'Spelling error: "wrogn" should be "wrong"'). If no errors, say 'No errors found.'>",
        "readability": "<Explanation for readability score, from the check_readability_with_textstat tool output>",
        "professionalism_and_tone": "<Explanation for professionalism score, from the check_professionalism_with_library tool output>",
        "factual_accuracy": "<Explanation for factual accuracy score. If claims were checked, summarize findings.>",
        "redundancy": "<Explanation for redundancy score, from the check_redundancy tool output>"
    }},
    "summary": "<A natural language summary of key issues and suggestions for improvement.>"
}}
"""

try:
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GOOGLE_CSE_ID"):
        raise ValueError("FATAL ERROR: GOOGLE_API_KEY and GOOGLE_CSE_ID must be in .env file.")

    # 1. Initialize the LLM (Using Gemini Flash for speed in an API context)
    llm = Gemini(model_name="gemini-1.5-flash-latest")

    # 2. Gather all local tools
    local_tools = [
        check_length_and_structure,
        check_readability_with_textstat,
        check_professionalism_with_library,
        check_redundancy
    ]
    
    # 3. Initialize the ADK-native Google Search tool
    # This automatically picks up the API key and CSE ID from the environment
    search_tool = google_search.create_from_config()
    
    all_tools = local_tools + [search_tool]

    # 4. Create the ADK Agent
    agent = adk.Agent(
        model=llm,
        tools=all_tools,
        system_instruction=ADK_SYSTEM_PROMPT
    )

    # 5. Create the Agent Executor
    executor = adk.AgentExecutor(agent=agent)
    logging.info("✅ Google ADK Agent Initialized. API is ready.")

except Exception as e:
    logging.critical(f"❌ FATAL ERROR: Failed to initialize Google ADK Agent: {e}")
    executor = None # This will cause the endpoint to fail gracefully

# ==============================================================================
# 6. FASTAPI ENDPOINTS
# ==============================================================================

@app.post("/validate", tags=["Validation"])
async def http_validate_content(request: ValidationRequest):
    """
    Run a comprehensive quality analysis on a piece of text using the Google ADK Agent.
    You must provide a `plain_text` field in the request body.
    """
    if executor is None:
        logging.error("Validator not initialized. Check API key .env setup.")
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys or failed to initialize agent.")

    input_text = request.plain_text
    
    if not input_text or not input_text.strip():
        raise HTTPException(status_code=400, detail="Input 'plain_text' cannot be empty or just whitespace.")

    logging.info("Received plain_text for validation. Invoking ADK agent...")
    
    report_json = None
    try:
        # Use the executor's run_async method.
        # The prompt is just the plain text. The agent's system prompt has the rest.
        result_text = await executor.run_async(prompt=input_text)
        
        # The ADK agent's final output *is* the string, not a dict.
        # We must extract the JSON from it, just as in your LangChain example.
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        
        if json_match:
            report_json = json.loads(json_match.group(0))
        else:
            logging.error(f"No JSON object found in agent output: {result_text}")
            raise json.JSONDecodeError("No JSON object found in agent output.", result_text, 0)

        logging.info(f"Successfully generated report. Overall score: {report_json.get('overall_score')}")
        return report_json

    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"❌ Error parsing agent output: {e}")
        raise HTTPException(status_code=500, detail={"error": "Agent failed to produce a valid JSON report.", "raw_output": result_text})
    except Exception as e:
        logging.error(f"❌ Unexpected agent error: {e}")
        raise HTTPException(status_code=500, detail={"error": f"An unexpected error occurred: {str(e)}"})

# ==============================================================================
# 7. RUN THE SERVER
# ==============================================================================

if __name__ == "__main__":
    # This block is only for direct execution (e.g., `python api_google_adk.py`)
    print("--- Starting Google ADK API server in development mode ---")
    print("API docs available at: http://127.0.0.1:3000/docs")
    print("Press CTRL+C to stop the server")
    
    # Use the name of this file ("api_google_adk")
    uvicorn.run("api_google_adk:app", host="127.0.0.1", port=3000, reload=True)