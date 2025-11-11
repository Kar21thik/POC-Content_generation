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
import pymongo
import textstat
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
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
# --- CHANGED: Read log level from .env, default to INFO ---
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

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
    # --- CHANGED: Constructor now takes model and temp as arguments ---
    def __init__(self, model: str, temperature: float):
        logging.info(f"Initializing LLM with model={model}, temperature={temperature}")
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.setup_agent()

    def setup_agent(self):
        tools = [
            Tool(name="ReadabilityCheck", func=check_readability_with_textstat, description="Use to get the readability score of a piece of text."),
            Tool(name="ProfessionalismCheck", func=check_professionalism_with_library, description="Use to check a piece of text for unprofessional language."),
            Tool(name="LengthAndStructureCheck", func=check_length_and_structure, description="Use to get word count and structure of a piece of text."),
            Tool(name="RedundancyCheck", func=check_redundancy, description="Use to check a piece of text for repetitive sentences."),
            TavilySearchResults(name="FactCheckSearch", max_results=3, description="Use to verify factual claims in a piece of text.")
        ]
        
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
        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=(log_level == "DEBUG")) # <-- CHANGED: Verbose if DEBUG

    async def validate_async(self, input_text: str) -> Dict[str, Any]:
        """Runs the agent asynchronously on a plain text string."""
        
        prompt = f"Please provide a comprehensive quality analysis of the following text:\n---\n{input_text}\n---"
        
        result = {} 
        try:
            result = await self.agent_executor.ainvoke({"input": prompt})
            
            # --- Robust JSON parsing ---
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
    description="API for running comprehensive quality checks on text from plain text or MongoDB."
)

# ==============================================================================
# 5. PYDANTIC MODELS
# ==============================================================================

class ValidationRequest(BaseModel):
    """Defines the expected JSON input for the /validate endpoint."""
    plain_text: str
    save_report: bool = False
    source_document_id: Optional[str] = None

class DBValidationRequest(BaseModel):
    """Defines the input for the /validate-from-db endpoint."""
    source_collection: str
    source_document_id: str
    field_to_check: str

# --- 🚀 NEW CODE ADDED HERE ---
class StructuredDocRequest(BaseModel):
    """Defines the input for the /validate-structured-doc endpoint."""
    source_collection: str
    source_document_id: str
# --- END OF NEW CODE ---

# ==============================================================================
# 6. MONGODB CONFIG & HELPERS
# ==============================================================================

# --- CHANGED: Read DB name from .env, default to "quality_db" ---
DB_NAME = os.getenv("MONGODB_DB_NAME", "quality_db")
REPORTS_COLLECTION = "reports" # This is the collection where reports will be SAVED

# ==============================================================================
# 7. GLOBAL AGENT INITIALIZATION
# ==============================================================================

logging.info("🚀 Initializing Content Quality Agent for the API...")
validator: Optional[ContentQualityAgent] = None

# Check for all required API keys
if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
    logging.critical("❌ FATAL ERROR: API keys (OPENAI_API_KEY, TAVILY_API_KEY) must be in .env file.")
    validator = None
# --- CHANGED: Check for MONGODB_URL ---
elif not os.getenv("MONGODB_URL"):
    logging.critical("❌ FATAL ERROR: MONGODB_URL must be in .env file.")
    validator = None
else:
    try:
        # --- CHANGED: Load model config from .env ---
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        temperature = float(os.getenv("TEMPERATURE", 0.0))
        
        validator = ContentQualityAgent(model=model_name, temperature=temperature)
        logging.info(f"✅ Agent Initialized (Model: {model_name}, Temp: {temperature}). API is ready.")
    except Exception as e:
        logging.exception(f"Failed to initialize ContentQualityAgent: {e}")
        validator = None

# ==============================================================================
# 8. FASTAPI EVENTS
# ==============================================================================

@app.on_event("startup")
async def startup_db_client():
    """Connect to MongoDB on server startup."""
    # --- CHANGED: Read MONGODB_URL ---
    mongodb_uri = os.getenv("MONGODB_URL")
    if mongodb_uri:
        logging.info(f"Connecting to MongoDB at {mongodb_uri}...")
        app.state.mongodb_client = AsyncIOMotorClient(mongodb_uri)
        app.state.db = app.state.mongodb_client[DB_NAME] # DB_NAME is now from your .env
        
        # Try to connect and log server info to confirm connection
        try:
            await app.state.mongodb_client.server_info()
            logging.info(f"✅ Connected to MongoDB (Database: {DB_NAME})")
        except Exception as e:
            logging.critical(f"❌ Failed to connect to MongoDB: {e}")
            app.state.mongodb_client = None
            app.state.db = None
    else:
        # --- CHANGED: Warning for MONGODB_URL ---
        logging.warning("MONGODB_URL not set. Database functionality will be disabled.")
        app.state.mongodb_client = None
        app.state.db = None

@app.on_event("shutdown")
async def shutdown_db_client():
    """Disconnect from MongoDB on server shutdown."""
    if app.state.mongodb_client:
        logging.info("Disconnecting from MongoDB...")
        app.state.mongodb_client.close()
        logging.info("✅ Disconnected from MongoDB.")


# ==============================================================================
# 9. FASTAPI ENDPOINTS
# ==============================================================================

@app.post("/validate", tags=["Validation"])
async def http_validate_content(request: ValidationRequest, fastApiRequest: Request):
    """
    Run a comprehensive quality analysis on a piece of text.
    Set `save_report: true` in the request body to save the report to MongoDB.
    """
    if validator is None:
        logging.error("Validator not initialized. Check API key .env setup.")
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys or failed to init.")

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

    # --- Save report to MongoDB if requested ---
    if request.save_report:
        if fastApiRequest.app.state.db is None:
            logging.warning("Tried to save report, but DB is not configured. Skipping.")
            report["save_status"] = "failed_db_not_configured"
        else:
            try:
                # Add metadata to the report before saving
                report_to_save = report.copy()
                report_to_save["created_at"] = datetime.utcnow()
                report_to_save["source_text"] = input_to_agent
                if request.source_document_id:
                    report_to_save["source_document_id"] = request.source_document_id
                
                insert_result = await fastApiRequest.app.state.db[REPORTS_COLLECTION].insert_one(report_to_save)
                report["report_id"] = str(insert_result.inserted_id)
                report["save_status"] = "success"
                logging.info(f"Report saved to MongoDB with ID: {report['report_id']}")
            
            except Exception as e:
                logging.error(f"Failed to save report to MongoDB: {e}")
                report["save_status"] = f"failed_db_error: {e}"

    return report


@app.post("/validate-from-db", tags=["Validation"])
async def http_validate_from_db(request: DBValidationRequest, fastApiRequest: Request):
    """
    Fetch a document from MongoDB, run quality analysis on a field,
    and save the report to the 'reports' collection.
    
    Example:
    {
      "source_collection": "bulk_generate_content_metadata",
      "source_document_id": "your_mongo_document_id",
      "field_to_check": "your_field_with_text"
    }
    """
    if validator is None:
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys or failed to init.")
    
    if fastApiRequest.app.state.db is None:
        raise HTTPException(status_code=500, detail="Database is not configured on the server.")

    db = fastApiRequest.app.state.db
    doc_id = None

    # --- 1. Fetch the document ---
    try:
        doc_id = ObjectId(request.source_document_id)
        logging.info(f"Fetching doc {doc_id} from collection {request.source_collection}")
        document = await db[request.source_collection].find_one({"_id": doc_id})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Document ID format or DB error: {e}")
        
    if not document:
        raise HTTPException(status_code=404, detail=f"Document not found with ID {request.source_document_id} in collection {request.source_collection}")

    # --- 2. Extract the text ---
    text_to_check = document.get(request.field_to_check)
    
    if text_to_check is None:
        raise HTTPException(status_code=404, detail=f"Field '{request.field_to_check}' not found in document.")
    if not isinstance(text_to_check, str) or not text_to_check.strip():
        raise HTTPException(status_code=400, detail=f"Field '{request.field_to_check}' is empty or not a string.")

    # --- 3. Run Validation ---
    logging.info(f"Invoking agent for DB document: {request.source_document_id}")
    report = await validator.validate_async(text_to_check)

    if "error" in report:
        logging.error(f"Agent failed for DB document. Raw output: {report.get('raw_output')}")
        raise HTTPException(status_code=500, detail=report)

    # --- 4. Save the report to MongoDB ---
    logging.info(f"Saving report for document: {request.source_document_id}")
    try:
        report_to_save = report.copy()
        report_to_save["created_at"] = datetime.utcnow()
        report_to_save["source_text"] = text_to_check
        report_to_save["source_collection"] = request.source_collection
        report_to_save["source_document_id"] = request.source_document_id
        
        insert_result = await db[REPORTS_COLLECTION].insert_one(report_to_save)
        report["report_id"] = str(insert_result.inserted_id)
        report["save_status"] = "success"
        logging.info(f"Report saved to MongoDB with ID: {report['report_id']}")
    
    except Exception as e:
        logging.error(f"Failed to save report to MongoDB: {e}")
        # Even if saving fails, return the report to the user
        report["save_status"] = f"failed_db_error: {e}"

    return report


# --- 🚀 NEW (REFACTORED) CODE ---
# The function below has been rewritten to use an Aggregation Pipeline,
# removing the inefficient nested Python loops.

@app.post("/validate-structured-doc", tags=["Validation"])
async def http_validate_structured_doc(request: StructuredDocRequest, fastApiRequest: Request):
    """
    Fetches a single complex document from MongoDB, finds all text content
    within its nested outline using an efficient Aggregation Pipeline, 
    runs quality analysis on each piece, and saves a report for each one.
    """
    if validator is None:
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys or failed to init.")
    
    if fastApiRequest.app.state.db is None:
        raise HTTPException(status_code=500, detail="Database is not configured on the server.")

    db = fastApiRequest.app.state.db
    doc_id = None
    
    # --- 1. Validate Document ID and Check Existence ---
    try:
        doc_id = ObjectId(request.source_document_id)
        # We still fetch the main doc just to ensure it exists before aggregating
        document_exists = await db[request.source_collection].find_one(
            {"_id": doc_id}, 
            projection={"_id": 1}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Document ID format: {e}")
        
    if not document_exists:
        raise HTTPException(status_code=404, detail=f"Document not found with ID {request.source_document_id}")

    logging.info(f"Starting efficient crawl for doc {doc_id} from {request.source_collection}")

    # --- 2. Define the Aggregation Pipeline ---
    # This pipeline deconstructs the nested arrays on the SERVER-SIDE,
    # filters for only the text content we want, and returns a flat list.
    
    pipeline = [
        # Stage 1: Match the specific document
        { "$match": { "_id": doc_id } },
        
        # Stage 2: Deconstruct the 'sections' array
        { "$unwind": { "path": "$data.outline.sections", "preserveNullAndEmptyArrays": False } },
        
        # Stage 3: Deconstruct the 'chapters' array
        { "$unwind": { "path": "$data.outline.sections.chapters", "preserveNullAndEmptyArrays": False } },
        
        # Stage 4: Deconstruct the 'learningContent' array
        { "$unwind": { "path": "$data.outline.sections.chapters.learningContent", "preserveNullAndEmptyArrays": False } },
        
        # Stage 5: Filter for *only* 'text' type content
        { "$match": { 
            "data.outline.sections.chapters.learningContent.type": "text" 
        }},
        
        # Stage 6: Filter for *only* items that have non-empty text
        { "$match": {
            "data.outline.sections.chapters.learningContent.data.content": { 
                "$exists": True, 
                "$ne": None, 
                "$ne": "" 
            }
        }},
        
        # Stage 7: Project into a clean, flat list for our loop
        { "$project": {
            "_id": 0, # Exclude the default _id
            "text_to_check": "$data.outline.sections.chapters.learningContent.data.content",
            
            # Include 'breadcrumbs' so we know where the text came from
            "source_section_id": "$data.outline.sections.sectionId",
            "source_section_name": "$data.outline.sections.sectionName",
            "source_chapter_id": "$data.outline.sections.chapters.chapterId",
            "source_chapter_name": "$data.outline.sections.chapters.chapterName"
        }}
    ]

    # --- 3. Execute the Pipeline and Process Results ---
    report_ids = []
    reports_failed = 0
    
    try:
        # The cursor now returns a flat list, no more nested loops!
        cursor = db[request.source_collection].aggregate(pipeline)
        
        # We now have ONE simple loop instead of three nested ones.
        async for item in cursor:
            text_to_check = item.get("text_to_check")
            
            if not text_to_check: # Should be caught by pipeline, but as a safeguard
                continue

            logging.info(f"Validating text from chapter {item.get('source_chapter_id')}")
            
            # --- 4. Run Validation on this piece of text ---
            report = await validator.validate_async(text_to_check)

            if "error" in report:
                reports_failed += 1
                logging.error(f"Agent failed for sub-content. Raw: {report.get('raw_output')}")
            else:
                # --- 5. Save the individual report ---
                try:
                    report_to_save = report.copy()
                    report_to_save["created_at"] = datetime.utcnow()
                    report_to_save["source_text"] = text_to_check
                    report_to_save["source_collection"] = request.source_collection
                    report_to_save["source_document_id"] = request.source_document_id
                    
                    # Add the metadata 'breadcrumbs' from our pipeline
                    report_to_save["source_section_id"] = item.get("source_section_id")
                    report_to_save["source_section_name"] = item.get("source_section_name")
                    report_to_save["source_chapter_id"] = item.get("source_chapter_id")
                    report_to_save["source_chapter_name"] = item.get("source_chapter_name")
                    
                    insert_result = await db[REPORTS_COLLECTION].insert_one(report_to_save)
                    report_ids.append(str(insert_result.inserted_id))
                except Exception as save_e:
                    reports_failed += 1
                    logging.error(f"Failed to SAVE report for chapter {item.get('source_chapter_id')}: {save_e}")

    except Exception as e:
        logging.exception(f"Error while processing aggregation for document {request.source_document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed during aggregation/crawl: {str(e)}")

    logging.info(f"Crawl complete. Generated {len(report_ids)} reports. Failed: {reports_failed}")

    return {
        "status": "Validation crawl complete",
        "source_document_id": request.source_document_id,
        "reports_generated": len(report_ids),
        "reports_failed": reports_failed,
        "report_ids": report_ids
    }
# --- END OF REFACTORED CODE ---


# ==============================================================================
# 10. RUN THE SERVER
# ==============================================================================

if __name__ == "__main__":
    # This block is only for direct execution (e.g., `python api.py`)
    print("--- Starting API server in development mode ---")
    print("API docs available at: http://127.0.0.1:3000/docs")
    print("Press CTRL+C to stop the server")
    uvicorn.run(__name__ + ":app", host="127.0.0.1", port=3000, reload=True)