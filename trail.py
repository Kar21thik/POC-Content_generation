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
# --- CHANGED: Import InvalidId for error handling ---
from bson import ObjectId
from bson.errors import InvalidId
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from better_profanity import profanity

# --- LangChain Imports ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# --- 🚀 Removed AgentExecutor and Tools, they are no longer needed ---
# from langchain.agents import AgentExecutor, create_openai_functions_agent
# from langchain.tools import Tool
# from langchain_community.tools.tavily_search import TavilySearchResults

# ==============================================================================
# 1. INITIAL SETUP (Env, Logging)
# ==============================================================================
# ... (No changes here) ...
# Load environment variables from .env file
load_dotenv()

# Setup professional logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================================================================
# 2. TOOL LOGIC FUNCTIONS
# ==============================================================================
# ... (All check_... functions are here, no changes) ...
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
        level = "Very Confusing "
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
# 3. AGENT CLASS DEFINITION (REFACTORED WITH HALLUCINATION CHECK)
# ==============================================================================

class ContentQualityAgent:
    
    def __init__(self, model: str, temperature: float):
        logging.info(f"Initializing Synthesizing LLM with model={model}, temperature={temperature}")
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.setup_synthesizer()

    def setup_synthesizer(self):
        self.synthesizer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Quality Assurance Synthesizer.
            You will be given the full text of a document AND a set of JSON reports from various quality tools (Readability, Professionalism, etc.).

            **Your Job:**
            1.  You MUST **perform your own grammar and spelling check** on the original text. Look for spelling mistakes, incorrect punctuation, etc.
            2.  You MUST **perform your own factual accuracy (hallucination) check** on the original text. Use your internal knowledge to find claims that are factually incorrect.
            3.  You will then synthesize all of this information—the tool reports AND your own analysis—into a final JSON report.

            **FINAL ANSWER FORMATTING INSTRUCTIONS:**
            Your final answer MUST be a single JSON object with the specified structure.
            {{
                "overall_score": <An integer score from 1-10, which is the average of all category scores>,
                "category_scores": {{
                    "grammar_and_spelling": <Score 1-10, based on YOUR analysis. Deduct points for errors.>,
                    "readability": <Score 1-10, from the ReadabilityCheck tool output>,
                    "professionalism_and_tone": <Score 1-10, from the ProfessionalismCheck tool output>,
                    "redundancy": <Score 1-10, from the RedundancyCheck tool output>,
                    "factual_accuracy": <Score 1-10, based on YOUR analysis. Deduct points for factual errors. Default to 10 if no verifiable claims are made or no errors are found.>
                }},
                "score_explanations": {{
                    "grammar_and_spelling": "<Explanation for grammar score, including a list of errors YOU found (e.g., 'Spelling error: "wrogn" should be "wrong"'). If no errors, say 'No errors found.'>",
                    "readability": "<Explanation for readability score, from the ReadabilityCheck tool output>",
                    "professionalism_and_tone": "<Explanation for professionalism score, from the ProfessionalismCheck tool output>",
                    "redundancy": "<Explanation for redundancy score, from the RedundancyCheck tool output>",
                    "factual_accuracy": "<Explanation for factual accuracy score. List any factual errors YOU found (e.g., 'Factual error: "The sun is green" is incorrect.'). If no errors, say 'No factual errors found.'>"
                }},
                "summary": "<A natural language summary of key issues and suggestions for improvement.>"
            }}
            """),
            ("human", """Here is all the data. Please synthesize the final report.

            **ORIGINAL TEXT:**
            ---
            {input_text}
            ---

            **TOOL REPORTS:**
            ---
            Readability Report:
            {readability_report}

            Professionalism Report:
            {professionalism_report}

            Length/Structure Report:
            {length_report}

            Redundancy Report:
            {redundancy_report}
            ---
            """),
        ])
        
        # Create a simple chain, not a complex agent
        self.synthesizer_chain = self.synthesizer_prompt | self.llm

    async def validate_async(self, input_text: str) -> Dict[str, Any]:
        """
        Runs the full validation process:
        1. Runs all local tools instantly.
        2. Makes ONE LLM call to synthesize the results.
        """
        
        # --- STEP 1: Run all local tools ---
        logging.info("Running local tool: ReadabilityCheck")
        readability_report = check_readability_with_textstat(input_text)
        
        logging.info("Running local tool: ProfessionalismCheck")
        professionalism_report = check_professionalism_with_library(input_text)
        
        logging.info("Running local tool: LengthAndStructureCheck")
        length_report = check_length_and_structure(input_text)
        
        logging.info("Running local tool: RedundancyCheck")
        redundancy_report = check_redundancy(input_text)

        # --- STEP 2: Make ONE LLM call to synthesize everything ---
        logging.info("Making single, final LLM call to synthesize results (with grammar + fact check)...")
        try:
            # We parse the tool outputs to get just the score for the final LLM
            readability_data = json.loads(readability_report)
            professionalism_data = json.loads(professionalism_report)
            redundancy_data = json.loads(redundancy_report)

            # Pass all data to the synthesizer
            result = await self.synthesizer_chain.ainvoke({
                "input_text": input_text,
                "readability_report": readability_report,
                "professionalism_report": professionalism_report,
                "length_report": length_report,
                "redundancy_report": redundancy_report
            })
            
            # --- STEP 3: Parse the final JSON response ---
            raw_output = result.content
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                final_report = json.loads(json_match.group(0))
                
                # --- 🚀 EXTRA LOGIC: Add the tool scores to the final report ---
                if "category_scores" not in final_report:
                    final_report["category_scores"] = {}
                if "score_explanations" not in final_report:
                    final_report["score_explanations"] = {}
                    
                final_report["category_scores"]["readability"] = readability_data.get("readability_score", 5)
                final_report["score_explanations"]["readability"] = readability_data.get("score_explanation", "No explanation.")
                
                final_report["category_scores"]["professionalism_and_tone"] = professionalism_data.get("score", 5)
                final_report["score_explanations"]["professionalism_and_tone"] = professionalism_data.get("score_explanation", "No explanation.")

                final_report["category_scores"]["redundancy"] = redundancy_data.get("score", 5)
                final_report["score_explanations"]["redundancy"] = redundancy_data.get("score_explanation", "No explanation.")

                # Calculate overall score if missing (and ensure LLM-generated scores are present)
                if "grammar_and_spelling" not in final_report["category_scores"]:
                    final_report["category_scores"]["grammar_and_spelling"] = 10 # Default to 10 if LLM forgets
                
                # --- 🚀 NEW: Add default for factual_accuracy ---
                if "factual_accuracy" not in final_report["category_scores"]:
                    final_report["category_scores"]["factual_accuracy"] = 10 # Default to 10
                
                if "overall_score" not in final_report:
                    scores = [
                        final_report["category_scores"].get("grammar_and_spelling", 5),
                        final_report["category_scores"].get("readability", 5),
                        final_report["category_scores"].get("professionalism_and_tone", 5),
                        final_report["category_scores"].get("redundancy", 5),
                        final_report["category_scores"].get("factual_accuracy", 10) # --- 🚀 NEW ---
                    ]
                    final_report["overall_score"] = int(round(sum(scores) / len(scores)))

                return final_report
            else:
                logging.error(f"No JSON object found in synthesizer output: {raw_output}")
                return {"error": "Failed to parse synthesizer JSON output.", "raw_output": raw_output}
        
        except (json.JSONDecodeError, KeyError) as e:
            logging.error(f"❌ Error during synthesis or parsing: {e}")
            return {"error": "Failed to generate a valid report.", "raw_output": str(e)}
        except Exception as e:
            logging.error(f"❌ Unexpected synthesis error: {e}")
            return {"error": f"An unexpected error occurred: {str(e)}", "raw_output": "Synthesis failed."}


# ==============================================================================
# 4. FASTAPI APP INITIALIZATION
# ==============================================================================
# ... (No changes here) ...
app = FastAPI(
    title="Content Quality Agent API",
    description="API for running comprehensive quality checks on text from plain text or MongoDB."
)

# ==============================================================================
# 5. PYDANTIC MODELS
# ==============================================================================
# ... (No changes here) ...
class ValidationRequest(BaseModel):
    plain_text: str
    save_report: bool = False
    source_document_id: Optional[str] = None

class DBValidationRequest(BaseModel):
    source_collection: str
    source_document_id: str
    field_to_check: str

class StructuredDocRequest(BaseModel):
    source_collection: str
    source_document_id: str

# ==============================================================================
# 6. MONGODB CONFIG & HELPERS
# ==============================================================================
# ... (No changes here) ...
DB_NAME = os.getenv("MONGODB_DB_NAME", "quality_db")
REPORTS_COLLECTION = "reports" 

# ==============================================================================
# 7. GLOBAL AGENT INITIALIZATION
# ==============================================================================
# ... (No changes here) ...
logging.info("🚀 Initializing Content Quality Agent for the API...")
validator: Optional[ContentQualityAgent] = None

if not os.getenv("OPENAI_API_KEY"): # --- 🚀 REMOVED TAVILY CHECK, NO LONGER NEEDED ---
    logging.critical("❌ FATAL ERROR: API key OPENAI_API_KEY must be in .env file.")
    validator = None
elif not os.getenv("MONGODB_URL"):
    logging.critical("❌ FATAL ERROR: MONGODB_URL must be in .env file.")
    validator = None
else:
    try:
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
# ... (No changes here) ...
@app.on_event("startup")
async def startup_db_client():
    """Connect to MongoDB on server startup."""
    mongodb_uri = os.getenv("MONGODB_URL")
    if mongodb_uri:
        logging.info(f"Connecting to MongoDB at {mongodb_uri}...")
        app.state.mongodb_client = AsyncIOMotorClient(mongodb_uri)
        app.state.db = app.state.mongodb_client[DB_NAME] 
        
        try:
            await app.state.mongodb_client.server_info()
            logging.info(f"✅ Connected to MongoDB (Database: {DB_NAME})")
        except Exception as e:
            logging.critical(f"❌ Failed to connect to MongoDB: {e}")
            app.state.mongodb_client = None
            app.state.db = None
    else:
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
# ... (No changes here) ...
@app.get("/reports/{report_id}", tags=["Reports"])
async def http_get_report_by_id(report_id: str, fastApiRequest: Request):
    """
    Fetch a single, complete quality report from the 'reports' collection by its ID.
    """
    if fastApiRequest.app.state.db is None:
        raise HTTPException(status_code=500, detail="Database is not configured on the server.")

    db = fastApiRequest.app.state.db
    doc_id = None

    try:
        doc_id = ObjectId(report_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"Invalid ID format. '{report_id}' is not a valid ObjectId.")
    
    logging.info(f"Fetching report with ID: {doc_id}")
    report = await db[REPORTS_COLLECTION].find_one({"_id": doc_id})

    if not report:
        raise HTTPException(status_code=404, detail=f"Report not found with ID {report_id}")

    # Convert ObjectId to string for JSON response
    report["_id"] = str(report["_id"])
    
    return report


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

    if request.save_report:
        if fastApiRequest.app.state.db is None:
            logging.warning("Tried to save report, but DB is not configured. Skipping.")
            report["save_status"] = "failed_db_not_configured"
        else:
            try:
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
    
    This endpoint NOW supports dot notation for 'field_to_check' (e.g., "data.transcript").
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

    # --- 2. Extract the text (with dot-notation fix) ---
    text_to_check = None
    try:
        keys = request.field_to_check.split('.')
        value = document
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                value = None
                break
        text_to_check = value
    except Exception as e:
        logging.error(f"Error accessing nested key '{request.field_to_check}': {e}")
        text_to_check = None

    if text_to_check is None:
        raise HTTPException(status_code=404, detail=f"Field '{request.field_to_check}' not found or is null in document.")
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
        report["save_status"] = f"failed_db_error: {e}"

    return report


@app.post("/validate-structured-doc", tags=["Validation"])
async def http_validate_structured_doc(request: StructuredDocRequest, fastApiRequest: Request):
    # ... (No changes here) ...
    """
    Fetches a single complex document, finds all nested text content,
    runs quality analysis on each piece, saves all reports,
    AND returns a single OVERALL SCORE for the entire document.
    """
    if validator is None:
        raise HTTPException(status_code=500, detail="Server not configured. Missing API keys or failed to init.")
    
    if fastApiRequest.app.state.db is None:
        raise HTTPException(status_code=500, detail="Database is not configured on the server.")

    db = fastApiRequest.app.state.db
    doc_id = None
    
    try:
        doc_id = ObjectId(request.source_document_id)
        document_exists = await db[request.source_collection].find_one(
            {"_id": doc_id}, 
            projection={"_id": 1}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Document ID format: {e}")
        
    if not document_exists:
        raise HTTPException(status_code=404, detail=f"Document not found with ID {request.source_document_id}")

    logging.info(f"Starting efficient crawl for doc {doc_id} from {request.source_collection}")

    pipeline = [
        { "$match": { "_id": doc_id } },
        { "$unwind": { "path": "$data.outline.sections", "preserveNullAndEmptyArrays": False } },
        { "$unwind": { "path": "$data.outline.sections.chapters", "preserveNullAndEmptyArrays": False } },
        { "$unwind": { "path": "$data.outline.sections.chapters.learningContent", "preserveNullAndEmptyArrays": False } },
        { "$match": { 
            "data.outline.sections.chapters.learningContent.type": "text" 
        }},
        { "$match": {
            "data.outline.sections.chapters.learningContent.data.content": { 
                "$exists": True, 
                "$ne": None, 
                "$ne": "" 
            }
        }},
        { "$project": {
            "_id": 0,
            "text_to_check": "$data.outline.sections.chapters.learningContent.data.content",
            "source_section_id": "$data.outline.sections.sectionId",
            "source_section_name": "$data.outline.sections.sectionName",
            "source_chapter_id": "$data.outline.sections.chapters.chapterId",
            "source_chapter_name": "$data.outline.sections.chapters.chapterName"
        }}
    ]

    report_ids = []
    reports_failed = 0
    all_scores = []
    all_summaries = []
    
    try:
        cursor = db[request.source_collection].aggregate(pipeline)
        
        # --- 🚀 WARNING: THIS LOOP IS STILL SLOW! ---
        # It's slow because it calls the (now faster) validator
        # many times in a row.
        #
        # A 40-item crawl will still take:
        # 40 items * 1.5 minutes/item = 60 minutes!
        #
        # We must fix this with parallel processing (asyncio.gather)
        # But for now, the /validate-from-db endpoint is fast.
        
        async for item in cursor:
            text_to_check = item.get("text_to_check")
            
            if not text_to_check:
                continue

            logging.info(f"Validating text from chapter {item.get('source_chapter_id')}")
            
            report = await validator.validate_async(text_to_check) # <-- This is the slow part

            if "error" in report:
                reports_failed += 1
                logging.error(f"Agent failed for sub-content. Raw output: {report.get('raw_output')}")
            else:
                score = report.get("overall_score")
                summary = report.get("summary")
                if score is not None:
                    all_scores.append(score)
                if summary:
                    all_summaries.append(f"Chapter '{item.get('source_chapter_name')}': {summary}")

                try:
                    report_to_save = report.copy()
                    report_to_save["created_at"] = datetime.utcnow()
                    report_to_save["source_text"] = text_to_check
                    report_to_save["source_collection"] = request.source_collection
                    report_to_save["source_document_id"] = request.source_document_id
                    
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

    final_average_score = 0
    if all_scores:
        final_average_score = round(sum(all_scores) / len(all_scores), 2)

    return {
        "status": "Validation crawl complete",
        "source_document_id": request.source_document_id,
        "overall_document_score": final_average_score,
        "reports_generated": len(report_ids),
        "reports_failed": reports_failed,
        "summary_of_issues": all_summaries,
        "report_ids": report_ids
    }


# ==============================================================================
# 10. RUN THE SERVER
# ==============================================================================

if __name__ == "__main__":
    print("--- Starting API server in development mode ---")
    print("API docs available at: http://127.0.0.1:3000/docs")
    print("Press CTRL+C to stop the server")
    uvicorn.run(__name__ + ":app", host="127.0.0.1", port=3000, reload=True)