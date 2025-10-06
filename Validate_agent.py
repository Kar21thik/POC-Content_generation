import json
import re
import os
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ContentQualityValidator:
    def __init__(self, model="gpt-4o-mini", temperature=0):
        """Initialize the content quality validator with OpenAI LLM."""
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.setup_validation_chain()
    
    def setup_validation_chain(self):
        """Setup the LangChain validation chain with comprehensive prompt."""
        prompt_template = """
You are an expert Content Quality Control agent. Analyze the provided text chunk and evaluate it based on comprehensive quality criteria.

EVALUATION CRITERIA (Rate each 1-10):

1. **LENGTH & STRUCTURE**: 
   - Appropriate content length
   - Logical organization and flow
   - Clear paragraph structure
   - Proper content hierarchy

2. **GRAMMAR & LANGUAGE**:
   - Perfect grammar and spelling
   - Correct punctuation and capitalization
   - Proper sentence structure
   - Professional language use

3. **CLARITY & READABILITY**:
   - Clear, understandable language
   - Appropriate vocabulary level
   - Good sentence variety
   - Easy to follow content

4. **CONTENT QUALITY**:
   - No redundancy or unnecessary repetition
   - Factual accuracy (flag suspicious claims)
   - Relevant and focused content
   - No hallucinations or false information

5. **FORMATTING & PROFESSIONALISM**:
   - No profanity or inappropriate language
   - Professional tone throughout
   - Proper formatting (no excessive caps/symbols)
   - Consistent writing style

SCORING SYSTEM:
- 9-10: Excellent (Publication ready)
- 7-8: Good (Minor improvements needed)
- 5-6: Average (Moderate issues to address)
- 3-4: Poor (Major improvements required)
- 1-2: Unacceptable (Complete revision needed)

IMPORTANT: Return ONLY a valid JSON object. No additional text or formatting.

{{
    "overall_score": <integer 1-10>,
    "is_valid": <true if overall_score >= 7, false otherwise>,
    "category_scores": {{
        "length_structure": <integer 1-10>,
        "grammar_language": <integer 1-10>,
        "clarity_readability": <integer 1-10>,
        "content_quality": <integer 1-10>,
        "formatting_professionalism": <integer 1-10>
    }},
    "issues": [
        {{
            "category": "<category_name>",
            "severity": "<high|medium|low>",
            "description": "<specific issue description>"
        }}
    ],
    "suggestions": [
        "<specific improvement suggestion>"
    ],
    "word_count": <integer>,
    "readability_level": "<beginner|intermediate|advanced>",
    "estimated_reading_time": "<X minutes>"
}}

Text to evaluate:
"{chunk}"
"""
        
        self.prompt = PromptTemplate(input_variables=["chunk"], template=prompt_template)
        self.validation_chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def basic_pre_validation(self, text: str) -> Dict[str, Any]:
        """Perform basic validation checks before LLM analysis."""
        issues = []
        word_count = len(text.split())
        char_count = len(text.strip())
        
        # Critical validation checks
        if not text.strip():
            issues.append({
                "category": "length_structure",
                "severity": "high",
                "description": "Content is empty or contains only whitespace"
            })
            return {
                "word_count": 0,
                "char_count": 0,
                "critical_issues": issues,
                "can_proceed": False
            }
        
        # Length validation
        if word_count < 5:
            issues.append({
                "category": "length_structure",
                "severity": "high",
                "description": f"Content too short: {word_count} words (minimum 5 required)"
            })
        
        # Basic format validation
        if re.search(r'[A-Z]{15,}', text):
            issues.append({
                "category": "formatting_professionalism",
                "severity": "medium",
                "description": "Contains excessive capitalization (15+ consecutive caps)"
            })
            
        if re.search(r'[!@#$%^&*]{5,}', text):
            issues.append({
                "category": "formatting_professionalism",
                "severity": "medium", 
                "description": "Contains excessive special characters"
            })
        
        # Basic profanity check
        profanity_patterns = [
            r'\bfuck\b', r'\bshit\b', r'\bdamn\b', r'\bbitch\b', 
            r'\basshole\b', r'\bcrap\b', r'\bhell\b'
        ]
        found_profanity = []
        for pattern in profanity_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found_profanity.append(pattern.strip(r'\b'))
        
        if found_profanity:
            issues.append({
                "category": "formatting_professionalism",
                "severity": "high",
                "description": f"Contains inappropriate language: {', '.join(found_profanity)}"
            })
        
        return {
            "word_count": word_count,
            "char_count": char_count,
            "critical_issues": issues,
            "can_proceed": len([i for i in issues if i["severity"] == "high"]) == 0
        }
    
    def validate_chunk(self, chunk: str) -> Dict[str, Any]:
        """Validate a single chunk using OpenAI LLM."""
        # Pre-validation check
        pre_check = self.basic_pre_validation(chunk)
        
        # If critical issues found, return early
        if not pre_check["can_proceed"]:
            return {
                "overall_score": 1,
                "is_valid": False,
                "category_scores": {
                    "length_structure": 1,
                    "grammar_language": 1,
                    "clarity_readability": 1,
                    "content_quality": 1,
                    "formatting_professionalism": 1
                },
                "issues": pre_check["critical_issues"],
                "suggestions": ["Address critical issues before proceeding with evaluation"],
                "word_count": pre_check["word_count"],
                "readability_level": "unassessable",
                "estimated_reading_time": "0 minutes",
                "pre_validation": pre_check,
                "llm_evaluation": False
            }
        
        # LLM-based evaluation
        try:
            print(f"🤖 Evaluating chunk with gpt-4o-mini... ({pre_check['word_count']} words)")
            
            result_str = self.validation_chain.run(chunk=chunk)
            
            # Clean and parse JSON response
            result_str = result_str.strip()
            
            # Remove potential markdown formatting
            if result_str.startswith('```json'):
                result_str = result_str[7:]
            elif result_str.startswith('```'):
                result_str = result_str[3:]
            if result_str.endswith('```'):
                result_str = result_str[:-3]
            
            result_str = result_str.strip()
            
            # Parse JSON
            result_json = json.loads(result_str)
            
            # Add pre-validation issues to LLM issues
            all_issues = pre_check["critical_issues"] + result_json.get("issues", [])
            result_json["issues"] = all_issues
            result_json["pre_validation"] = pre_check
            result_json["llm_evaluation"] = True
            
            return result_json
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {str(e)}")
            return {
                "error": "Invalid JSON response from OpenAI",
                "raw_output": result_str[:500] + "..." if len(result_str) > 500 else result_str,
                "json_error": str(e),
                "pre_validation": pre_check,
                "llm_evaluation": False
            }
        except Exception as e:
            print(f"❌ LLM evaluation error: {str(e)}")
            return {
                "error": f"OpenAI evaluation failed: {str(e)}",
                "pre_validation": pre_check,
                "llm_evaluation": False
            }
    
    def validate_large_text(self, text: str, chunk_size: int = 600, overlap: int = 100) -> Dict[str, Any]:
        """Validate large text by splitting into chunks and using OpenAI for evaluation."""
        
        if not text or not text.strip():
            return {
                "error": "Empty or whitespace-only text provided",
                "chunks": [],
                "overall_summary": {
                    "total_chunks": 0,
                    "valid_chunks": 0,
                    "average_score": 0,
                    "is_valid": False,
                    "total_cost_estimate": "$0.00"
                }
            }
        
        print(f"📄 Processing text: {len(text)} characters, {len(text.split())} words")
        
        # Split text into manageable chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " "]
        )
        chunks = splitter.split_text(text)
        
        print(f"📋 Split into {len(chunks)} chunks for evaluation")
        
        # Validate each chunk
        results = []
        total_score = 0
        valid_chunks = 0
        llm_calls = 0
        
        for i, chunk in enumerate(chunks, start=1):
            print(f"📊 Evaluating chunk {i}/{len(chunks)}...")
            
            chunk_result = self.validate_chunk(chunk)
            chunk_result["chunk_number"] = i
            chunk_result["chunk_preview"] = chunk[:100] + "..." if len(chunk) > 100 else chunk
            
            results.append(chunk_result)
            
            # Calculate stats
            if "overall_score" in chunk_result:
                total_score += chunk_result["overall_score"]
                if chunk_result["is_valid"]:
                    valid_chunks += 1
                if chunk_result.get("llm_evaluation", False):
                    llm_calls += 1
        
        # Calculate overall metrics
        average_score = total_score / len(chunks) if chunks else 0
        estimated_cost = llm_calls * 0.002  # Rough estimate for gpt-4o-mini
        
        overall_summary = {
            "total_chunks": len(chunks),
            "valid_chunks": valid_chunks,
            "invalid_chunks": len(chunks) - valid_chunks,
            "average_score": round(average_score, 2),
            "is_valid": average_score >= 7 and valid_chunks == len(chunks),
            "validation_percentage": round((valid_chunks / len(chunks)) * 100, 2) if chunks else 0,
            "llm_evaluations": llm_calls,
            "estimated_cost": f"${estimated_cost:.4f}"
        }
        
        return {
            "chunks": results,
            "overall_summary": overall_summary,
            "total_word_count": len(text.split()),
            "total_char_count": len(text),
            "model_used": "gpt-4o-mini"
        }

def main():
    """Main function for testing the validator with OpenAI."""
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return
    
    print("🚀 Initializing Content Quality Validator with OpenAI gpt-4o-mini...")
    validator = ContentQualityValidator()
    
    # Test with sample content
    sample_text ="""
Artificial Intelligence represents a transformative paradigm in computational science, fundamentally altering how machines process information and make decisions. Modern AI systems leverage sophisticated algorithms, including neural networks and deep learning architectures, to analyze complex data patterns and generate intelligent responses.
 """
    print("\n🔍 CONTENT QUALITY VALIDATION REPORT")
    print("=" * 60)
    
    # Run validation
    results = validator.validate_large_text(sample_text)
    
    # Display overall summary
    summary = results["overall_summary"]
    print(f"\n📊 OVERALL SUMMARY:")
    print(f"   📈 Total Chunks: {summary['total_chunks']}")
    print(f"   ✅ Valid Chunks: {summary['valid_chunks']}")
    print(f"   ❌ Invalid Chunks: {summary['invalid_chunks']}")
    print(f"   📊 Average Score: {summary['average_score']}/10")
    print(f"   🎯 Success Rate: {summary['validation_percentage']}%")
    print(f"   🤖 LLM Evaluations: {summary['llm_evaluations']}")
    print(f"   💰 Estimated Cost: {summary['estimated_cost']}")
    print(f"   🏆 Overall Status: {'✅ PASSED' if summary['is_valid'] else '❌ FAILED'}")
    
    # Display detailed chunk results
    print(f"\n📋 DETAILED CHUNK ANALYSIS:")
    print("-" * 60)
    
    for chunk_result in results["chunks"]:
        chunk_num = chunk_result["chunk_number"]
        print(f"\n🔸 CHUNK {chunk_num}")
        print(f"Preview: {chunk_result['chunk_preview']}")
        
        if "overall_score" in chunk_result:
            print(f"Overall Score: {chunk_result['overall_score']}/10")
            print(f"Status: {'✅ VALID' if chunk_result['is_valid'] else '❌ INVALID'}")
            print(f"LLM Evaluation: {'✅' if chunk_result.get('llm_evaluation') else '❌'}")
            
            # Category scores
            if "category_scores" in chunk_result:
                print(f"Category Breakdown:")
                for category, score in chunk_result["category_scores"].items():
                    category_name = category.replace('_', ' ').title()
                    print(f"  • {category_name}: {score}/10")
            
            # Issues found
            if chunk_result.get("issues"):
                print(f"Issues Found ({len(chunk_result['issues'])}):")
                for issue in chunk_result["issues"]:
                    severity_icon = "🔴" if issue["severity"] == "high" else "🟡" if issue["severity"] == "medium" else "🟢"
                    print(f"  {severity_icon} [{issue['category']}] {issue['description']}")
            
            # Suggestions
            if chunk_result.get("suggestions"):
                print(f"Suggestions:")
                for suggestion in chunk_result["suggestions"]:
                    print(f"  💡 {suggestion}")
                    
            # Additional info
            if "readability_level" in chunk_result:
                print(f"Readability: {chunk_result['readability_level'].title()}")
            if "estimated_reading_time" in chunk_result:
                print(f"Reading Time: {chunk_result['estimated_reading_time']}")
        else:
            print(f"❌ Error: {chunk_result.get('error', 'Unknown error occurred')}")

if __name__ == "__main__":
    main()
