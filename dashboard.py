import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import asyncio
from typing import Dict, Any
import os
import time

# Import your enhanced validator
from Validate_agent import ContentQualityAgent  # ⭐ KEY CHANGE: Use enhanced agent

# Page configuration
st.set_page_config(
    page_title="Enhanced Content Quality Validation Dashboard",
    page_icon="🔍",
    layout="wide"
)


class EnhancedDashboard:
    def __init__(self):
        self.validator = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state."""
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = None
        if 'input_type' not in st.session_state:
            st.session_state.input_type = "plain_text"
    
    def check_api_key(self):
        """Check if required API keys are available."""
        openai_key = os.getenv("OPENAI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        
        if not openai_key:
            st.error("🚨 **OpenAI API Key Missing!**")
            st.info("💡 Please set OPENAI_API_KEY in environment variables")
            return False
        
        if not tavily_key:
            st.warning("⚠️ **Tavily API Key Missing!** Fact-checking will be disabled.")
            st.info("💡 Set TAVILY_API_KEY for full functionality")
        
        return True
    
    def initialize_validator(self):
        """Initialize the enhanced content validator."""
        try:
            # ⭐ Use your enhanced ContentQualityAgent
            self.validator = ContentQualityAgent(model="gpt-4o-mini", temperature=0)
            return True
        except Exception as e:
            st.error(f"❌ **Error initializing validator:** {str(e)}")
            return None

    def render_input_section(self):
        """Enhanced input section supporting both text and iText payloads."""
        st.markdown("## 📝 CONTENT INPUT")
        
        # Input type selector
        input_type = st.radio(
            "**Choose Input Type:**",
            ["📄 Plain Text", "🔧 iText Payload", "📋 Sample Content"],
            horizontal=True,
            key="input_type_radio"
        )
        
        content_to_validate = None
        
        # ⭐ PLAIN TEXT INPUT
        if input_type == "📄 Plain Text":
            st.session_state.input_type = "plain_text"
            content_to_validate = st.text_area(
                "Enter your text content:",
                height=200,
                placeholder="Paste your content here for comprehensive quality analysis...",
                key="plain_text_input"
            )
        
        # ⭐ iTEXT PAYLOAD INPUT  
        elif input_type == "🔧 iText Payload":
            st.session_state.input_type = "itext_payload"
            
            st.markdown("### 🔧 iText Payload Structure")
            st.info("Enter a JSON payload with message, content, simplified, and elaborated fields")
            
            payload_text = st.text_area(
                "Enter iText JSON payload:",
                height=300,
                placeholder="""{
    "message": "Successfully generated iText block data.",
    "data": [
        {
            "type": "iText",
            "data": {
                "content": "Your original content here...",
                "simplified": "Simplified version here...",
                "elaborated": "Elaborated version here..."
            }
        }
    ]
}""",
                key="itext_payload_input"
            )
            
            # Validate JSON format
            if payload_text.strip():
                try:
                    content_to_validate = json.loads(payload_text)
                    st.success("✅ Valid JSON format")
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON: {str(e)}")
                    content_to_validate = None
        
        # ⭐ SAMPLE CONTENT
        elif input_type == "📋 Sample Content":
            sample_type = st.selectbox(
                "Choose sample:",
                ["Plain Text - Good Quality", "Plain Text - Poor Quality", "iText Payload Sample"]
            )
            
            if sample_type == "Plain Text - Good Quality":
                st.session_state.input_type = "plain_text"
                content_to_validate = """
Artificial Intelligence represents a transformative paradigm in computational science, fundamentally altering how machines process information and make decisions. Modern AI systems leverage sophisticated algorithms, including neural networks and deep learning architectures, to analyze complex data patterns and generate intelligent responses.

Machine learning, a critical subset of AI, enables systems to improve performance through experience without explicit programming. The practical applications of AI span numerous industries, from healthcare diagnostic algorithms to financial fraud detection systems.
"""
            elif sample_type == "Plain Text - Poor Quality":
                st.session_state.input_type = "plain_text"  
                content_to_validate = """
This fucking research paper is complete bullshit and the authors don't know what the hell they're talking about. The damn methodology is flawed and their conclusions are crap. This is a piece of shit study that shouldn't have been published in any respectable journal. This fucking research paper is complete bullshit and the authors don't know what the hell they're talking about.
"""
            else:  # iText Payload Sample
                st.session_state.input_type = "itext_payload"
                content_to_validate = {
                    "message": "Successfully generated iText block data.",
                    "data": [
                        {
                            "type": "iText",
                            "data": {
                                "content": "Deep learning foundations serve as the bedrock for understanding neural networks, which are computational models inspired by the human brain.",
                                "simplified": "Deep learning is like building blocks for understanding how computers can learn like humans do.",
                                "elaborated": "The comprehensive foundations of deep learning encompass a sophisticated range of principles and methodologies that are absolutely essential for developing a thorough understanding of how neural networks function in computational environments."
                            }
                        }
                    ]
                }
            
            # Display selected sample
            if isinstance(content_to_validate, dict):
                st.json(content_to_validate)
            else:
                st.text_area("Selected sample:", value=content_to_validate, height=150, disabled=True)
        
        return content_to_validate

    async def run_validation_async(self, content):
        """Run validation asynchronously."""
        try:
            # ⭐ Use the enhanced validator's validate method
            result = await self.validator.validate_async(content)
            return result
        except Exception as e:
            return {"error": f"Validation failed: {str(e)}"}


    async def render_validation_controls(self, content_to_validate):
        """Render validation controls and execute validation."""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            validate_btn = st.button(
                "🚀 Analyze Content Quality", 
                type="primary",
                disabled=not content_to_validate,
                use_container_width=True
            )
        
        with col2:
            if content_to_validate:
                if isinstance(content_to_validate, dict):
                    # Count words in iText payload
                    total_words = 0
                    try:
                        data = content_to_validate.get("data", [{}])[0].get("data", {})
                        for field in ["content", "simplified", "elaborated"]:
                            total_words += len(data.get(field, "").split())
                    except:
                        total_words = 0
                    st.metric("Total Words", total_words)
                else:
                    word_count = len(content_to_validate.split())
                    st.metric("Word Count", word_count)
        
        with col3:
            input_type = st.session_state.get('input_type', 'plain_text')
            st.metric("Input Type", "iText" if input_type == "itext_payload" else "Text")
        
        # ⭐ VALIDATION EXECUTION
        if validate_btn and content_to_validate:
            with st.spinner("🤖 Running comprehensive analysis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Progress updates
                    progress_bar.progress(25)
                    status_text.text("🔍 Initializing validation tools...")
                    time.sleep(0.5)
                    
                    progress_bar.progress(50)
                    status_text.text("🤖 Running AI analysis...")
                    
                    # ⭐ KEY: Run the enhanced validation
                    if asyncio.get_event_loop().is_running():
                        # If already in async context, create task
                        loop = asyncio.get_event_loop()
                        result = await self.run_validation_async(content_to_validate)
                    else:
                        # Create new event loop
                        result = asyncio.run(self.run_validation_async(content_to_validate))
                    
                    progress_bar.progress(75)
                    status_text.text("📊 Processing results...")
                    time.sleep(0.5)
                    
                    # Store results
                    st.session_state.validation_results = result
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.success("🎉 **Content analysis completed successfully!**")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ **Validation failed:** {str(e)}")
                    progress_bar.empty()
                    status_text.empty()

    def render_results_dashboard(self):
        """Render enhanced results dashboard."""
        if not st.session_state.validation_results:
            st.info("👆 Enter content above and click 'Analyze Content Quality' to see results")
            return
        
        results = st.session_state.validation_results
        
        # Handle different result formats
        if "error" in results:
            st.error(f"❌ **Error:** {results['error']}")
            return
        
        # ⭐ PARSE ENHANCED AGENT OUTPUT
        agent_output = results.get('output', '')
        
        st.markdown("## 📊 COMPREHENSIVE ANALYSIS RESULTS")
        st.markdown("=" * 60)
        
        # Display raw agent output in expandable section
        with st.expander("🤖 **Detailed Agent Analysis**", expanded=True):
            st.markdown(agent_output)
        
        # Try to extract structured data from agent output
        self.extract_and_display_metrics(agent_output)
        
        # ⭐ EXPORT OPTIONS
        self.render_export_options(results)

    def extract_and_display_metrics(self, agent_output):
        """Extract and display metrics from agent output."""
        
        # Look for JSON patterns in the output
        json_patterns = re.findall(r'\{[^{}]*\}', agent_output)
        
        metrics_found = {}
        for pattern in json_patterns:
            try:
                data = json.loads(pattern)
                if "category" in data and "score" in data:
                    metrics_found[data["category"]] = data
            except:
                continue
 
        if metrics_found:
            st.markdown("## 📊 CATEGORY BREAKDOWN")
            
            # Display category scores
            cols = st.columns(len(metrics_found))
            for i, (category, data) in enumerate(metrics_found.items()):
                with cols[i]:
                    score = data.get("score", 0)
                    category_name = category.replace('_', ' ').title()
                    
                    # Color based on score
                    if score >= 8:
                        color = "🟢"
                    elif score >= 6:
                        color = "🟡"
                    else:
                        color = "🔴"
                    
                    st.metric(
                        f"{color} {category_name}",
                        f"{score}/10"
                    )
            
            # Issues summary
            all_issues = []
            for data in metrics_found.values():
                all_issues.extend(data.get("issues", []))
            
            if all_issues:
                st.markdown("## ⚠️ ISSUES IDENTIFIED")
                for i, issue in enumerate(all_issues, 1):
                    st.markdown(f"{i}. {issue}")

    def render_export_options(self, results):
        """Render export options."""
        st.markdown("## 📤 EXPORT OPTIONS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📊 Download Analysis Report",
                data=json.dumps(results, indent=2),
                file_name=f"content_analysis_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # Create summary
            summary = {
                "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                "input_type": st.session_state.get('input_type', 'unknown'),
                "agent_output": results.get('output', ''),
                "analysis_summary": "Enhanced multi-tool content quality analysis"
            }
            
            st.download_button(
                "📋 Download Summary",
                data=json.dumps(summary, indent=2),
                file_name=f"analysis_summary_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col3:
            # Text export
            text_report = f"""
CONTENT QUALITY ANALYSIS REPORT
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
Input Type: {st.session_state.get('input_type', 'unknown')}

ANALYSIS RESULTS:
{results.get('output', 'No detailed output available')}
"""
            
            st.download_button(
                "📄 Download Text Report",
                data=text_report,
                file_name=f"analysis_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ⭐ MAIN FUNCTION WITH ENHANCED INTEGRATION
def main():
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 ENHANCED CONTENT QUALITY DASHBOARD</h1>
        <p style="font-size: 1.2rem;">Multi-Tool AI Analysis with iText Support</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize dashboard
    dashboard = EnhancedDashboard()
    
    # Check API keys
    if not dashboard.check_api_key():
        st.stop()
    
    # Initialize validator
    if not dashboard.initialize_validator():
        st.stop()
    
    # Render interface
    content_to_validate = dashboard.render_input_section()
    dashboard.render_validation_controls(content_to_validate)
    dashboard.render_results_dashboard()

if __name__ == "__main__":
    main()
