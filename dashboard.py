import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
from typing import Dict, Any
import os
import time
from Validate_agent import ContentQualityValidator

# Page configuration
st.set_page_config(
    page_title="Content Quality Validation Dashboard",
    page_icon="🔍",
    layout="wide"
)

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
    .success-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .danger-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .chunk-analysis {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .status-passed {
        color: #28a745;
        font-weight: bold;
    }
    .status-failed {
        color: #dc3545;
        font-weight: bold;
    }
    .score-excellent { background: #28a745; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; }
    .score-good { background: #17a2b8; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; }
    .score-average { background: #ffc107; color: black; padding: 0.2rem 0.8rem; border-radius: 15px; }
    .score-poor { background: #fd7e14; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; }
    .score-fail { background: #dc3545; color: white; padding: 0.2rem 0.8rem; border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

def check_api_key():
    """Check if OpenAI API key is available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("🚨 **OpenAI API Key Missing!**")
        st.info("💡 Please set your OPENAI_API_KEY in environment variables or .env file")
        return False
    return True

def initialize_validator():
    """Initialize the content validator."""
    try:
        return ContentQualityValidator()
    except Exception as e:
        st.error(f"❌ **Error initializing validator:** {str(e)}")
        return None

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 CONTENT QUALITY VALIDATION DASHBOARD</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">Real-time AI-Powered Content Analysis</p>
    </div>
    """, unsafe_allow_html=True)

def render_overall_summary(results):
    """Render overall summary section like console output."""
    st.markdown("## 📊 OVERALL SUMMARY")
    st.markdown("=" * 60)
    
    summary = results["overall_summary"]
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Total Chunks</h3>
            <h2>{summary['total_chunks']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card success-card">
            <h3>✅ Valid Chunks</h3>
            <h2>{summary['valid_chunks']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card danger-card">
            <h3>❌ Invalid Chunks</h3>
            <h2>{summary['invalid_chunks']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        score = summary['average_score']
        if score >= 8:
            score_class = "score-excellent"
        elif score >= 6:
            score_class = "score-good"
        elif score >= 4:
            score_class = "score-average"
        elif score >= 2:
            score_class = "score-poor"
        else:
            score_class = "score-fail"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Average Score</h3>
            <h2><span class="{score_class}">{score}/10</span></h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Success Rate", f"{summary['validation_percentage']}%")
    
    with col2:
        st.metric("🤖 LLM Evaluations", summary['llm_evaluations'])
    
    with col3:
        st.metric("💰 Estimated Cost", summary['estimated_cost'])
    
    with col4:
        status = "✅ PASSED" if summary['is_valid'] else "❌ FAILED"
        status_class = "status-passed" if summary['is_valid'] else "status-failed"
        st.markdown(f'<p class="{status_class}">🏆 Overall Status: {status}</p>', unsafe_allow_html=True)

def render_chunk_analysis(results):
    """Render detailed chunk analysis like console output."""
    st.markdown("## 📋 DETAILED CHUNK ANALYSIS")
    st.markdown("-" * 60)
    
    chunks = results.get("chunks", [])
    
    for i, chunk in enumerate(chunks, 1):
        with st.container():
            st.markdown(f"""
            <div class="chunk-analysis">
                <h3>🔸 CHUNK {i}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Content preview
            if "chunk_preview" in chunk:
                st.text(f"Preview: {chunk['chunk_preview']}")
            
            # Check if chunk has validation results
            if "overall_score" in chunk:
                # Main chunk metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    score = chunk['overall_score']
                    st.markdown(f"**Overall Score:** {score}/10")
                    
                    status = "✅ VALID" if chunk['is_valid'] else "❌ INVALID"
                    status_color = "🟢" if chunk['is_valid'] else "🔴"
                    st.markdown(f"**Status:** {status_color} {status}")
                    
                    llm_eval = "✅" if chunk.get('llm_evaluation', False) else "❌"
                    st.markdown(f"**LLM Evaluation:** {llm_eval}")
                
                with col2:
                    if "word_count" in chunk:
                        st.markdown(f"**Word Count:** {chunk['word_count']}")
                    if "readability_level" in chunk:
                        st.markdown(f"**Readability:** {chunk['readability_level'].title()}")
                    if "estimated_reading_time" in chunk:
                        st.markdown(f"**Reading Time:** {chunk['estimated_reading_time']}")
                
                # Category scores breakdown
                if "category_scores" in chunk:
                    st.markdown("**Category Breakdown:**")
                    for category, score in chunk["category_scores"].items():
                        category_name = category.replace('_', ' ').title()
                        st.markdown(f"  • **{category_name}:** {score}/10")
                
                # Issues found
                if chunk.get("issues"):
                    st.markdown(f"**Issues Found ({len(chunk['issues'])}):**")
                    for issue in chunk["issues"]:
                        severity_icon = {
                            "high": "🔴", 
                            "medium": "🟡", 
                            "low": "🟢"
                        }.get(issue["severity"], "ℹ️")
                        
                        category = issue["category"].replace('_', ' ').title()
                        st.markdown(f"  {severity_icon} **[{category}]** {issue['description']}")
                
                # Suggestions
                if chunk.get("suggestions"):
                    st.markdown("**Suggestions:**")
                    for suggestion in chunk["suggestions"]:
                        st.markdown(f"  💡 {suggestion}")
                
            else:
                # Error case
                error_msg = chunk.get('error', 'Unknown error occurred')
                st.error(f"❌ **Error:** {error_msg}")
            
            st.markdown("---")

def render_statistics_charts(results):
    """Render visual statistics."""
    st.markdown("## 📊 VALIDATION STATISTICS")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        chunks = results.get("chunks", [])
        scores = [chunk.get("overall_score", 0) for chunk in chunks if "overall_score" in chunk]
        
        if scores:
            fig_hist = px.histogram(
                x=scores,
                nbins=10,
                title="Quality Score Distribution",
                labels={"x": "Quality Score", "y": "Number of Chunks"},
                color_discrete_sequence=["#667eea"]
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Category performance radar
        categories = ["length_structure", "grammar_language", "clarity_readability", 
                     "content_quality", "formatting_professionalism"]
        
        avg_scores = []
        for category in categories:
            scores = []
            for chunk in chunks:
                if "category_scores" in chunk and category in chunk["category_scores"]:
                    scores.append(chunk["category_scores"][category])
            avg_scores.append(sum(scores) / len(scores) if scores else 0)
        
        category_names = [cat.replace('_', ' ').title() for cat in categories]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_scores,
            theta=category_names,
            fill='toself',
            name='Performance',
            line_color='#764ba2',
            fillcolor='rgba(118, 75, 162, 0.25)'
        ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Category Performance",
            title_x=0.5
        )
        st.plotly_chart(fig_radar, use_container_width=True)

def main():
    # Initialize
    render_header()
    
    # Check API key
    if not check_api_key():
        st.stop()
    
    # Initialize validator
    validator = initialize_validator()
    if not validator:
        st.stop()
    
    # Input section
    st.markdown("## 📝 CONTENT INPUT")
    
    # Sample texts
    sample_options = {
        "Custom Text": "",
        "Good Quality Sample": """
Artificial Intelligence represents a transformative paradigm in computational science, fundamentally altering how machines process information and make decisions. Modern AI systems leverage sophisticated algorithms, including neural networks and deep learning architectures, to analyze complex data patterns and generate intelligent responses.

Machine learning, a critical subset of AI, enables systems to improve performance through experience without explicit programming. The practical applications of AI span numerous industries, from healthcare diagnostic algorithms to financial fraud detection systems.
""",
        "Poor Quality Sample": """
Setting Up the Python Environment

This fucking research paper is complete bullshit and the authors don't know what the hell they're talking about. The damn methodology is flawed and their conclusions are crap. This is a piece of shit study that shouldn't have been published in any respectable journal.
"""
    }
    
    selected_sample = st.selectbox("Choose sample text or enter custom:", list(sample_options.keys()))
    
    if selected_sample == "Custom Text":
        text_content = st.text_area("Enter your content:", height=200)
    else:
        text_content = sample_options[selected_sample]
        st.text_area("Selected sample text:", value=text_content, height=200, disabled=True)
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size", 300, 1200, 600)
    with col2:
        overlap = st.slider("Chunk Overlap", 50, 300, 100)
    
    # Validate button
    if st.button("🚀 Validate Content", type="primary", disabled=not text_content.strip()):
        if text_content.strip():
            with st.spinner("🤖 Analyzing content..."):
                # Run validation
                results = validator.validate_large_text(
                    text_content, 
                    chunk_size=chunk_size, 
                    overlap=overlap
                )
                
                # Store in session state
                st.session_state['validation_results'] = results
                
                st.success("✅ Validation completed!")
                st.rerun()
    
    # Display results if available
    if 'validation_results' in st.session_state and st.session_state['validation_results']:
        results = st.session_state['validation_results']
        
        # Render all sections
        render_overall_summary(results)
        render_statistics_charts(results)
        render_chunk_analysis(results)
        
        # Export section
        st.markdown("## 📤 EXPORT OPTIONS")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                "📊 Download Summary JSON",
                data=json.dumps(results["overall_summary"], indent=2),
                file_name=f"validation_summary_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            st.download_button(
                "📋 Download Full Report",
                data=json.dumps(results, indent=2),
                file_name=f"validation_report_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # Create CSV data
            csv_data = []
            for chunk in results.get("chunks", []):
                if "overall_score" in chunk:
                    row = {
                        "chunk_number": chunk.get("chunk_number", 0),
                        "overall_score": chunk.get("overall_score", 0),
                        "is_valid": chunk.get("is_valid", False),
                        "word_count": chunk.get("word_count", 0),
                        "issues_count": len(chunk.get("issues", [])),
                    }
                    csv_data.append(row)
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                st.download_button(
                    "📈 Download CSV Data",
                    data=df.to_csv(index=False),
                    file_name=f"validation_data_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
