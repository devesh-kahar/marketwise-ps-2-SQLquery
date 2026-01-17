"""
🧠 QueryMind - Natural Language to SQL
Professional UI with Sidebar + Main Content
"""

import streamlit as st
import pandas as pd
import sqlite3
import time
from datetime import datetime
from nl_to_sql import MasterOrchestrator, LLMProvider
import mysql.connector

# Import API keys from config
try:
    from config import (GEMINI_API_KEY, GROQ_API_KEY, GROQ_API_KEYS, DEEPSEEK_API_KEY,
                       HUGGINGFACE_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY,
                       MODEL_PRIORITY)
except ImportError:
    st.error("❌ config.py not found! Please create it with your API keys.")
    st.stop()
except:
    # Fallback if GROQ_API_KEYS doesn't exist
    from config import (GEMINI_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY,
                       HUGGINGFACE_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY,
                       MODEL_PRIORITY)
    GROQ_API_KEYS = [GROQ_API_KEY] if GROQ_API_KEY else []
    st.stop()

# Check if at least one key is configured
has_key = False
if OPENROUTER_API_KEY and OPENROUTER_API_KEY != "YOUR_OPENROUTER_KEY_HERE" and OPENROUTER_API_KEY != "":
    has_key = True
    st.sidebar.success("✅ OpenRouter (FREE)")
if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_DEEPSEEK_KEY_HERE" and DEEPSEEK_API_KEY != "":
    has_key = True
if HUGGINGFACE_API_KEY and HUGGINGFACE_API_KEY != "YOUR_HF_KEY_HERE" and HUGGINGFACE_API_KEY != "":
    has_key = True
    st.sidebar.success("✅ HuggingFace (FREE)")
if TOGETHER_API_KEY and TOGETHER_API_KEY != "YOUR_TOGETHER_KEY_HERE" and TOGETHER_API_KEY != "":
    has_key = True
    st.sidebar.success("✅ Together AI (FREE)")
if GROQ_API_KEY and GROQ_API_KEY != "YOUR_NEW_GROQ_KEY_HERE" and GROQ_API_KEY != "":
    has_key = True
if GEMINI_API_KEY and GEMINI_API_KEY != "":
    has_key = True

if not has_key:
    st.error("⚠️ No API keys configured!")
    st.markdown("""
    <div style="background: rgba(232, 93, 117, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #E85D75; margin: 1rem 0;">
        <p style="color: #E8E6E3; margin: 0; line-height: 1.6;">
        <strong>Get FREE API Key:</strong><br><br>
        
        <strong>OpenRouter (BEST - Access to ALL models)</strong><br>
        → <a href="https://openrouter.ai/keys" target="_blank" style="color: #6B7FFF;">openrouter.ai/keys</a><br>
        → FREE $1 credit + many free models!<br><br>
        
        Then update <code style="background: #0A0B0F; padding: 0.2rem 0.4rem; border-radius: 4px;">config.py</code> and refresh!
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Page config
st.set_page_config(
    page_title="QueryMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS - Eye-Soothing Dark Mode
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    
    * { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        letter-spacing: 0.01em;
    }
    
    /* Main app background - warm dark */
    .stApp {
        background: linear-gradient(135deg, #0F1117 0%, #1A1B26 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        max-width: 900px;
        padding-bottom: 3rem;
    }
    
    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #1E1F2B 0%, #252736 100%);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        border: 1px solid rgba(107, 127, 255, 0.1);
    }
    
    .hero h1 {
        background: linear-gradient(135deg, #6B7FFF 0%, #8B7FFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        margin: 0;
        font-weight: 600;
    }
    
    .hero p {
        color: #9B9C9E;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    
    /* Cards - soft dark panels */
    .card {
        background: #1A1B26;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        margin-bottom: 1.5rem;
        border: 1px solid #2A2D3A;
    }
    
    .card-header {
        font-size: 1.1rem;
        font-weight: 500;
        color: #E8E6E3;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Answer box - gradient accent */
    .answer {
        background: linear-gradient(135deg, rgba(107, 127, 255, 0.15) 0%, rgba(139, 127, 255, 0.15) 100%);
        color: #E8E6E3;
        padding: 1.5rem;
        border-radius: 12px;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 1.5rem 0;
        border-left: 3px solid #6B7FFF;
        box-shadow: 0 4px 16px rgba(107, 127, 255, 0.1);
    }
    
    /* SQL code block */
    .sql-box {
        background: #0A0B0F;
        color: #E8E6E3;
        padding: 1.2rem;
        border-radius: 10px;
        font-family: 'JetBrains Mono', 'Fira Code', 'Courier New', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
        border: 1px solid #2A2D3A;
        line-height: 1.6;
    }
    
    /* Metrics row */
    .metric {
        background: #1E1F2B;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #2A2D3A;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        background: linear-gradient(135deg, #6B7FFF 0%, #8B7FFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #9B9C9E;
        text-transform: uppercase;
        margin-top: 0.3rem;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #1A1B26;
        border-right: 1px solid #2A2D3A;
    }
    
    section[data-testid="stSidebar"] h1 {
        color: #E8E6E3;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Table items in sidebar */
    .table-item {
        background: #1E1F2B;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #2A2D3A;
        transition: all 0.2s ease;
    }
    
    .table-item:hover {
        background: #252736;
        border-color: #6B7FFF;
    }
    
    .table-name {
        font-weight: 500;
        color: #E8E6E3;
        font-size: 0.9rem;
    }
    
    .table-count {
        background: rgba(107, 127, 255, 0.2);
        color: #6B7FFF;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Recent search items */
    .search-item {
        background: #1E1F2B;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid #4CAF7D;
        color: #9B9C9E;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    
    .search-item:hover {
        background: #252736;
    }
    
    .search-item.failed {
        border-left-color: #E85D75;
    }
    
    /* Buttons - soft accent */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        background: #6B7FFF;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: #8B7FFF;
        box-shadow: 0 4px 12px rgba(107, 127, 255, 0.3);
        transform: translateY(-1px);
    }
    
    /* Text inputs */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #1E1F2B;
        color: #E8E6E3;
        border: 1px solid #2A2D3A;
        border-radius: 8px;
        padding: 0.6rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #6B7FFF;
        box-shadow: 0 0 0 2px rgba(107, 127, 255, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #1E1F2B;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #9B9C9E;
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: #6B7FFF;
        color: white;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: #1A1B26;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(76, 175, 125, 0.1);
        color: #4CAF7D;
        border-left: 3px solid #4CAF7D;
    }
    
    .stError {
        background: rgba(232, 93, 117, 0.1);
        color: #E85D75;
        border-left: 3px solid #E85D75;
    }
    
    .stWarning {
        background: rgba(245, 158, 66, 0.1);
        color: #F59E42;
        border-left: 3px solid #F59E42;
    }
    
    .stInfo {
        background: rgba(107, 127, 255, 0.1);
        color: #6B7FFF;
        border-left: 3px solid #6B7FFF;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #6B7FFF;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: #1E1F2B;
        color: #E8E6E3;
        border: 1px solid #2A2D3A;
    }
    
    .stDownloadButton > button:hover {
        background: #252736;
        border-color: #6B7FFF;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #6B7FFF;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1A1B26;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #2A2D3A;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6B7FFF;
    }
    
    /* Labels */
    label {
        color: #9B9C9E !important;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Captions */
    .caption {
        color: #6B7280;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE
# ============================================================================

if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'connection_type' not in st.session_state:
    st.session_state.connection_type = None
if 'system' not in st.session_state:
    st.session_state.system = None
if 'csv_handler' not in st.session_state:
    st.session_state.csv_handler = None
if 'db_name' not in st.session_state:
    st.session_state.db_name = None
if 'tables' not in st.session_state:
    st.session_state.tables = []
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None


# ============================================================================
# CSV HANDLER
# ============================================================================

class CSVQueryHandler:
    def __init__(self, dataframes: dict, llm: LLMProvider):
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        self.llm = llm
        self.tables = {}
        
        for name, df in dataframes.items():
            clean_name = name.replace(' ', '_').replace('.csv', '').lower()
            df.to_sql(clean_name, self.conn, index=False, if_exists='replace')
            self.tables[clean_name] = {'columns': list(df.columns), 'rows': len(df)}
    
    def query(self, question: str) -> dict:
        schema = "DATABASE: CSV Files\n\n"
        for table, info in self.tables.items():
            schema += f"TABLE: {table} ({info['rows']} rows)\n"
            schema += "COLUMNS: " + ", ".join(info['columns']) + "\n\n"
        
        prompt = f'Generate SQLite query for: "{question}"\n\n{schema}\n\nReturn ONLY the SQL query.'
        
        try:
            response = self.llm.generate(prompt)
            sql = response.strip().replace('```sql', '').replace('```', '').strip()
            df = pd.read_sql_query(sql, self.conn)
            
            return {
                'success': True,
                'sql': sql,
                'data': df,
                'rows': len(df),
                'answer': f"Found {len(df)} results."
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'sql': sql if 'sql' in locals() else ''}
    
    def close(self):
        self.conn.close()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("# 🧠 QueryMind")
    st.markdown('<p style="color: #9B9C9E; font-size: 0.85rem; margin-top: -0.5rem;">Multi-Agent NL-to-SQL (OPTIMIZED)</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: #4CAF50; font-size: 0.75rem; margin-top: -0.5rem;">⚡ 3-5 API calls/query (40% faster!)</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    if not st.session_state.connected:
        st.markdown('<p style="color: #E8E6E3; font-weight: 500; margin-bottom: 1rem;">🔌 Connect to Database</p>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["MySQL", "CSV Upload"])
        
        with tab1:
            with st.form("mysql"):
                host = st.text_input("Host", "localhost")
                user = st.text_input("User", "root")
                password = st.text_input("Password", type="password")
                database = st.text_input("Database")
                
                if st.form_submit_button("Connect", use_container_width=True):
                    if database:
                        try:
                            system = MasterOrchestrator(
                                {'host': host, 'user': user, 'password': password, 'database': database},
                                gemini_key=GEMINI_API_KEY,
                                groq_key=GROQ_API_KEY,
                                groq_keys=GROQ_API_KEYS,  # Pass all Groq keys
                                deepseek_key=DEEPSEEK_API_KEY,
                                huggingface_key=HUGGINGFACE_API_KEY,
                                together_key=TOGETHER_API_KEY,
                                openrouter_key=OPENROUTER_API_KEY,
                                model_priority=MODEL_PRIORITY
                            )
                            st.session_state.system = system
                            st.session_state.connected = True
                            st.session_state.connection_type = 'mysql'
                            st.session_state.db_name = database
                            st.session_state.tables = system.schema_analyzer.get_all_tables()
                            st.success("✅ Connected!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {str(e)[:60]}")
        
        with tab2:
            files = st.file_uploader("Upload CSV", type=['csv'], accept_multiple_files=True)
            
            if files:
                dfs = {}
                for f in files:
                    try:
                        dfs[f.name] = pd.read_csv(f)
                        st.success(f"✅ {f.name}")
                    except Exception as e:
                        st.error(f"❌ {f.name}")
                
                if dfs and st.button("Load", use_container_width=True):
                    try:
                        llm = LLMProvider(
                            gemini_key=GEMINI_API_KEY,
                            groq_key=GROQ_API_KEY,
                            groq_keys=GROQ_API_KEYS,  # Pass all Groq keys
                            deepseek_key=DEEPSEEK_API_KEY,
                            huggingface_key=HUGGINGFACE_API_KEY,
                            together_key=TOGETHER_API_KEY,
                            openrouter_key=OPENROUTER_API_KEY,
                            model_priority=MODEL_PRIORITY
                        )
                        handler = CSVQueryHandler(dfs, llm)
                        st.session_state.csv_handler = handler
                        st.session_state.connected = True
                        st.session_state.connection_type = 'csv'
                        st.session_state.db_name = "CSV Files"
                        st.session_state.tables = list(handler.tables.keys())
                        st.success("✅ Loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ {e}")
    
    else:
        st.markdown(f'<div style="background: rgba(76, 175, 125, 0.15); padding: 0.8rem; border-radius: 8px; border-left: 3px solid #4CAF7D; margin-bottom: 1rem;"><div style="color: #4CAF7D; font-weight: 500;">✅ Connected</div><div style="color: #9B9C9E; font-size: 0.85rem; margin-top: 0.3rem;">{st.session_state.db_name}</div><div style="color: #6B7280; font-size: 0.75rem; margin-top: 0.2rem;">{st.session_state.connection_type.upper()}</div></div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown(f'<p style="color: #E8E6E3; font-weight: 500; margin-bottom: 1rem;">📋 Tables ({len(st.session_state.tables)})</p>', unsafe_allow_html=True)
        
        for table in st.session_state.tables:
            if st.session_state.connection_type == 'mysql':
                count = st.session_state.system.schema_analyzer.get_table_row_count(table)
            else:
                count = st.session_state.csv_handler.tables.get(table, {}).get('rows', 0)
            
            st.markdown(f"""
            <div class="table-item">
                <span class="table-name">{table}</span>
                <span class="table-count">{count:,}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🔌 Disconnect", use_container_width=True):
            if st.session_state.connection_type == 'mysql' and st.session_state.system:
                st.session_state.system.close()
            elif st.session_state.csv_handler:
                st.session_state.csv_handler.close()
            st.session_state.connected = False
            st.session_state.system = None
            st.session_state.connection_type = None
            st.rerun()
    
    st.markdown("---")
    
    # Recent searches
    if st.session_state.history:
        st.markdown('<p style="color: #E8E6E3; font-weight: 500; margin-bottom: 1rem;">📜 Recent Queries</p>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            icon = "✅" if item['success'] else "❌"
            failed_class = ' failed' if not item['success'] else ''
            st.markdown(f"""
            <div class="search-item{failed_class}">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>{icon}</span>
                    <span style="flex: 1; font-size: 0.85rem;">{item['question'][:35]}...</span>
                </div>
                <div style="font-size: 0.7rem; color: #6B7280; margin-top: 0.3rem;">{item['time']}</div>
            </div>
            """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Hero
st.markdown("""
<div class="hero">
    <h1>🧠 QueryMind</h1>
    <p>Transform Natural Language into SQL with Multi-Agent AI</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.connected:
    st.markdown("""
    <div style="text-align: center; padding: 3rem 1rem; color: #9B9C9E;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">👈</div>
        <h3 style="color: #E8E6E3; font-weight: 500;">Connect to Get Started</h3>
        <p style="line-height: 1.6;">Connect to a MySQL database or upload CSV files from the sidebar to begin querying with natural language.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">✨ Natural Language</div>
            <p style="color: #9B9C9E; line-height: 1.6;">Ask questions in plain English, no SQL knowledge required</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">🤖 10 AI Agents</div>
            <p style="color: #9B9C9E; line-height: 1.6;">Multi-agent system with smart optimization (3-5 API calls, 40% faster)</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-header">🛡️ Safe & Smart</div>
            <p style="color: #9B9C9E; line-height: 1.6;">Read-only queries with automatic error recovery</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Query Input
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">💬 Ask Your Question</div>', unsafe_allow_html=True)
    
    query = st.text_area(
        "query",
        placeholder="e.g., How many customers are from Germany? or Show me the top 10 products by revenue",
        height=100,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        submit = st.button("🚀 Ask", use_container_width=True, type="primary")
    with col2:
        if st.button("📋 Schema", use_container_width=True):
            query = "What tables exist?"
            submit = True
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.current_result = None
            st.rerun()
    
    # Examples
    st.markdown('<p style="color: #9B9C9E; font-size: 0.9rem; margin-top: 1rem; margin-bottom: 0.5rem;">💡 Quick Examples:</p>', unsafe_allow_html=True)
    examples = ["Show first 10 rows", "Count all records", "Which table has most rows?"]
    cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        if cols[i].button(ex, key=f"ex_{i}"):
            query = ex
            submit = True
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query
    if submit and query:
        with st.spinner("🧠 AI Agents Working..."):
            try:
                start = time.time()
                
                if st.session_state.connection_type == 'mysql':
                    result = st.session_state.system.process_question(query, show_reasoning=False)
                    
                    # Store reasoning trace
                    reasoning_trace = result.get('reasoning_trace', {})
                    
                    display = {
                        'success': result['success'],
                        'sql': result.get('sql', ''),
                        'data': pd.DataFrame(result['result'].data) if result['success'] and result['result'].data else pd.DataFrame(),
                        'answer': result['result'].natural_language_answer if result['success'] else result['result'].error,
                        'rows': result['result'].rows_affected if result['success'] else 0,
                        'time': result['result'].execution_time if result['success'] else 0,
                        'reasoning_trace': reasoning_trace
                    }
                else:
                    result = st.session_state.csv_handler.query(query)
                    display = result
                    display['time'] = time.time() - start
                    display['reasoning_trace'] = None
                
                st.session_state.current_result = display
                st.session_state.history.append({
                    'question': query,
                    'success': display['success'],
                    'time': datetime.now().strftime('%H:%M:%S')
                })
                st.rerun()
            
            except Exception as e:
                error_msg = str(e)
                if "All LLM providers failed" in error_msg or "rate limit" in error_msg.lower() or "payment" in error_msg.lower():
                    st.error("❌ **API Error**")
                    
                    if "payment" in error_msg.lower() or "402" in error_msg:
                        st.markdown("""
                        <div style="background: rgba(232, 93, 117, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #E85D75; margin: 1rem 0;">
                            <p style="color: #E8E6E3; margin: 0; line-height: 1.6;">
                            <strong>DeepSeek requires payment/credits.</strong><br><br>
                            
                            <strong>Quick Fix - Use FREE alternatives:</strong><br><br>
                            
                            <strong>Option 1: HuggingFace (100% FREE)</strong><br>
                            1. Get token: <a href="https://huggingface.co/settings/tokens" target="_blank" style="color: #6B7FFF;">huggingface.co/settings/tokens</a><br>
                            2. Open <code style="background: #0A0B0F; padding: 0.2rem 0.4rem; border-radius: 4px;">config.py</code><br>
                            3. Set: <code style="background: #0A0B0F; padding: 0.2rem 0.4rem; border-radius: 4px;">HUGGINGFACE_API_KEY = "hf_..."</code><br>
                            4. Set: <code style="background: #0A0B0F; padding: 0.2rem 0.4rem; border-radius: 4px;">MODEL_PRIORITY = ["huggingface"]</code><br>
                            5. Refresh page<br><br>
                            
                            <strong>Option 2: Together AI (FREE $25 credit)</strong><br>
                            1. Get key: <a href="https://api.together.xyz/settings/api-keys" target="_blank" style="color: #6B7FFF;">api.together.xyz</a><br>
                            2. Update config.py same way<br><br>
                            
                            <strong>Option 3: Use Groq (if you have new key)</strong><br>
                            Get new key from different email: <a href="https://console.groq.com" target="_blank" style="color: #6B7FFF;">console.groq.com</a>
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: rgba(232, 93, 117, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid #E85D75; margin: 1rem 0;">
                            <p style="color: #E8E6E3; margin: 0; line-height: 1.6;">
                            API rate limits reached.<br><br>
                            <strong>To fix:</strong><br>
                            1. Get a new Groq API key from <a href="https://console.groq.com" target="_blank" style="color: #6B7FFF;">console.groq.com</a> (use new email)<br>
                            2. Open <code style="background: #0A0B0F; padding: 0.2rem 0.4rem; border-radius: 4px;">config.py</code> in your editor<br>
                            3. Replace <code style="background: #0A0B0F; padding: 0.2rem 0.4rem; border-radius: 4px;">GROQ_API_KEY</code> with your new key<br>
                            4. Refresh this page
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Error: {error_msg}")
                
                st.session_state.current_result = {
                    'success': False,
                    'error': error_msg,
                    'reasoning_trace': None
                }
                st.session_state.history.append({
                    'question': query,
                    'success': False,
                    'time': datetime.now().strftime('%H:%M:%S')
                })
    
    # Display results
    if st.session_state.current_result:
        result = st.session_state.current_result
        
        if result['success']:
            # Answer
            st.markdown(f'<div class="answer">💬 <strong>Answer:</strong><br>{result["answer"]}</div>', unsafe_allow_html=True)
            
            # Reasoning Trace (if available)
            if result.get('reasoning_trace') and result['reasoning_trace'].get('steps'):
                with st.expander("🧠 **Reasoning Trace** - See How 10 AI Agents Solved This", expanded=False):
                    trace = result['reasoning_trace']
                    st.markdown(f"**Total Time:** {trace['total_time']:.2f}s | **Steps:** {len(trace['steps'])}")
                    st.markdown("---")
                    
                    for step in trace['steps']:
                        status_color = "#4CAF7D" if step['status'] == 'completed' else "#F59E42" if step['status'] == 'in_progress' else "#E85D75"
                        status_icon = "✅" if step['status'] == 'completed' else "⚠️" if step['status'] == 'in_progress' else "❌"
                        
                        st.markdown(f"""
                        <div style="background: #1E1F2B; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; border-left: 3px solid {status_color};">
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                                <span style="font-size: 1.2rem;">{step['icon']}</span>
                                <strong style="color: #E8E6E3;">Step {step['number']}: {step['name']}</strong>
                                <span style="margin-left: auto; color: {status_color};">{status_icon} {step['status']}</span>
                                <span style="color: #9B9C9E; font-size: 0.85rem;">{step['time']:.2f}s</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show details
                        if step.get('details'):
                            with st.expander(f"Details for Step {step['number']}", expanded=False):
                                st.json(step['details'])
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class="metric">
                    <div class="metric-value">{result.get('rows', 0):,}</div>
                    <div class="metric-label">Rows Returned</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric">
                    <div class="metric-value">{result.get('time', 0):.2f}s</div>
                    <div class="metric-label">Execution Time</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                cols = len(result['data'].columns) if isinstance(result.get('data'), pd.DataFrame) else 0
                st.markdown(f"""
                <div class="metric">
                    <div class="metric-value">{cols}</div>
                    <div class="metric-label">Columns</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Tabs
            tab1, tab2 = st.tabs(["📊 Results", "💻 SQL Query"])
            
            with tab1:
                if isinstance(result.get('data'), pd.DataFrame) and not result['data'].empty:
                    df = result['data']
                    
                    # Pagination
                    if len(df) > 10:
                        rows_per_page = st.slider("Rows per page", 10, 100, 25, 5, key="pagination")
                        total_pages = (len(df) - 1) // rows_per_page + 1
                        
                        if total_pages > 1:
                            page = st.number_input("Page", 1, total_pages, 1, key="page_num")
                            start = (page - 1) * rows_per_page
                            end = start + rows_per_page
                            st.dataframe(df.iloc[start:end], use_container_width=True, height=400)
                            st.caption(f"Showing rows {start + 1}-{min(end, len(df))} of {len(df)}")
                        else:
                            st.dataframe(df, use_container_width=True, height=400)
                    else:
                        st.dataframe(df, use_container_width=True, height=400)
                    
                    # Download
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        csv,
                        f"querymind_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("✨ Query executed successfully but returned no data")
            
            with tab2:
                st.code(result.get('sql', ''), language='sql')
        
        else:
            st.error(f"❌ {result.get('error', 'Query failed')}")
            if result.get('sql'):
                st.code(result['sql'], language='sql')
