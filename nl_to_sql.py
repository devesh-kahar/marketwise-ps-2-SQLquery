"""
Natural Language to SQL System - OPTIMIZED MULTI-AGENT ARCHITECTURE
====================================================================
10 Specialized Agents with SMART OPTIMIZATION:
1. Question Analyzer - Deep understanding of user intent
2. Schema Intelligence - Relevant tables/columns identification
3. Exploration Agent - Probe queries (conditional - only if highly ambiguous)
4. Clarification Generator - Handle ambiguous queries (conditional)
5. Query Planner - Step-by-step SQL strategy
6. SQL Generator - Safe, efficient SQL generation
7. SQL Critic - Self-review (conditional - only for complex queries)
8. Error Analyzer - Diagnose and fix failures (conditional - only on errors)
9. Result Interpreter - Natural language answers
10. Master Orchestrator - Coordinate all agents

OPTIMIZATIONS:
- Merged agents 1+2 into 1 API call (combined analysis)
- Merged agents 5+6 into 1 API call (combined SQL generation)
- Skip critic for simple queries (saves 1 call)
- Skip exploration unless highly ambiguous (saves 1 call)
- Skip interpretation for empty results (saves 1 call)
- Single error retry instead of 2 (saves 1 call)

RESULT: 3-5 API calls per query (was 6-10) - 40% reduction!
"""

import mysql.connector
from tabulate import tabulate
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time

# LLM imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Configuration
LARGE_TABLE_THRESHOLD = 10000
VERY_LARGE_TABLE_THRESHOLD = 100000


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class QuestionAnalysis:
    intent: str
    entities: List[str]
    constraints: List[str]
    implicit_assumptions: List[str]
    ambiguity_score: float
    ambiguous_terms: List[Dict]
    complexity_level: str
    requires_exploration: bool
    reasoning: str


@dataclass
class QueryPlan:
    strategy: str
    steps: List[Dict]
    expected_result: str
    edge_cases: List[str]
    safety_constraints: List[str]
    reasoning: str


@dataclass
class QueryResult:
    success: bool
    data: Optional[List[Dict]]
    error: Optional[str]
    rows_affected: int
    execution_time: float
    sql_used: str = ""
    was_corrected: bool = False
    correction_attempts: int = 0
    natural_language_answer: str = ""


@dataclass 
class PerformanceMetrics:
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    corrections_made: int = 0
    clarifications_asked: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return (self.successful_queries / self.total_queries) * 100


# ============================================================================
# LLM PROVIDER
# ============================================================================

class LLMProvider:
    def __init__(self, gemini_key: str = None, groq_key: str = None, groq_keys: list = None,
                 deepseek_key: str = None, huggingface_key: str = None, 
                 together_key: str = None, openrouter_key: str = None,
                 model_priority: list = None):
        self.gemini_key = gemini_key
        self.groq_key = groq_key
        self.groq_keys = groq_keys or [groq_key] if groq_key else []  # Support multiple Groq keys
        self.deepseek_key = deepseek_key
        self.huggingface_key = huggingface_key
        self.together_key = together_key
        self.openrouter_key = openrouter_key
        self.model_priority = model_priority or ["openrouter"]
        
        self.gemini_model = None
        self.groq_clients = []  # Multiple Groq clients
        self.last_provider = None
        
        # Only initialize models that are in priority list
        # Initialize Gemini only if in priority AND has valid key
        if "gemini" in self.model_priority and GEMINI_AVAILABLE and gemini_key and gemini_key != "" and gemini_key != "YOUR_GEMINI_KEY_HERE":
            try:
                genai.configure(api_key=gemini_key)
                self.gemini_model = genai.GenerativeModel('models/gemini-2.5-flash')
                print("✅ Gemini 2.5 Flash initialized")
            except Exception as e:
                print(f"⚠️ Gemini init failed: {e}")
        
        # Initialize multiple Groq clients if in priority AND has valid keys
        if "groq" in self.model_priority and GROQ_AVAILABLE and self.groq_keys:
            for i, key in enumerate(self.groq_keys):
                if key and key != "":
                    try:
                        client = Groq(api_key=key)
                        self.groq_clients.append(client)
                        print(f"✅ Groq Llama 3.3 70B initialized (Key {i+1})")
                    except Exception as e:
                        print(f"⚠️ Groq Key {i+1} init failed: {e}")
        
        # Check OpenRouter
        if "openrouter" in self.model_priority and openrouter_key and openrouter_key != "YOUR_OPENROUTER_KEY_HERE":
            print("✅ OpenRouter available (FREE $1 credit + free models)")
        
        # Check DeepSeek
        if "deepseek" in self.model_priority and deepseek_key and deepseek_key != "YOUR_DEEPSEEK_KEY_HERE":
            print("✅ DeepSeek R1 available")
        
        # Check HuggingFace
        if "huggingface" in self.model_priority and huggingface_key and huggingface_key != "YOUR_HF_KEY_HERE":
            print("✅ HuggingFace API available (FREE)")
        
        # Check Together AI
        if "together" in self.model_priority and together_key and together_key != "YOUR_TOGETHER_KEY_HERE":
            print("✅ Together AI available (FREE $25 credit)")
        
        # Check if at least one model is configured
        has_model = False
        if "openrouter" in self.model_priority and openrouter_key and openrouter_key != "YOUR_OPENROUTER_KEY_HERE" and openrouter_key != "":
            has_model = True
        if "deepseek" in self.model_priority and deepseek_key and deepseek_key != "YOUR_DEEPSEEK_KEY_HERE" and deepseek_key != "":
            has_model = True
        if "huggingface" in self.model_priority and huggingface_key and huggingface_key != "YOUR_HF_KEY_HERE" and huggingface_key != "":
            has_model = True
        if "together" in self.model_priority and together_key and together_key != "YOUR_TOGETHER_KEY_HERE" and together_key != "":
            has_model = True
        if "groq" in self.model_priority and self.groq_clients:
            has_model = True
        if "gemini" in self.model_priority and self.gemini_model:
            has_model = True
            
        if not has_model:
            print("⚠️ No LLM provider configured! Add API keys to config.py")
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API (Access to ALL models)"""
        if not self.openrouter_key or self.openrouter_key == "YOUR_OPENROUTER_KEY_HERE":
            raise Exception("OpenRouter API key not configured")
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "QueryMind"
        }
        data = {
            "model": "meta-llama/llama-3.1-70b-instruct:free",  # Free model!
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _call_deepseek(self, prompt: str) -> str:
        """Call DeepSeek R1 API (FREE - Best reasoning model)"""
        if not self.deepseek_key or self.deepseek_key == "YOUR_DEEPSEEK_KEY_HERE":
            raise Exception("DeepSeek API key not configured")
        
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.deepseek_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-reasoner",  # R1 reasoning model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call HuggingFace Inference API (FREE)"""
        if not self.huggingface_key or self.huggingface_key == "YOUR_HF_KEY_HERE":
            raise Exception("HuggingFace API key not configured")
        
        # Using Qwen 2.5 72B (great for reasoning)
        url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
        headers = {"Authorization": f"Bearer {self.huggingface_key}"}
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 4000,
                "temperature": 0.1,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result[0]['generated_text']
    
    def _call_together(self, prompt: str) -> str:
        """Call Together AI API (FREE $25 credit)"""
        if not self.together_key or self.together_key == "YOUR_TOGETHER_KEY_HERE":
            raise Exception("Together AI API key not configured")
        
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.together_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4000
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def generate(self, prompt: str, expect_json: bool = True) -> str:
        """Generate response, trying models in priority order"""
        response = None
        errors = []
        
        for provider in self.model_priority:
            try:
                if provider == "openrouter" and self.openrouter_key and self.openrouter_key != "YOUR_OPENROUTER_KEY_HERE":
                    response = self._call_openrouter(prompt)
                    self.last_provider = "OpenRouter"
                    return response
                
                elif provider == "deepseek" and self.deepseek_key and self.deepseek_key != "YOUR_DEEPSEEK_KEY_HERE":
                    response = self._call_deepseek(prompt)
                    self.last_provider = "DeepSeek R1"
                    return response
                
                elif provider == "huggingface" and self.huggingface_key and self.huggingface_key != "YOUR_HF_KEY_HERE":
                    response = self._call_huggingface(prompt)
                    self.last_provider = "HuggingFace"
                    return response
                
                elif provider == "together" and self.together_key and self.together_key != "YOUR_TOGETHER_KEY_HERE":
                    response = self._call_together(prompt)
                    self.last_provider = "Together AI"
                    return response
                
                elif provider == "groq" and self.groq_clients:
                    # Try all Groq keys in sequence
                    for i, groq_client in enumerate(self.groq_clients):
                        try:
                            result = groq_client.chat.completions.create(
                                model="llama-3.3-70b-versatile",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.1,
                                max_tokens=4000
                            )
                            response = result.choices[0].message.content
                            self.last_provider = f"Groq (Key {i+1})"
                            return response
                        except Exception as groq_error:
                            error_msg = str(groq_error)
                            if "429" in error_msg or "rate limit" in error_msg.lower():
                                print(f"⚠️ Groq Key {i+1} rate limited, trying next key...")
                                if i == len(self.groq_clients) - 1:  # Last key
                                    errors.append(f"groq: All {len(self.groq_clients)} keys rate limited")
                                continue
                            else:
                                print(f"⚠️ Groq Key {i+1} error: {error_msg[:50]}")
                                if i == len(self.groq_clients) - 1:  # Last key
                                    errors.append(f"groq: {error_msg[:50]}")
                                continue
                
                elif provider == "gemini" and self.gemini_model and self.gemini_key and self.gemini_key != "" and self.gemini_key != "YOUR_GEMINI_KEY_HERE":
                    result = self.gemini_model.generate_content(prompt)
                    response = result.text
                    self.last_provider = "Gemini"
                    return response
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "rate limit" in error_msg.lower():
                    errors.append(f"{provider}: Rate limited")
                    print(f"⚠️ {provider} rate limited, trying next...")
                else:
                    errors.append(f"{provider}: {error_msg[:50]}")
                    print(f"⚠️ {provider} error: {error_msg[:50]}")
                continue
        
        # All providers failed
        error_summary = "\n".join(errors)
        raise Exception(f"All LLM providers failed!\n{error_summary}")
    
    def parse_json_response(self, response: str) -> Dict:
        """Extract and parse JSON from LLM response"""
        try:
            # Try to find JSON in response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                json_str = match.group()
                # Clean control characters
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
        return {}


# ============================================================================
# SCHEMA ANALYZER (Database Agnostic)
# ============================================================================

class SchemaAnalyzer:
    def __init__(self, conn):
        self.conn = conn
        self.schema_cache = {}
        self.table_sizes = {}
        self.relationships = []
        self.db_name = self._get_database_name()
        self._load_table_sizes()
        self._load_relationships()
    
    def _get_database_name(self) -> str:
        cursor = self.conn.cursor()
        cursor.execute("SELECT DATABASE()")
        name = cursor.fetchone()[0]
        cursor.close()
        return name
        
    def _load_table_sizes(self):
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT TABLE_NAME, TABLE_ROWS 
            FROM information_schema.tables 
            WHERE table_schema = DATABASE()
        """)
        for row in cursor.fetchall():
            self.table_sizes[row['TABLE_NAME']] = row['TABLE_ROWS'] or 0
        cursor.close()
    
    def _load_relationships(self):
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL
        """)
        self.relationships = cursor.fetchall()
        cursor.close()
        
    def get_all_tables(self) -> List[str]:
        cursor = self.conn.cursor()
        cursor.execute("SHOW TABLES")
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables
    
    def get_table_schema(self, table_name: str) -> List[Dict]:
        if table_name in self.schema_cache:
            return self.schema_cache[table_name]
        cursor = self.conn.cursor(dictionary=True)
        cursor.execute(f"DESCRIBE `{table_name}`")
        schema = cursor.fetchall()
        cursor.close()
        self.schema_cache[table_name] = schema
        return schema
    
    def get_foreign_keys(self, table_name: str) -> List[Dict]:
        return [r for r in self.relationships if r['TABLE_NAME'] == table_name]
    
    def get_table_row_count(self, table_name: str) -> int:
        return self.table_sizes.get(table_name, 0)
    
    def get_schema_dump(self) -> str:
        """Get complete schema dump for LLM context"""
        tables = self.get_all_tables()
        dump = f"DATABASE: {self.db_name}\n\n"
        
        for table in tables:
            schema = self.get_table_schema(table)
            row_count = self.get_table_row_count(table)
            fks = self.get_foreign_keys(table)
            
            dump += f"TABLE: {table} ({row_count:,} rows)\n"
            dump += "COLUMNS:\n"
            for col in schema:
                pk = " [PK]" if col['Key'] == 'PRI' else ""
                fk = " [FK]" if col['Key'] == 'MUL' else ""
                dump += f"  - {col['Field']}: {col['Type']}{pk}{fk}\n"
            
            if fks:
                dump += "FOREIGN KEYS:\n"
                for fk in fks:
                    dump += f"  - {fk['COLUMN_NAME']} -> {fk['REFERENCED_TABLE_NAME']}.{fk['REFERENCED_COLUMN_NAME']}\n"
            dump += "\n"
        
        if self.relationships:
            dump += "ALL RELATIONSHIPS:\n"
            for rel in self.relationships:
                dump += f"  - {rel['TABLE_NAME']}.{rel['COLUMN_NAME']} -> {rel['REFERENCED_TABLE_NAME']}.{rel['REFERENCED_COLUMN_NAME']}\n"
        
        return dump


# ============================================================================
# AGENT 1: QUESTION ANALYZER
# ============================================================================

class QuestionAnalyzerAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def analyze(self, question: str) -> QuestionAnalysis:
        prompt = f'''You are a Question Analysis Agent for a natural language to SQL system.
Analyze this question and return JSON:

Question: "{question}"

Return ONLY valid JSON:
{{
  "intent": "retrieve|count|aggregate|compare|list|describe",
  "entities": ["tables, columns, values mentioned"],
  "constraints": ["filters, sorting, limits"],
  "implicit_assumptions": ["unstated assumptions"],
  "ambiguity_score": 0.0-1.0,
  "ambiguous_terms": [
    {{"term": "word", "reason": "why ambiguous", "needs_clarification": true}}
  ],
  "complexity_level": "simple|moderate|complex|multi_step",
  "requires_exploration": true/false,
  "reasoning": "your analysis"
}}

Ambiguous terms include: best, top, recent, popular, active, high, low, most, etc.
'''
        
        response = self.llm.generate(prompt)
        data = self.llm.parse_json_response(response)
        
        return QuestionAnalysis(
            intent=data.get('intent', 'retrieve'),
            entities=data.get('entities', []),
            constraints=data.get('constraints', []),
            implicit_assumptions=data.get('implicit_assumptions', []),
            ambiguity_score=data.get('ambiguity_score', 0.0),
            ambiguous_terms=data.get('ambiguous_terms', []),
            complexity_level=data.get('complexity_level', 'simple'),
            requires_exploration=data.get('requires_exploration', False),
            reasoning=data.get('reasoning', '')
        )


# ============================================================================
# AGENT 2: SCHEMA INTELLIGENCE
# ============================================================================

class SchemaIntelligenceAgent:
    def __init__(self, llm: LLMProvider, schema_analyzer: SchemaAnalyzer):
        self.llm = llm
        self.schema_analyzer = schema_analyzer
    
    def identify_relevant_schema(self, question: str, analysis: QuestionAnalysis) -> Dict:
        schema_dump = self.schema_analyzer.get_schema_dump()
        
        prompt = f'''You are a Schema Intelligence Agent.
Identify relevant tables and columns for this question.

Question: "{question}"
Analysis: {json.dumps(analysis.__dict__)}

Database Schema:
{schema_dump}

Return ONLY valid JSON:
{{
  "relevant_tables": [
    {{
      "table_name": "name",
      "confidence": 0.0-1.0,
      "reason": "why relevant",
      "relevant_columns": ["col1", "col2"]
    }}
  ],
  "relationships": [
    {{
      "from_table": "t1",
      "to_table": "t2", 
      "join_type": "INNER|LEFT",
      "join_condition": "t1.id = t2.id"
    }}
  ],
  "warnings": ["potential issues"]
}}
'''
        
        response = self.llm.generate(prompt)
        return self.llm.parse_json_response(response)


# ============================================================================
# AGENT 3: EXPLORATION AGENT (Probe Queries)
# ============================================================================

class ExplorationAgent:
    def __init__(self, llm: LLMProvider, conn):
        self.llm = llm
        self.conn = conn
    
    def generate_probes(self, question: str, analysis: QuestionAnalysis, schema_info: Dict) -> List[Dict]:
        prompt = f'''You are an Exploration Agent. Generate lightweight probe queries.

Question: "{question}"
Analysis: {json.dumps(analysis.__dict__)}
Schema Info: {json.dumps(schema_info)}

Return ONLY valid JSON:
{{
  "probe_queries": [
    {{
      "purpose": "what this tells us",
      "sql": "SELECT ... LIMIT 10",
      "priority": "high|medium|low"
    }}
  ]
}}

Rules:
- Use COUNT, DISTINCT, MIN, MAX, LIMIT
- Never full table scans
- For "recent" terms, probe date ranges
- For value matching, check actual values exist
'''
        
        response = self.llm.generate(prompt)
        data = self.llm.parse_json_response(response)
        return data.get('probe_queries', [])
    
    def execute_probes(self, probes: List[Dict]) -> List[Dict]:
        results = []
        cursor = self.conn.cursor(dictionary=True)
        
        for probe in probes:
            try:
                sql = probe.get('sql', '')
                if not sql:
                    continue
                cursor.execute(sql)
                data = cursor.fetchall()
                results.append({
                    'purpose': probe.get('purpose'),
                    'sql': sql,
                    'result': data[:10],  # Limit results
                    'success': True
                })
            except Exception as e:
                results.append({
                    'purpose': probe.get('purpose'),
                    'sql': sql,
                    'error': str(e),
                    'success': False
                })
        
        cursor.close()
        return results


# ============================================================================
# AGENT 4: CLARIFICATION GENERATOR
# ============================================================================

class ClarificationAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def generate_clarifications(self, question: str, analysis: QuestionAnalysis, 
                                probe_results: List[Dict]) -> Dict:
        if analysis.ambiguity_score < 0.5:
            return {"clarification_needed": False}
        
        prompt = f'''You are a Clarification Agent. Generate specific clarifying questions.

Question: "{question}"
Ambiguous terms: {json.dumps(analysis.ambiguous_terms)}
Probe results: {json.dumps(probe_results)}

Return ONLY valid JSON:
{{
  "clarification_needed": true,
  "clarifying_questions": [
    {{
      "ambiguous_term": "the term",
      "question": "What do you mean by X?",
      "options": [
        {{"label": "Option 1", "technical_value": "value1", "context": "returns N rows"}}
      ],
      "default_assumption": "what we'll assume if not clarified"
    }}
  ]
}}

Rules:
- Provide SPECIFIC options based on actual data
- Include row counts/context for each option
- Always provide a reasonable default
'''
        
        response = self.llm.generate(prompt)
        return self.llm.parse_json_response(response)


# ============================================================================
# AGENT 5: QUERY PLANNER
# ============================================================================

class QueryPlannerAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def create_plan(self, question: str, schema_info: Dict, 
                    probe_results: List[Dict] = None) -> QueryPlan:
        prompt = f'''You are a Query Planning Agent. Create a step-by-step plan.

Question: "{question}"
Schema: {json.dumps(schema_info)}
Probe Results: {json.dumps(probe_results) if probe_results else "None"}

Return ONLY valid JSON:
{{
  "strategy": "high-level approach",
  "steps": [
    {{
      "step_number": 1,
      "action": "what to do",
      "tables_involved": ["table1"],
      "reasoning": "why"
    }}
  ],
  "expected_result": "what query returns",
  "edge_cases": ["potential issues"],
  "safety_constraints": ["LIMIT clauses, etc"],
  "reasoning": "overall thought process"
}}
'''
        
        response = self.llm.generate(prompt)
        data = self.llm.parse_json_response(response)
        
        return QueryPlan(
            strategy=data.get('strategy', ''),
            steps=data.get('steps', []),
            expected_result=data.get('expected_result', ''),
            edge_cases=data.get('edge_cases', []),
            safety_constraints=data.get('safety_constraints', []),
            reasoning=data.get('reasoning', '')
        )


# ============================================================================
# AGENT 6: SQL GENERATOR
# ============================================================================

class SQLGeneratorAgent:
    def __init__(self, llm: LLMProvider, schema_analyzer: SchemaAnalyzer):
        self.llm = llm
        self.schema_analyzer = schema_analyzer
    
    def generate_sql(self, question: str, plan: QueryPlan, schema_info: Dict) -> Dict:
        prompt = f'''You are a SQL Generation Agent. Generate safe, efficient MySQL.

Question: "{question}"
Plan: {json.dumps(plan.__dict__)}
Schema: {json.dumps(schema_info)}

Return ONLY valid JSON:
{{
  "sql": "SELECT ... LIMIT 1000",
  "explanation": "what this query does",
  "safety_measures": ["LIMIT applied", etc],
  "assumptions": ["any assumptions"]
}}

Rules:
- ALWAYS include LIMIT (default 1000)
- Use explicit JOIN types (INNER/LEFT)
- Use table aliases
- Handle NULLs appropriately
- NO dangerous operations (DELETE, DROP, etc)
'''
        
        response = self.llm.generate(prompt)
        return self.llm.parse_json_response(response)


# ============================================================================
# AGENT 7: SQL CRITIC
# ============================================================================

class SQLCriticAgent:
    def __init__(self, llm: LLMProvider, schema_analyzer: SchemaAnalyzer):
        self.llm = llm
        self.schema_analyzer = schema_analyzer
    
    def review_sql(self, question: str, sql: str, plan: QueryPlan) -> Dict:
        schema_dump = self.schema_analyzer.get_schema_dump()
        
        prompt = f'''You are a SQL Critic Agent. Review this SQL BEFORE execution.

Question: "{question}"
Generated SQL: {sql}
Plan: {json.dumps(plan.__dict__)}
Schema: {schema_dump}

Return ONLY valid JSON:
{{
  "approval_status": "approved|needs_revision|rejected",
  "confidence_score": 0.0-1.0,
  "issues": [
    {{
      "severity": "critical|warning|info",
      "category": "correctness|performance|safety",
      "description": "what's wrong",
      "suggested_fix": "how to fix"
    }}
  ],
  "revised_sql": "corrected SQL or null if approved"
}}

Check for:
- Correctness (answers the question?)
- Safety (LIMIT, no dangerous ops)
- Performance (unnecessary joins?)
- Column/table names exist in schema
'''
        
        response = self.llm.generate(prompt)
        return self.llm.parse_json_response(response)


# ============================================================================
# AGENT 8: ERROR ANALYZER
# ============================================================================

class ErrorAnalyzerAgent:
    def __init__(self, llm: LLMProvider, schema_analyzer: SchemaAnalyzer):
        self.llm = llm
        self.schema_analyzer = schema_analyzer
    
    def analyze_error(self, question: str, failed_sql: str, error: str) -> Dict:
        schema_dump = self.schema_analyzer.get_schema_dump()
        
        prompt = f'''You are an Error Analysis Agent. Diagnose SQL failure and fix it.

Question: "{question}"
Failed SQL: {failed_sql}
Error: {error}
Schema: {schema_dump}

Return ONLY valid JSON:
{{
  "error_type": "syntax|missing_table|missing_column|type_mismatch|other",
  "root_cause": "what went wrong",
  "user_friendly_explanation": "simple explanation",
  "revised_sql": "corrected SQL query",
  "should_retry": true/false
}}

Common fixes:
- Column not found: check schema for correct name
- Syntax error: fix SQL syntax
- Ambiguous column: add table alias
- Type mismatch: fix data types
'''
        
        response = self.llm.generate(prompt)
        return self.llm.parse_json_response(response)


# ============================================================================
# AGENT 9: RESULT INTERPRETER
# ============================================================================

class ResultInterpreterAgent:
    def __init__(self, llm: LLMProvider):
        self.llm = llm
    
    def interpret(self, question: str, sql: str, results: List[Dict], 
                  row_count: int) -> Dict:
        # Limit results for prompt
        sample_results = results[:10] if results else []
        
        prompt = f'''You are a Result Interpretation Agent. Turn SQL results into natural language.

Question: "{question}"
SQL: {sql}
Results ({row_count} rows): {json.dumps(sample_results, default=str)}

Return ONLY valid JSON:
{{
  "natural_language_answer": "conversational answer to the question",
  "key_insights": ["interesting findings"],
  "data_summary": {{
    "total_rows": {row_count},
    "key_metrics": {{}}
  }},
  "follow_up_suggestions": ["related questions"]
}}

Rules:
- Start with DIRECT answer to the question
- For 0 rows, explain what that MEANS (not just "no results")
- Highlight interesting patterns
- Be conversational, not technical
'''
        
        response = self.llm.generate(prompt)
        return self.llm.parse_json_response(response)


# ============================================================================
# SAFE QUERY EXECUTOR
# ============================================================================

class SafeQueryExecutor:
    def __init__(self, conn, max_rows: int = 1000):
        self.conn = conn
        self.max_rows = max_rows
    
    def validate_query(self, sql: str) -> Tuple[bool, str]:
        """Check for dangerous operations"""
        sql_upper = sql.upper().strip()
        
        dangerous = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 
                     'ALTER', 'TRUNCATE', 'GRANT', 'REVOKE']
        for op in dangerous:
            if sql_upper.startswith(op) or f' {op} ' in sql_upper:
                return False, f"🛡️ BLOCKED: {op} not allowed (read-only system)"
        
        if sql.count(';') > 1:
            return False, "🛡️ BLOCKED: Multiple statements not allowed"
        
        return True, ""
    
    def execute(self, sql: str) -> QueryResult:
        start_time = time.time()
        
        # Validate
        is_valid, error = self.validate_query(sql)
        if not is_valid:
            return QueryResult(False, None, error, 0, 0, sql)
        
        try:
            cursor = self.conn.cursor(dictionary=True)
            
            # Add LIMIT if not present
            sql_upper = sql.upper()
            if 'SELECT' in sql_upper and 'LIMIT' not in sql_upper:
                sql = sql.rstrip(';') + f' LIMIT {self.max_rows}'
            
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            
            exec_time = time.time() - start_time
            return QueryResult(True, results, None, len(results), exec_time, sql)
            
        except Exception as e:
            return QueryResult(False, None, str(e), 0, time.time() - start_time, sql)


# ============================================================================
# AGENT 10: MASTER ORCHESTRATOR
# ============================================================================

class MasterOrchestrator:
    def __init__(self, db_config: Dict, gemini_key: str = None, groq_key: str = None, groq_keys: list = None,
                 deepseek_key: str = None, huggingface_key: str = None, 
                 together_key: str = None, openrouter_key: str = None,
                 model_priority: list = None):
        self.conn = mysql.connector.connect(**db_config)
        self.llm = LLMProvider(
            gemini_key=gemini_key, 
            groq_key=groq_key,
            groq_keys=groq_keys,  # Support multiple Groq keys
            deepseek_key=deepseek_key,
            huggingface_key=huggingface_key,
            together_key=together_key,
            openrouter_key=openrouter_key,
            model_priority=model_priority
        )
        self.schema_analyzer = SchemaAnalyzer(self.conn)
        self.executor = SafeQueryExecutor(self.conn)
        self.metrics = PerformanceMetrics()
        
        # Initialize all agents
        self.question_analyzer = QuestionAnalyzerAgent(self.llm)
        self.schema_intelligence = SchemaIntelligenceAgent(self.llm, self.schema_analyzer)
        self.exploration_agent = ExplorationAgent(self.llm, self.conn)
        self.clarification_agent = ClarificationAgent(self.llm)
        self.query_planner = QueryPlannerAgent(self.llm)
        self.sql_generator = SQLGeneratorAgent(self.llm, self.schema_analyzer)
        self.sql_critic = SQLCriticAgent(self.llm, self.schema_analyzer)
        self.error_analyzer = ErrorAnalyzerAgent(self.llm, self.schema_analyzer)
        self.result_interpreter = ResultInterpreterAgent(self.llm)
    
    def _combined_analysis(self, question: str) -> Tuple[QuestionAnalysis, Dict]:
        """OPTIMIZATION: Combine question analysis + schema intelligence in 1 API call"""
        
        # SAFETY CHECK: Block dangerous operations immediately
        question_upper = question.upper()
        dangerous_keywords = ['DELETE', 'DROP', 'TRUNCATE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE', 'GRANT', 'REVOKE']
        for keyword in dangerous_keywords:
            if keyword in question_upper:
                # Return a special analysis that will be caught later
                return QuestionAnalysis(
                    intent='dangerous_operation',
                    entities=[],
                    constraints=[],
                    implicit_assumptions=[],
                    ambiguity_score=0.0,
                    ambiguous_terms=[],
                    complexity_level='blocked',
                    requires_exploration=False,
                    reasoning=f'SECURITY: Detected dangerous operation "{keyword}" - read-only system'
                ), {}
        
        schema_dump = self.schema_analyzer.get_schema_dump()
        
        prompt = f'''You are an intelligent SQL assistant. Analyze this question AND identify relevant schema.

Question: "{question}"

Database Schema:
{schema_dump}

Return ONLY valid JSON with BOTH analysis and schema info:
{{
  "analysis": {{
    "intent": "retrieve|count|aggregate|compare|list",
    "complexity_level": "simple|moderate|complex",
    "ambiguity_score": 0.0-1.0,
    "ambiguous_terms": ["term1", "term2"],
    "reasoning": "brief analysis"
  }},
  "schema": {{
    "relevant_tables": [
      {{
        "table_name": "name",
        "confidence": 0.9,
        "relevant_columns": ["col1", "col2"]
      }}
    ],
    "relationships": [
      {{"from_table": "t1", "to_table": "t2", "join_condition": "t1.id = t2.id"}}
    ]
  }}
}}
'''
        
        response = self.llm.generate(prompt)
        data = self.llm.parse_json_response(response)
        
        # Parse analysis
        analysis_data = data.get('analysis', {})
        analysis = QuestionAnalysis(
            intent=analysis_data.get('intent', 'retrieve'),
            entities=[],
            constraints=[],
            implicit_assumptions=[],
            ambiguity_score=analysis_data.get('ambiguity_score', 0.0),
            ambiguous_terms=[{'term': t, 'reason': '', 'needs_clarification': False} 
                           for t in analysis_data.get('ambiguous_terms', [])],
            complexity_level=analysis_data.get('complexity_level', 'simple'),
            requires_exploration=analysis_data.get('ambiguity_score', 0.0) > 0.7,
            reasoning=analysis_data.get('reasoning', '')
        )
        
        # Parse schema info
        schema_info = data.get('schema', {})
        
        return analysis, schema_info
    
    def _combined_sql_generation(self, question: str, schema_info: Dict, 
                                 probe_results: List[Dict], analysis: QuestionAnalysis) -> Tuple[str, QueryPlan]:
        """OPTIMIZATION: Combine query planning + SQL generation in 1 API call"""
        
        prompt = f'''You are a SQL expert. Generate SQL directly with planning context.

Question: "{question}"
Intent: {analysis.intent}
Complexity: {analysis.complexity_level}
Schema: {json.dumps(schema_info)}
Probe Results: {json.dumps(probe_results) if probe_results else "None"}

Return ONLY valid JSON:
{{
  "plan": {{
    "strategy": "high-level approach",
    "reasoning": "why this approach"
  }},
  "sql": "SELECT ... LIMIT 1000"
}}

CRITICAL SAFETY RULES:
- ONLY generate SELECT queries
- NEVER generate DELETE, UPDATE, INSERT, DROP, ALTER, TRUNCATE
- If user asks for dangerous operation, return: {{"sql": "BLOCKED", "plan": {{"strategy": "Security block", "reasoning": "Read-only system"}}}}
- ALWAYS include LIMIT (default 1000)
- Use explicit JOIN types
- Handle NULLs appropriately
'''
        
        response = self.llm.generate(prompt)
        data = self.llm.parse_json_response(response)
        
        sql = data.get('sql', '')
        plan_data = data.get('plan', {})
        
        plan = QueryPlan(
            strategy=plan_data.get('strategy', ''),
            steps=[],
            expected_result='',
            edge_cases=[],
            safety_constraints=[],
            reasoning=plan_data.get('reasoning', '')
        )
        
        return sql, plan
    
    def process_question(self, question: str, show_reasoning: bool = True) -> Dict:
        """Main orchestration flow - OPTIMIZED for fewer API calls"""
        total_start = time.time()
        
        # Store reasoning trace
        reasoning_trace = {
            'steps': [],
            'total_time': 0,
            'success': False
        }
        
        if show_reasoning:
            print("\n" + "="*80)
            print(f"🎯 QUESTION: {question}")
            print("="*80)
        
        # OPTIMIZATION 1: Combined Analysis (1 API call instead of 2)
        # Merge Question Analysis + Schema Intelligence into single call
        step_start = time.time()
        if show_reasoning:
            print("\n📊 STEP 1: Combined Analysis (Question + Schema)...")
        
        analysis, schema_info = self._combined_analysis(question)
        step_time = time.time() - step_start
        
        reasoning_trace['steps'].append({
            'number': 1,
            'name': 'Combined Analysis',
            'icon': '📊',
            'status': 'completed',
            'time': step_time,
            'details': {
                'intent': analysis.intent,
                'complexity': analysis.complexity_level,
                'ambiguity_score': analysis.ambiguity_score,
                'relevant_tables': [t['table_name'] for t in schema_info.get('relevant_tables', [])],
                'optimization': 'Merged 2 agents into 1 API call'
            }
        })
        
        if show_reasoning:
            print(f"   Intent: {analysis.intent} | Complexity: {analysis.complexity_level}")
            print(f"   Ambiguity: {analysis.ambiguity_score:.1f}")
            if schema_info.get('relevant_tables'):
                tables = [t['table_name'] for t in schema_info['relevant_tables']]
                print(f"   Tables: {tables}")
            print(f"   ✅ Saved 1 API call (merged analysis)")
        
        # SAFETY CHECK: Block dangerous operations immediately
        if analysis.intent == 'dangerous_operation':
            reasoning_trace['steps'].append({
                'number': 'BLOCKED',
                'name': 'Security Validation',
                'icon': '🛡️',
                'status': 'blocked',
                'time': 0,
                'details': {
                    'reason': analysis.reasoning,
                    'action': 'Query blocked before SQL generation'
                }
            })
            
            if show_reasoning:
                print(f"\n🛡️ SECURITY BLOCK: {analysis.reasoning}")
                print("\n" + "="*80)
                print("❌ QUERY BLOCKED")
                print("="*80)
                print(f"\n🛡️ This system is READ-ONLY. Dangerous operations are not allowed.")
                print(f"   Allowed: SELECT queries only")
                print(f"   Blocked: DELETE, UPDATE, INSERT, DROP, ALTER, etc.")
            
            reasoning_trace['total_time'] = time.time() - total_start
            reasoning_trace['success'] = False
            
            result = QueryResult(
                success=False,
                data=None,
                error=f"🛡️ BLOCKED: {analysis.reasoning}",
                rows_affected=0,
                execution_time=0,
                sql_used="",
                was_corrected=False,
                correction_attempts=0,
                natural_language_answer=f"🛡️ Security Block: This system is read-only. {analysis.reasoning}"
            )
            
            self.metrics.total_queries += 1
            self.metrics.failed_queries += 1
            
            return {
                'question': question,
                'analysis': analysis,
                'plan': None,
                'sql': '',
                'result': result,
                'success': False,
                'reasoning_trace': reasoning_trace
            }
        
        # OPTIMIZATION 2: Skip exploration for simple queries (saves 1-2 API calls)
        probe_results = []
        if analysis.ambiguity_score > 0.7:  # Only for HIGHLY ambiguous (was 0.5)
            step_start = time.time()
            if show_reasoning:
                print("\n🔬 STEP 2: Exploration (High Ambiguity Detected)...")
            
            probes = self.exploration_agent.generate_probes(question, analysis, schema_info)
            probe_results = self.exploration_agent.execute_probes(probes)
            step_time = time.time() - step_start
            
            reasoning_trace['steps'].append({
                'number': 2,
                'name': 'Data Exploration',
                'icon': '🔬',
                'status': 'completed',
                'time': step_time,
                'details': {
                    'probes_run': len(probe_results),
                    'findings': [p.get('purpose') for p in probe_results[:2]]
                }
            })
            
            if show_reasoning:
                print(f"   Ran {len(probe_results)} probes")
        elif show_reasoning:
            print("\n⏭️  STEP 2: Exploration SKIPPED (low ambiguity - saved 1 API call)")
        
        # OPTIMIZATION 3: Combined Planning + SQL Generation (1 API call instead of 2)
        step_start = time.time()
        if show_reasoning:
            print("\n⚡ STEP 3: Combined Planning + SQL Generation...")
        
        sql, plan = self._combined_sql_generation(question, schema_info, probe_results, analysis)
        step_time = time.time() - step_start
        
        reasoning_trace['steps'].append({
            'number': 3,
            'name': 'SQL Generation',
            'icon': '⚡',
            'status': 'completed',
            'time': step_time,
            'details': {
                'sql': sql,
                'strategy': plan.strategy if plan else 'Direct generation',
                'optimization': 'Merged planning + generation'
            }
        })
        
        if show_reasoning:
            print(f"   SQL: {sql[:80]}..." if len(sql) > 80 else f"   SQL: {sql}")
            print(f"   ✅ Saved 1 API call (merged planning + generation)")
        
        # OPTIMIZATION 4: Skip critic for simple queries (saves 1 API call)
        if analysis.complexity_level in ['complex', 'multi_step'] or analysis.ambiguity_score > 0.6:
            step_start = time.time()
            if show_reasoning:
                print("\n🔎 STEP 4: SQL Validation (Complex Query)...")
            
            review = self.sql_critic.review_sql(question, sql, plan)
            step_time = time.time() - step_start
            
            reasoning_trace['steps'].append({
                'number': 4,
                'name': 'SQL Validation',
                'icon': '🔎',
                'status': 'completed',
                'time': step_time,
                'details': {
                    'approval_status': review.get('approval_status', 'unknown'),
                    'confidence': review.get('confidence_score', 0)
                }
            })
            
            if review.get('approval_status') == 'needs_revision' and review.get('revised_sql'):
                sql = review['revised_sql']
                if show_reasoning:
                    print(f"   ⚠️ SQL revised")
        elif show_reasoning:
            print("\n⏭️  STEP 4: Validation SKIPPED (simple query - saved 1 API call)")
        
        # STEP 5: Execute
        step_start = time.time()
        if show_reasoning:
            print("\n🚀 STEP 5: Executing Query...")
        
        result = self.executor.execute(sql)
        step_time = time.time() - step_start
        
        reasoning_trace['steps'].append({
            'number': 5,
            'name': 'Query Execution',
            'icon': '🚀',
            'status': 'completed' if result.success else 'failed',
            'time': step_time,
            'details': {
                'success': result.success,
                'rows': result.rows_affected,
                'error': result.error if not result.success else None
            }
        })
        
        # STEP 6: Error Recovery (if needed) - Only 1 retry to save API calls
        if not result.success:
            step_start = time.time()
            if show_reasoning:
                print(f"\n🔧 STEP 6: Error Recovery...")
                print(f"   Error: {result.error[:60]}...")
            
            error_analysis = self.error_analyzer.analyze_error(question, sql, result.error)
            step_time = time.time() - step_start
            
            reasoning_trace['steps'].append({
                'number': 6,
                'name': 'Error Recovery',
                'icon': '🔧',
                'status': 'completed',
                'time': step_time,
                'details': {
                    'error_type': error_analysis.get('error_type'),
                    'root_cause': error_analysis.get('root_cause')
                }
            })
            
            if error_analysis.get('should_retry') and error_analysis.get('revised_sql'):
                sql = error_analysis['revised_sql']
                result = self.executor.execute(sql)
                result.was_corrected = True
                result.correction_attempts = 1
        
        # STEP 7: Interpret Results (only for successful queries with data)
        if result.success and result.data:
            step_start = time.time()
            if show_reasoning:
                print("\n💬 STEP 7: Interpreting Results...")
            
            interpretation = self.result_interpreter.interpret(
                question, sql, result.data or [], result.rows_affected)
            result.natural_language_answer = interpretation.get(
                'natural_language_answer', f"Found {result.rows_affected} results.")
            step_time = time.time() - step_start
            
            reasoning_trace['steps'].append({
                'number': 7,
                'name': 'Result Interpretation',
                'icon': '💬',
                'status': 'completed',
                'time': step_time,
                'details': {
                    'answer': result.natural_language_answer
                }
            })
        else:
            # Simple answer for empty results or failures
            if result.success:
                result.natural_language_answer = f"Query executed successfully. Found {result.rows_affected} results."
            else:
                result.natural_language_answer = f"Query failed: {result.error}"
            if show_reasoning:
                print("\n⏭️  STEP 7: Interpretation SKIPPED (saved 1 API call)")
        
        # Update metrics
        self.metrics.total_queries += 1
        if result.success:
            self.metrics.successful_queries += 1
        else:
            self.metrics.failed_queries += 1
        if result.was_corrected:
            self.metrics.corrections_made += 1
        
        # Finalize reasoning trace
        reasoning_trace['total_time'] = time.time() - total_start
        reasoning_trace['success'] = result.success
        
        # Display Results
        if show_reasoning:
            print("\n" + "="*80)
            print("📊 RESULTS")
            print("="*80)
            
            if result.success:
                print(f"\n💬 ANSWER: {result.natural_language_answer}")
                
                if result.data:
                    print(f"\n📋 DATA ({result.rows_affected} rows):\n")
                    # Limit columns for readability
                    display_data = result.data[:20]
                    if display_data:
                        # Get column names
                        all_cols = list(display_data[0].keys())
                        # If too many columns, show only first 5
                        if len(all_cols) > 5:
                            show_cols = all_cols[:5]
                            truncated_data = [{k: row[k] for k in show_cols} for row in display_data]
                            print(tabulate(truncated_data, headers="keys", tablefmt="simple", maxcolwidths=30))
                            print(f"   (Showing {len(show_cols)} of {len(all_cols)} columns)")
                        else:
                            print(tabulate(display_data, headers="keys", tablefmt="simple", maxcolwidths=30))
                    if result.rows_affected > 20:
                        print(f"   ... and {result.rows_affected - 20} more rows")
            else:
                print(f"\n❌ FAILED: {result.error}")
            
            total_time = time.time() - total_start
            print(f"\n⏱️ Total Time: {total_time:.2f}s | Success Rate: {self.metrics.success_rate:.1f}%")
        
        return {
            'question': question,
            'analysis': analysis,
            'plan': plan,
            'sql': sql,
            'result': result,
            'success': result.success,
            'reasoning_trace': reasoning_trace  # NEW: Include reasoning trace
        }
    
    def show_metrics(self):
        print("\n" + "="*80)
        print("📊 PERFORMANCE METRICS")
        print("="*80)
        print(f"Total Queries: {self.metrics.total_queries}")
        print(f"Successful: {self.metrics.successful_queries}")
        print(f"Failed: {self.metrics.failed_queries}")
        print(f"Success Rate: {self.metrics.success_rate:.1f}%")
        print(f"Auto-Corrections: {self.metrics.corrections_made}")
        print(f"Clarifications Asked: {self.metrics.clarifications_asked}")
    
    def show_schema(self):
        print("\n" + "="*80)
        print(f"📊 DATABASE SCHEMA: {self.schema_analyzer.db_name}")
        print("="*80)
        for table in self.schema_analyzer.get_all_tables():
            schema = self.schema_analyzer.get_table_schema(table)
            count = self.schema_analyzer.get_table_row_count(table)
            print(f"\n📋 {table} ({count:,} rows)")
            print(tabulate(schema, headers="keys", tablefmt="simple"))
    
    def close(self):
        if self.conn:
            self.conn.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    # Import API Keys from config
    try:
        from config import (GEMINI_API_KEY, GROQ_API_KEY, DEEPSEEK_API_KEY, 
                           HUGGINGFACE_API_KEY, TOGETHER_API_KEY, OPENROUTER_API_KEY,
                           MODEL_PRIORITY)
    except ImportError:
        print("⚠️ config.py not found! Using default keys...")
        GEMINI_API_KEY = None
        GROQ_API_KEY = None
        DEEPSEEK_API_KEY = None
        HUGGINGFACE_API_KEY = None
        TOGETHER_API_KEY = None
        OPENROUTER_API_KEY = None
        MODEL_PRIORITY = ["groq", "gemini"]
    
    print("🚀 Natural Language to SQL - MULTI-AGENT SYSTEM")
    print("="*80)
    
    # Ask user for database configuration
    print("\n📊 DATABASE CONFIGURATION")
    print("-"*40)
    
    host = input("Host [localhost]: ").strip() or "localhost"
    user = input("Username [THOR]: ").strip() or "THOR"
    password = input("Password [Thor_123]: ").strip() or "Thor_123"
    database = input("Database name: ").strip()
    
    if not database:
        print("❌ Database name is required!")
        return
    
    DB_CONFIG = {
        'host': host,
        'user': user,
        'password': password,
        'database': database
    }
    
    print("\n🤖 10 SPECIALIZED AGENTS (OPTIMIZED):")
    print("  1. Question Analyzer - Understands intent & ambiguity")
    print("  2. Schema Intelligence - Identifies relevant tables")
    print("  3. Exploration Agent - Probes data (conditional)")
    print("  4. Clarification Agent - Handles ambiguous questions (conditional)")
    print("  5. Query Planner - Creates step-by-step strategy")
    print("  6. SQL Generator - Writes safe, efficient SQL")
    print("  7. SQL Critic - Reviews before execution (conditional)")
    print("  8. Error Analyzer - Diagnoses and fixes failures (conditional)")
    print("  9. Result Interpreter - Natural language answers")
    print("  10. Master Orchestrator - Coordinates everything")
    print("\n⚡ OPTIMIZATIONS: 3-5 API calls per query (was 6-10) - 40% reduction!")
    print()
    
    try:
        system = MasterOrchestrator(
            DB_CONFIG, 
            gemini_key=GEMINI_API_KEY, 
            groq_key=GROQ_API_KEY,
            deepseek_key=DEEPSEEK_API_KEY,
            huggingface_key=HUGGINGFACE_API_KEY,
            together_key=TOGETHER_API_KEY,
            openrouter_key=OPENROUTER_API_KEY,
            model_priority=MODEL_PRIORITY
        )
        print(f"\n✅ Connected to database: {system.schema_analyzer.db_name}")
        print(f"📋 Tables found: {len(system.schema_analyzer.get_all_tables())}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    print("\n" + "="*80)
    print("Commands:")
    print("  - Type any question in natural language")
    print("  - 'schema' - Show database schema")
    print("  - 'metrics' - Show performance metrics")
    print("  - 'demo' - Run demo queries")
    print("  - 'quit' - Exit")
    print("="*80)
    
    while True:
        try:
            user_input = input("\n💬 Ask: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            if not user_input:
                continue
            if user_input.lower() == 'schema':
                system.show_schema()
                continue
            if user_input.lower() == 'metrics':
                system.show_metrics()
                continue
            if user_input.lower() == 'demo':
                run_demo(system)
                continue
            
            system.process_question(user_input)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    system.close()
    print("\n👋 Goodbye!")


def run_demo(system):
    """Run demo queries based on actual database"""
    tables = system.schema_analyzer.get_all_tables()
    
    # Find a table with data
    sample_table = None
    for t in tables:
        if system.schema_analyzer.get_table_row_count(t) > 0:
            sample_table = t
            break
    
    if not sample_table:
        sample_table = tables[0] if tables else "table"
    
    print("\n" + "="*80)
    print("🎯 DEMO: Multi-Agent System in Action")
    print(f"📊 Database: {system.schema_analyzer.db_name}")
    print("="*80)
    
    demos = [
        f"How many rows are in the {sample_table} table?",
        "What tables exist in this database?",
        "Which table has the most rows?",
        f"Show me the first 5 records from {sample_table}",
    ]
    
    for query in demos:
        print(f"\n{'='*80}")
        system.process_question(query)
        input("\n[Press Enter to continue...]")


if __name__ == "__main__":
    main()
