import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure")

API_KEY = os.getenv("OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = "2025-03-01-preview"

AZURE_MODELS = {
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}
BEDROCK_MODELS = {
    # 7 RPM
    "claude-3.5-sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
}
VERTEX_MODELS = {
    # Have thinking, good reasoning and math
    "gemini-3-pro": "gemini-3-pro-preview",
    # Might need if resources issue
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.0-flash": "gemini-2.0-flash-001",
    "gemini-1.5-pro": "gemini-1.5-pro-002",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
    # Open weight
    "llama-4-maverick": "meta/llama-4-maverick-17b-128e-instruct-maas",
    "llama-4-scout": "meta/llama-4-scout-17b-16e-instruct-maas",
}

VERTEX_LLAMA_LOCATION = os.getenv("VERTEX_LLAMA_LOCATION", "us-east5")

MODEL = os.getenv("LLM_MODEL", "gpt-4o")

def get_model_id() -> str:
    if LLM_PROVIDER == "bedrock":
        return BEDROCK_MODELS.get(MODEL, MODEL)
    if LLM_PROVIDER == "vertex":
        return VERTEX_MODELS.get(MODEL, MODEL)
    return AZURE_MODELS.get(MODEL, MODEL)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "global")
VERTEX_CREDENTIALS_FILE = os.getenv("VERTEX_CREDENTIALS_FILE")
VERTEX_THINKING_BUDGET = int(os.getenv("VERTEX_THINKING_BUDGET", "4096"))

NUM_SEEDS = 30
DOMAINS = ["medical", "banking", "filesystem"]
DIFFICULTIES = ["hard", "medium", "easy"]
METHODS = ["Oracle", "Random-Safe", "ReAct", "ReAct-CoT", "ReAct-Conservative", "LLM-Check", "Code-Check", "VerAct"]

# Too many nodes significantly increase solving time, might worth noting
NUM_NODES_BY_DIFFICULTY = {
    "easy": 8,
    "medium": 10,
    "hard": 12,
    "adversarial": 14
}

def get_num_nodes(difficulty: str) -> int:
    return NUM_NODES_BY_DIFFICULTY[difficulty]

MIN_PATH_LENGTH = 3

# Empirically ~95% of seeds generate solvable env on first try
# 10 retries handles edge cases without slowing down benchmarks
SOLVABILITY_MAX_RETRIES = 10
SOLVABILITY_MAX_DEPTH = 30
SOLVABILITY_MAX_ITER = 5000
SOLVABILITY_TIMEOUT = 30.0

# Need proof with ablation study
N_CANDIDATES = 5
MAX_RETRIES = 5

MAX_DEPTH = 30
# Tuned by mass trial and error
TEMPERATURE = 0.4

# No creative in check
LLM_CHECK_TEMPERATURE = 0.1
LLM_CHECK_MAX_TOKENS = 500

PROPOSAL_MAX_TOKENS = 1000

Z3_TIMEOUT_MS = 5000

OUTPUT_DIR = "benchmark_results"
FIGURES_DIR = "figures"
VERBOSE = True
SAVE_DETAILED_LOGS = True
