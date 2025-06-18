import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from app.interface.query_plus import ask_question
from app.tools.summarize import summarize_chunks 
from app.interface.query_eval import evaluate_response
from app.interface.query_plus import load_vectorstore

class Agent:
    def __init__(self, name, run_fn):
        self.name = name
        self.run_fn = run_fn

    def run(self, input_data, context=None):
        return self.run_fn(input_data, context or {})

class AgentOrchestrator:
    def __init__(self, agents):
        self.agents = agents

    def run(self, input_data):
        context = {}
        for agent in self.agents:
            output = agent.run(input_data, context)
            context[agent.name] = output
        return context


# Example agent functions (in practice, import from query_plus, summarize, etc.)
def retrieval_agent(input_data, context):
    db = load_vectorstore()
    context_docs = db.similarity_search(input_data, k=3)
    context["docs"] = context_docs
    return context_docs


def summarize_agent(input_data, context):
    docs = context.get("docs", [])
    return summarize_chunks(docs)


def answer_agent(input_data, context):
    return ask_question(question=input_data, filter_all=True)


def critic_agent(input_data, context):
    answer = context.get("answer_agent")
    rules = [
        {"type": "contains", "value": "AI"},
        {"type": "length_gt", "value": 50},
    ]
    return evaluate_response(answer, expected_answer="AI/ML engineer", rules=rules)


if __name__ == "__main__":
    orchestrator = AgentOrchestrator([
        Agent("retrieval_agent", retrieval_agent),
        Agent("summarize_agent", summarize_agent),
        Agent("answer_agent", answer_agent),
        Agent("critic_agent", critic_agent)
    ])

    user_query = "What did the Mayo Clinic project involve?"
    results = orchestrator.run(user_query)
    print("\n🧠 Final Agent Context:")
    for k, v in results.items():
        print(f"\n🔹 {k} →", v if isinstance(v, str) else str(v)[:300])
        
        