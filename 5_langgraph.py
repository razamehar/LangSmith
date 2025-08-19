# pip install -U langgraph langchain-openai pydantic python-dotenv langsmith

import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langsmith import traceable
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os

# ---------- Setup ----------
load_dotenv()
os.environ['LANGSMITH_PROJECT'] = 'LangGraph'

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------- Structured schema & model ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# ---------- Sample essay ----------
essay2 = """ AI and Medicine
Artificial Intelligence (AI) is transforming the field of medicine by enhancing the way diseases are diagnosed, treated, and managed. AI-powered systems can analyze large amounts of medical data, such as patient records, lab results, and medical images, much faster and more accurately than humans. For example, AI tools are being used to detect early signs of cancer in X-rays and MRI scans, predict the risk of heart disease, and assist doctors in making personalized treatment plans. By reducing human error and speeding up diagnosis, AI has the potential to improve patient outcomes and save countless lives.

Beyond diagnosis, AI is also revolutionizing drug discovery and healthcare management. Machine learning models can predict how different drugs will interact with the body, significantly reducing the time and cost required to develop new medicines. In hospitals, AI-based systems help optimize schedules, manage patient flow, and even support virtual assistants that guide patients through their treatments. However, while AI brings enormous opportunities, it also raises concerns about privacy, ethics, and the need for human oversight. If used responsibly, AI can serve as a powerful partner to doctors, making healthcare more efficient, accurate, and accessible.

"""

# ---------- LangGraph state ----------
class UPSCState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]  # merges parallel lists
    avg_score: float

# ---------- Traced node functions ----------
@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: UPSCState):
    prompt = (
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"language_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_analysis_fn", tags=["dimension:analysis"], metadata={"dimension": "analysis"})
def evaluate_analysis(state: UPSCState):
    prompt = (
        "Evaluate the depth of analysis of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_thought_fn", tags=["dimension:clarity"], metadata={"dimension": "clarity_of_thought"})
def evaluate_thought(state: UPSCState):
    prompt = (
        "Evaluate the clarity of thought of the following essay and provide feedback "
        "and assign a score out of 10.\n\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="final_evaluation_fn", tags=["aggregate"])
def final_evaluation(state: UPSCState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
    )
    overall = model.invoke(prompt).content
    scores = state.get("individual_scores", []) or []
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}

# ---------- Build graph ----------
graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

# Fan-out → join
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# ---------- Direct invoke without wrapper ----------
if __name__ == "__main__":
    result = workflow.invoke(
        {"essay": essay2},
        config={
            "run_name": "evaluate_upsc_essay",  # becomes root run name
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay2),
                "model": "gpt-4o-mini",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

    print("\n=== Evaluation Results ===")
    print("Language feedback:\n", result.get("language_feedback", ""), "\n")
    print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
    print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
    print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
    print("Individual scores:", result.get("individual_scores", []))
    print("Average score:", result.get("avg_score", 0.0))
