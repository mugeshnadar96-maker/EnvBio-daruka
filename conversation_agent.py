"""
conversation_agent.py
----------------------
Conversational orchestration layer: intent classification, sliding-window
multi-turn memory (accumulates known SiteState fields across turns, mirroring
LangChain's ConversationBufferWindowMemory(k=5) semantics), clarifying
questions when input is incomplete, RAG retrieval from ChromaDB, and fusion
with the hardcoded multi-metric reasoning engine into the mandatory output
format.

NOTE on LangChain: the spec calls for LangChain's ConversationBufferWindowMemory
and an agent wrapper. This module implements the same *contract* (last-k-turns
window, accumulating slot state, retriever with k=5) directly, because the
sandbox this was built in cannot reach PyPI's dependency-resolution network
lookups reliably for the full `langchain` metapackage within the available
disk budget (see README, "Known Sandbox Limitation"). Swapping in
`langchain.memory.ConversationBufferWindowMemory` and
`langchain.vectorstores.Chroma(...).as_retriever(search_kwargs={"k": 5})`
is a drop-in replacement — see `langchain_adapter_notes.md`.
"""

import re
import os
from collections import deque
from dataclasses import asdict
from typing import Optional, List, Dict, Any

import chromadb

from embedder import get_embedder
from reasoning_engine import SiteState, run_reasoning, missing_core_vars, Recommendation
from bhi_lookup import lookup_bhi

DB_DIR = os.path.join(os.path.dirname(__file__), "env_knowledge_db")
COLLECTION_NAME = "environmental_knowledge"
MEMORY_WINDOW = 5
RETRIEVAL_K = 5

INTENTS = ["diagnosis", "recommendation", "metric_explanation", "data_request"]

METRIC_EXPLAIN_TRIGGERS = [
    "what is", "what does", "explain", "define", "how is", "mean by"
]
DATA_REQUEST_TRIGGERS = [
    "show me the data", "raw data", "which dataset", "where does this come from",
    "source data", "download", "citation list"
]
RECOMMENDATION_TRIGGERS = [
    "recommend", "what should i do", "how do i improve", "how can i improve",
    "suggest", "advice", "fix", "improve"
]


# ---------------------------------------------------------------------------
# Input parsing: free text -> SiteState
# ---------------------------------------------------------------------------

_LAND_USE_KEYWORDS = {
    "monoculture wheat": "monoculture_wheat",
    "monoculture cereal": "monoculture_cereal",
    "monoculture": "monoculture",
    "mixed cropping": "mixed",
    "mixed crop": "mixed",
    "polyculture": "mixed",
    "agroforestry": "agroforestry",
    "pasture": "pasture",
    "grassland": "grassland",
    "fallow": "fallow",
}


def parse_free_text(text: str) -> SiteState:
    t = text.lower()
    s = SiteState()

    m = re.search(r"soc\D{0,10}(\d+(\.\d+)?)\s*%|soil organic carbon\D{0,15}(\d+(\.\d+)?)\s*%|carbon (?:is|of)\s*(\d+(\.\d+)?)\s*%", t)
    if m:
        val = next(g for g in m.groups() if g is not None and re.match(r"^\d+(\.\d+)?$", str(g)))
        try:
            s.soc_percent = float(val)
        except Exception:
            pass

    m = re.search(r"ph\D{0,5}(\d+(\.\d+)?)", t)
    if m:
        s.ph = float(m.group(1))

    m = re.search(r"(\d{2,5})\s*mm", t)
    if m:
        s.rainfall_mm = float(m.group(1))
    if re.search(r"rainfall.{0,15}\blow\b|\blow rainfall\b", t):
        s.rainfall_category = "low"
    elif re.search(r"rainfall.{0,15}\bmedium\b|\bmoderate rainfall\b", t):
        s.rainfall_category = "medium"
    elif re.search(r"rainfall.{0,15}\bhigh\b|\bhigh rainfall\b", t):
        s.rainfall_category = "high"

    for phrase, code in _LAND_USE_KEYWORDS.items():
        if phrase in t:
            s.land_use = code
            break

    if "semi-arid" in t or "semiarid" in t:
        s.region = "semi-arid"
    elif "arid" in t:
        s.region = "arid"
    elif "tropical" in t:
        s.region = "tropical"
    elif "temperate" in t:
        s.region = "temperate"

    m = re.search(r"fragmentation\D{0,10}(\d+(\.\d+)?)", t)
    if m:
        s.fragmentation_index = float(m.group(1))
    elif "fragment" in t and ("high" in t or "severe" in t):
        s.fragmentation_index = 0.7

    m = re.search(r"species richness\D{0,10}(\d+(\.\d+)?)", t)
    if m:
        s.species_richness_index = float(m.group(1))
    # "biodiversity is declining" carries no quantitative richness value by
    # itself; species_richness_index stays None and is caught by the
    # clarifying-question logic in missing_core_vars() / the agent loop.

    if "no-till" in t or "no till" in t or "low tillage" in t:
        s.tillage_intensity = "low"
    elif "heavy tillage" in t or "intensive tillage" in t or "conventional tillage" in t:
        s.tillage_intensity = "high"

    if "high fertilizer" in t or "heavy fertilizer" in t or "fertilizer-intensive" in t:
        s.fertilizer_use = "high"
    elif "low fertilizer" in t or "no fertilizer" in t:
        s.fertilizer_use = "low"

    if "deforestation" in t or "forest clearing" in t or "cleared forest" in t:
        s.deforestation_adjacent = True

    if "low pollinator" in t or "few pollinators" in t or "pollinators are declining" in t:
        s.pollinator_abundance = "low"

    if "dry soil" in t or "low moisture" in t:
        s.moisture = "low"

    if "rotational grazing" in t:
        s.grazing_type = "rotational"
    elif "continuous grazing" in t or "continuously grazed" in t:
        s.grazing_type = "continuous"

    return s


def parse_json_input(payload: Dict[str, Any]) -> SiteState:
    s = SiteState()
    soil = payload.get("soil", {})
    land = payload.get("land", {})
    climate = payload.get("climate", {})
    bio = payload.get("biodiversity", {})
    human = payload.get("human_impact", {})

    s.soc_percent = soil.get("organic_carbon_percent")
    s.ph = soil.get("ph")
    s.moisture = soil.get("moisture")
    s.bulk_density = soil.get("bulk_density")

    s.land_use = land.get("use_type")
    s.region = land.get("region")
    s.fragmentation_index = land.get("fragmentation_index")

    s.rainfall_mm = climate.get("annual_rainfall_mm")
    s.mean_temp_c = climate.get("mean_temp_c")
    s.aridity_index = climate.get("aridity_index")

    s.species_richness_index = bio.get("species_richness_index")
    s.habitat_diversity_index = bio.get("habitat_diversity_index")
    s.pollinator_abundance = bio.get("pollinator_abundance")

    s.tillage_intensity = human.get("tillage_intensity")
    s.fertilizer_use = human.get("fertilizer_use")
    s.deforestation_adjacent = human.get("deforestation_adjacent")
    s.grazing_type = land.get("grazing_type")

    # Bonus geo-coordinate path (spec): real lookup against the CSIRO India
    # BHI v4 raster when coordinates fall inside its coverage. Silently
    # no-ops outside India or if the raster/rasterio is unavailable.
    geo = payload.get("geo")
    if geo and "lat" in geo and "lon" in geo:
        bhi = lookup_bhi(geo["lat"], geo["lon"])
        if bhi:
            s.biodiversity_habitat_index = bhi["bhi_latest"]
            s.bhi_trend = bhi["trend_2000_2024"]

    return s


def merge_state(old: SiteState, new: SiteState) -> SiteState:
    """New non-null fields overwrite old; old fields persist if new is silent
    on them. This is the accumulation behavior of the multi-turn memory."""
    merged = SiteState(**asdict(old))
    for field_name, value in asdict(new).items():
        if value is not None:
            setattr(merged, field_name, value)
    return merged


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def classify_intent(text: str) -> str:
    t = text.lower()
    if any(k in t for k in DATA_REQUEST_TRIGGERS):
        return "data_request"
    if any(k in t for k in METRIC_EXPLAIN_TRIGGERS):
        return "metric_explanation"
    if any(k in t for k in RECOMMENDATION_TRIGGERS):
        return "recommendation"
    return "diagnosis"


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

class Retriever:
    def __init__(self):
        self.embed_fn, self.backend_name = get_embedder()
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def retrieve(self, query: str, k: int = RETRIEVAL_K):
        if self.collection.count() == 0:
            return []
        q_emb = self.embed_fn([query])
        res = self.collection.query(query_embeddings=q_emb, n_results=min(k, self.collection.count()))
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"document": doc, "metadata": meta, "distance": dist})
        return out


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_recommendations(recs: List[Recommendation], retrieved: List[Dict[str, Any]]) -> str:
    if not recs:
        return (
            "I don't have enough matched evidence-backed rules for this exact combination yet. "
            "Here is the most relevant supporting evidence I found in the knowledge base, which "
            "you can use to guide next steps:\n\n" + _format_retrieved_only(retrieved)
        )

    lines = ["## Evidence-Backed Recommendations\n"]
    for i, r in enumerate(recs, 1):
        lines.append(f"### Recommendation {i}")
        lines.append(f"- **What to do:** {r.what_to_do}")
        lines.append(f"- **Impacted Metrics:** {'; '.join(r.impacted_metrics)}")
        lines.append(f"- **Scientific Reasoning:** {r.scientific_reasoning}")
        lines.append(f"- **Evidence:** {r.evidence}")
        lines.append(f"- **Time Horizon:** {r.time_horizon}")
        lines.append(f"- **Confidence:** {r.confidence}")
        lines.append("")

    if retrieved:
        lines.append("### Supporting knowledge-base evidence (RAG retrieval)")
        for item in retrieved[:5]:
            meta = item["metadata"]
            lines.append(f"- [{meta.get('source')} {meta.get('year')} #{meta.get('source_id')}] "
                         f"{item['document']}")
    return "\n".join(lines)


def _format_retrieved_only(retrieved: List[Dict[str, Any]]) -> str:
    if not retrieved:
        return "(No matching evidence found in the knowledge base for this query.)"
    lines = []
    for item in retrieved[:5]:
        meta = item["metadata"]
        lines.append(f"- [{meta.get('source')} {meta.get('year')} #{meta.get('source_id')}] "
                     f"{item['document']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class EnvAgent:
    def __init__(self):
        self.retriever = Retriever()
        self.turns = deque(maxlen=MEMORY_WINDOW)   # (user_text, intent, state_snapshot)
        self.state = SiteState()

    def _retrieval_query(self, text: str, state: SiteState) -> str:
        known = ", ".join(f"{k}={v}" for k, v in state.known_fields().items())
        return f"{text} | known variables: {known}"

    def handle_turn(self, user_input, is_json: bool = False) -> str:
        if is_json:
            new_state = parse_json_input(user_input)
            raw_text = str(user_input)
        else:
            new_state = parse_free_text(user_input)
            raw_text = user_input

        intent = classify_intent(raw_text)
        self.state = merge_state(self.state, new_state)

        missing = missing_core_vars(self.state)
        retrieved = self.retriever.retrieve(self._retrieval_query(raw_text, self.state))

        if intent in ("diagnosis", "recommendation") and missing:
            response = (
                "To provide targeted recommendations, I need a bit more information:\n"
                + "\n".join(f"- {m}" for m in missing)
            )
        elif intent == "metric_explanation":
            response = self._explain_metric(raw_text, retrieved)
        elif intent == "data_request":
            response = self._describe_sources(retrieved)
        else:  # diagnosis / recommendation, complete enough to reason over
            recs = run_reasoning(self.state)
            response = format_recommendations(recs, retrieved)

        self.turns.append({"user_input": raw_text, "intent": intent, "state": asdict(self.state)})
        return response

    def memory_summary(self) -> List[Dict[str, Any]]:
        return list(self.turns)

    def _explain_metric(self, query: str, retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return ("I don't have a knowledge-base entry directly explaining that term yet. "
                     "Could you name the specific metric (e.g. soil organic carbon, "
                     "habitat diversity index, fragmentation index)?")
        lines = ["Here's what the knowledge base says about that:\n"]
        for item in retrieved[:3]:
            meta = item["metadata"]
            lines.append(f"- [{meta.get('source')} {meta.get('year')} #{meta.get('source_id')}] "
                         f"{item['document']}")
        lines.append("\nAsk me for recommendations any time once you'd like next steps.")
        return "\n".join(lines)

    def _describe_sources(self, retrieved: List[Dict[str, Any]]) -> str:
        if not retrieved:
            return "No knowledge-base entries matched this query yet."
        lines = ["Sources backing the current conversation's evidence:\n"]
        seen = set()
        for item in retrieved:
            meta = item["metadata"]
            key = (meta.get("source"), meta.get("source_id"))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"- {meta.get('source')} ({meta.get('year')}), reference [{meta.get('source_id')}]")
        lines.append("\nFull dataset citation list is documented in README.md.")
        return "\n".join(lines)
