"""Assign actual CC0 geography concepts to new synthetic household contexts."""
from __future__ import annotations
import copy

def assigned_geography_context(vocabulary: dict, geography_qid: str) -> dict:
    """Resolve one explicitly requested non-person concept, without sampling."""
    entities = {row["qid"]: row for row in vocabulary.get("entities", [])}
    relations = vocabulary.get("relations", [])
    parents = [relation["subject_qid"] for relation in relations
               if relation.get("property_qid") == "P150"
               and relation.get("object_qid") == geography_qid]
    if "Q148" not in parents or geography_qid not in entities:
        raise ValueError("The requested concept is not in the supplied administrative vocabulary")
    entity = entities[geography_qid]
    type_ids = [relation["object_qid"] for relation in relations
                if relation.get("property_qid") == "P31"
                and relation.get("subject_qid") == geography_qid]
    if not entity.get("label_zh") or not type_ids:
        raise ValueError("The concept needs its actual Chinese label and administrative type")

    def label(qid: str) -> dict:
        known = entities.get(qid)
        if known is None or not known.get("label_zh"):
            raise ValueError("A relation points outside the supplied labeled vocabulary")
        return {key: known.get(key, "") for key in ("qid", "label_zh", "label_en", "kind")}

    return {
        **label(geography_qid),
        "assignment_type": "assigned_synthetic_context",
        "knowledge_role": "administrative_concept",
        "types": [label(qid) for qid in type_ids],
        "parents": [{**label(qid), "property_qid": "P150"} for qid in parents],
    }


def attach_assigned_geography(persona: dict, vocabulary: dict, *,
                             geography_qid: str, id_field: str = "household_id") -> dict:
    """Return a complete independent copy with an explicitly assigned context.

    The caller supplies the ordinary ID field and concept. No location,
    occupation, preference or probability is inferred from the input fields.
    """
    ordinary_id = persona.get(id_field)
    if not isinstance(ordinary_id, (str, int)) or isinstance(ordinary_id, bool) or str(ordinary_id) == "":
        raise ValueError("A nonempty ordinary persona ID is required")
    if "knowledge_context" in persona:
        raise ValueError("Existing knowledge context is not overwritten")
    result = copy.deepcopy(persona)
    result["knowledge_context"] = {
        "geography": assigned_geography_context(vocabulary, geography_qid),
        "occupation": None,
    }
    return result


