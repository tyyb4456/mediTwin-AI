"""
FDA / NLM API Client for Drug Safety Agent
All APIs are free, no authentication required.

APIs used:
  - NLM RxNav Interaction API: https://rxnav.nlm.nih.gov/InteractionAPIs.html
  - NLM RxNorm API:            https://rxnav.nlm.nih.gov/RxNormAPIs.html
  - FDA OpenFDA Drug Labels:   https://open.fda.gov/apis/drug/label/
"""
import httpx
import asyncio
from typing import Optional


RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug"

# HTTP timeout — external APIs can be slow
TIMEOUT = httpx.Timeout(15.0, connect=5.0)


# ── RxNorm: drug name → RxCUI code ───────────────────────────────────────────

async def get_rxcui(drug_name: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Look up the RxNorm concept unique identifier (RxCUI) for a drug name.
    RxCUI is needed for the interaction API.
    Returns None if not found.
    """
    # Strip dose info — "Amoxicillin 500mg" → "Amoxicillin"
    base_name = drug_name.split()[0].strip()

    try:
        url = f"{RXNAV_BASE}/rxcui.json"
        resp = await client.get(url, params={"name": base_name}, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

        rxcui = (
            data.get("idGroup", {})
            .get("rxnormId", [None])[0]
        )
        return rxcui
    except Exception as e:
        print(f"  RxCUI lookup failed for '{drug_name}': {e}")
        return None


async def get_rxcuis_batch(drug_names: list[str]) -> dict[str, Optional[str]]:
    """
    Resolve a list of drug names to RxCUI codes concurrently.
    Returns {drug_name: rxcui_or_None}
    """
    async with httpx.AsyncClient() as client:
        tasks = [get_rxcui(name, client) for name in drug_names]
        rxcuis = await asyncio.gather(*tasks)
    return dict(zip(drug_names, rxcuis))


# ── RxNav Interaction API ─────────────────────────────────────────────────────

async def check_rxnav_interactions(rxcui_list: list[str]) -> list[dict]:
    """
    Check drug-drug interactions for a list of RxCUI codes using NLM RxNav.
    Returns a list of interaction dicts.
    """
    if len(rxcui_list) < 2:
        return []  # Need at least 2 drugs to check

    rxcuis_str = "+".join(rxcui_list)

    try:
        async with httpx.AsyncClient() as client:
            url = f"{RXNAV_BASE}/interaction/list.json"
            resp = await client.get(
                url,
                params={"rxcuis": rxcuis_str},
                timeout=TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()

        interactions = []
        full_iaction = data.get("fullInteractionTypeGroup", [])

        for group in full_iaction:
            source = group.get("sourceName", "NLM RxNav")
            for itype in group.get("fullInteractionType", []):
                comment = itype.get("comment", "")
                for pair in itype.get("interactionPair", []):
                    severity = pair.get("severity", "N/A").upper()
                    description = pair.get("description", "")

                    # Extract drug names from the pair
                    concepts = pair.get("interactionConcept", [])
                    drug_names_in_pair = [
                        c.get("minConceptItem", {}).get("name", "Unknown")
                        for c in concepts
                    ]

                    interactions.append({
                        "drug_a": drug_names_in_pair[0] if len(drug_names_in_pair) > 0 else "Unknown",
                        "drug_b": drug_names_in_pair[1] if len(drug_names_in_pair) > 1 else "Unknown",
                        "severity": severity,
                        "description": description,
                        "comment": comment,
                        "source": source,
                        "clinical_recommendation": _severity_to_recommendation(severity, drug_names_in_pair),
                    })

        return interactions

    except Exception as e:
        print(f"  RxNav interaction check failed: {e}")
        return []


def _severity_to_recommendation(severity: str, drugs: list[str]) -> str:
    """Convert severity string to clinical recommendation."""
    pair = " + ".join(drugs) if drugs else "these drugs"
    sev = severity.upper()

    if sev in ("HIGH", "CRITICAL"):
        return f"AVOID combination of {pair} if possible. Consult pharmacist before prescribing."
    elif sev == "MODERATE":
        return f"Monitor patient closely when using {pair} together. Consider alternative if available."
    elif sev == "LOW":
        return f"Minor interaction between {pair}. Monitor for unexpected effects."
    else:
        return f"Interaction noted between {pair}. Clinical review recommended."


# ── OpenFDA: drug label warnings ─────────────────────────────────────────────

async def get_fda_warnings(drug_name: str) -> list[str]:
    """
    Fetch key warning text from FDA drug label (boxed_warning field).
    Returns list of warning strings (may be empty if none found).
    """
    base_name = drug_name.split()[0].strip()

    try:
        async with httpx.AsyncClient() as client:
            url = f"{OPENFDA_BASE}/label.json"
            resp = await client.get(
                url,
                params={
                    "search": f"openfda.generic_name:{base_name}",
                    "limit": 1
                },
                timeout=TIMEOUT
            )

            if resp.status_code == 404:
                return []  # Drug not found in FDA DB — not an error

            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            if not results:
                return []

            label = results[0]
            warnings = []

            # Boxed warnings (black box warnings — most serious)
            if "boxed_warning" in label:
                for w in label["boxed_warning"]:
                    warnings.append(f"[BLACK BOX] {w[:300]}")

            # Regular warnings (truncated)
            if "warnings" in label:
                for w in label["warnings"][:2]:
                    warnings.append(w[:200])

            return warnings

    except Exception as e:
        print(f"  FDA label lookup failed for '{drug_name}': {e}")
        return []