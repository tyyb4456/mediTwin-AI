"""
FDA / NLM API Client for Drug Safety Agent — v2.0
All APIs are free, no authentication required.

APIs used:
  - NLM RxNav Interaction API: https://rxnav.nlm.nih.gov/InteractionAPIs.html
  - NLM RxNorm API:            https://rxnav.nlm.nih.gov/RxNormAPIs.html
  - FDA OpenFDA Drug Labels:   https://open.fda.gov/apis/drug/label/
  - NLM RxClass API:           https://rxnav.nlm.nih.gov/RxClassAPIs.html
"""
import httpx
import asyncio
from typing import Optional

RXNAV_BASE   = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug"

TIMEOUT = httpx.Timeout(15.0, connect=5.0)

import logging
logger = logging.getLogger("fda_client")


# ── RxNorm: drug name → RxCUI code ───────────────────────────────────────────

async def get_rxcui(drug_name: str, client: httpx.AsyncClient) -> Optional[str]:
    """
    Look up the RxNorm concept unique identifier (RxCUI) for a drug name.
    Strips dose info before lookup: 'Amoxicillin 500mg' → 'Amoxicillin'
    """
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
        logger.error(f" ✘  RxCUI lookup failed for '{drug_name}': {e}")
        return None


async def get_rxcuis_batch(drug_names: list[str]) -> dict[str, Optional[str]]:
    """Resolve a list of drug names to RxCUI codes concurrently."""
    async with httpx.AsyncClient() as client:
        tasks = [get_rxcui(name, client) for name in drug_names]
        rxcuis = await asyncio.gather(*tasks)
    return dict(zip(drug_names, rxcuis))


# ── RxNav Interaction API ─────────────────────────────────────────────────────

async def check_rxnav_interactions(rxcui_list: list[str]) -> list[dict]:
    """
    RxNav interaction API was discontinued Jan 2024.
    Fallback: use OpenFDA label data to detect known interactions via warnings text.
    Drug names are passed via rxcui_list but we use the names from the caller's rxcui_map.
    """
    return []  # Placeholder — see check_interactions_via_openfda below


async def check_drug_interactions_by_name(drug_names: list[str]) -> list[dict]:
    """
    Check interactions using OpenFDA drug label warnings.
    Searches each drug's label for mentions of other drugs in the list.
    Free, no auth, works today.
    """
    if len(drug_names) < 2:
        return []

    labels = {}
    async with httpx.AsyncClient() as client:
        tasks = [_fetch_fda_label(name, client) for name in drug_names]
        results = await asyncio.gather(*tasks)
    
    for name, label in zip(drug_names, results):
        if label:
            labels[name] = label

    interactions = []
    seen = set()
    drug_names_lower = [d.lower().split()[0] for d in drug_names]

    for drug_a, label in labels.items():
        # Check drug_interactions and warnings sections for mentions of other drugs
        interaction_text = " ".join(
            label.get("drug_interactions", []) +
            label.get("warnings", []) +
            label.get("warnings_and_cautions", [])
        ).lower()

        for drug_b in drug_names:
            if drug_b == drug_a:
                continue
            key = frozenset([drug_a.lower(), drug_b.lower()])
            if key in seen:
                continue
            base_b = drug_b.lower().split()[0]
            if base_b in interaction_text:
                seen.add(key)
                interactions.append({
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "severity": "MODERATE",
                    "description": f"{drug_b} is mentioned in {drug_a} label interactions/warnings section.",
                    "source": "FDA Drug Label",
                    "clinical_recommendation": _severity_to_recommendation("MODERATE", [drug_a, drug_b]),
                })

    return interactions


async def _fetch_fda_label(drug_name: str, client: httpx.AsyncClient) -> Optional[dict]:
    """Fetch full FDA label for a drug."""
    base_name = drug_name.split()[0].strip()
    try:
        resp = await client.get(
            f"{OPENFDA_BASE}/label.json",
            params={"search": f"openfda.generic_name:{base_name}", "limit": 1},
            timeout=TIMEOUT,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        results = resp.json().get("results", [])
        return results[0] if results else None
    except Exception:
        return None


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


# ── OpenFDA: drug label warnings (batch) ─────────────────────────────────────

async def get_fda_warnings_batch(drug_names: list[str]) -> dict[str, list[str]]:
    """
    Fetch FDA black box and key warnings for a list of drugs concurrently.
    Returns {drug_name: [warning_strings]}
    """
    async with httpx.AsyncClient() as client:
        tasks = [_get_fda_warnings_single(name, client) for name in drug_names]
        results = await asyncio.gather(*tasks)
    return dict(zip(drug_names, results))


async def _get_fda_warnings_single(
    drug_name: str, client: httpx.AsyncClient
) -> list[str]:
    """Fetch FDA label warnings for a single drug."""
    base_name = drug_name.split()[0].strip()
    try:
        resp = await client.get(
            f"{OPENFDA_BASE}/label.json",
            params={"search": f"openfda.generic_name:{base_name}", "limit": 1},
            timeout=TIMEOUT,
        )
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return []

        label = results[0]
        warnings = []

        # Black box (most serious)
        for w in label.get("boxed_warning", []):
            warnings.append(f"[BLACK BOX] {w[:400]}")

        # Standard warnings (first 2)
        for w in label.get("warnings", [])[:2]:
            warnings.append(w[:300])

        # Contraindications from label
        for w in label.get("contraindications", [])[:1]:
            warnings.append(f"[CONTRAINDICATION] {w[:300]}")

        return warnings

    except Exception as e:
        logger.error(f"  ✘   FDA warning lookup failed for '{drug_name}': {e}")
        return []


# ── NLM RxClass: drug class lookup (optional enrichment) ─────────────────────

async def get_drug_class(rxcui: str) -> Optional[str]:
    """
    Look up the drug class for an RxCUI via RxClass API.
    Returns the first ATC or MESHPA class found, or None.
    """
    if not rxcui:
        return None
    try:
        async with httpx.AsyncClient() as client:
            url = f"{RXNAV_BASE}/rxclass/class/byRxcui.json"
            resp = await client.get(
                url,
                params={"rxcui": rxcui, "relaSource": "ATC1-4"},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()

        classes = (
            data.get("rxclassDrugInfoList", {})
                .get("rxclassDrugInfo", [])
        )
        if classes:
            return classes[0].get("rxclassMinConceptItem", {}).get("className")
        return None

    except Exception:
        return None