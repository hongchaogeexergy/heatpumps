import json
from copy import deepcopy

def export_exerpy_json_from_ean(ean, path: str) -> dict:
    """
    ean: ExerPy ExergyAnalysis instance (created via ExergyAnalysis.from_tespy)
    Schreibt ein JSON im ExerPy-Schema und gibt das dict zurück.
    """
    # ExerPy API: ExergyAnalysis hält Rohdaten in _component_data/_connection_data (siehe Doku/Attribute).
    data = {
        "components": deepcopy(getattr(ean, "_component_data")),
        "connections": deepcopy(getattr(ean, "_connection_data")),
        "ambient_conditions": {
            "Tamb": float(ean.Tamb),
            "Tamb_unit": "K",
            "pamb": float(ean.pamb),
            "pamb_unit": "Pa",
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return data


def patch_condenser_as_heat_exchanger(exerpy_json: dict, condenser_name: str = "Condenser") -> dict:
    """
    Verschiebt GENAU EINEN Condenser (per name/label) aus 'Condenser' nach 'HeatExchanger'.
    """
    data = deepcopy(exerpy_json)
    comps = data.get("components", {})

    if "Condenser" not in comps:
        # nichts zu tun
        return data

    cond_group = comps["Condenser"]
    if condenser_name not in cond_group:
        # falls du mehrere Condenser hast oder der Name anders ist:
        # -> hier kannst du entweder den richtigen Namen setzen oder alle verschieben (siehe Funktion unten)
        return data

    hx_group = comps.setdefault("HeatExchanger", {})
    hx_group[condenser_name] = cond_group[condenser_name]

    # entfernen aus Condenser-Gruppe
    del cond_group[condenser_name]
    if len(cond_group) == 0:
        del comps["Condenser"]

    return data


def patch_all_condensers_as_heat_exchanger(exerpy_json: dict) -> dict:
    """
    Optional: verschiebt ALLE Condenser nach HeatExchanger (falls du mehrere hast).
    """
    data = deepcopy(exerpy_json)
    comps = data.get("components", {})
    if "Condenser" not in comps:
        return data

    hx_group = comps.setdefault("HeatExchanger", {})
    for name, payload in list(comps["Condenser"].items()):
        hx_group[name] = payload
        del comps["Condenser"][name]

    del comps["Condenser"]
    return data
