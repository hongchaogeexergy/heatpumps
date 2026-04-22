# src/heatpumps/economics/exerpy_costing.py
import math
import os
import logging
from exerpy import ExergyAnalysis, ExergoeconomicAnalysis
import numpy as np
from collections.abc import MutableMapping
import json
import tempfile
from copy import deepcopy
from collections.abc import MutableMapping


import numpy as np


def _scalar(x):
    """Return scalar float if x is list/np array, else return x."""
    while isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return np.nan
        x = x[-1]  # take last element for time-series-like inputs
    return x


def _conn_temperature_C(hp, label, fallback=None):
    """Return connection temperature in degC from solved state or fallback params."""
    conn = getattr(hp, "conns", {}).get(label)
    if conn is not None:
        try:
            val = float(conn.T.val)
            if not np.isnan(val):
                return val
        except Exception:
            pass

    if label in ("B1", "B2", "B3"):
        try:
            if label == "B3":
                return hp.params.get("B2", {}).get("T", fallback)
            return hp.params.get(label, {}).get("T", fallback)
        except Exception:
            pass
    if label in ("C1", "C2", "C3"):
        try:
            return hp.params.get(label, {}).get("T", fallback)
        except Exception:
            pass
    return fallback


def determine_exergy_boundaries(hp, source_tol_K=1.0):
    """
    Determine fuel, product and loss boundaries from source-side temperature level.

    Scenarios
    ---------
    A. Environmental source:
       B1 close to ambient -> source side is treated as loss (B3 -> B1).
    B. Waste heat:
       B1 above ambient -> source inlet is fuel and source outlet is loss.
    C. Waste heat further usage:
       User explicitly enables further usage of return stream and B3 stays at or
       above ambient -> source inlet is fuel and no source-side loss is assigned.
    """
    conns = getattr(hp, "conns", {}) or {}
    ambient_T = float(hp.params["ambient"]["T"])
    env_out = "B3" if "B3" in conns else ("B2" if "B2" in conns else None)
    consider_further_usage = bool(
        hp.params.get("setup", {}).get("waste_heat_further_usage", False)
    )

    fuel = {"inputs": ["E0"] if "E0" in conns else [], "outputs": []}
    product = {
        "inputs": [x for x in ("C3",) if x in conns],
        "outputs": [x for x in ("C1",) if x in conns],
    }
    loss = {"inputs": [], "outputs": []}

    if "B1" not in conns or env_out is None:
        return {"fuel": fuel, "product": product, "loss": loss, "scenario": "fallback"}

    T_b1 = _conn_temperature_C(hp, "B1", fallback=ambient_T)
    T_bout = _conn_temperature_C(hp, env_out, fallback=ambient_T)
    return_below_ambient = False

    if T_b1 is None or T_bout is None:
        loss = {"inputs": [env_out], "outputs": ["B1"]}
        return {
            "fuel": fuel,
            "product": product,
            "loss": loss,
            "scenario": "fallback",
            "return_below_ambient": False,
        }

    if T_b1 <= ambient_T + source_tol_K:
        loss = {"inputs": [env_out], "outputs": ["B1"]}
        scenario = "case_a_environmental_source"
    elif consider_further_usage and T_bout >= ambient_T - source_tol_K:
        fuel["inputs"].append("B1")
        scenario = "case_c_waste_heat_further_usage"
    else:
        fuel["inputs"].append("B1")
        loss = {"inputs": [env_out], "outputs": []}
        scenario = "case_b_waste_heat"
        return_below_ambient = T_bout < ambient_T - source_tol_K

    # Intercooler cooling loop (HeatPumpIC / cascade IC): treat discharged cooling
    # water as loss boundary independent of source-side classification.
    if getattr(hp, "__class__", None) and hp.__class__.__name__ in (
        "HeatPumpIC", "HeatPumpICTrans", "HeatPumpCascadeIC", "HeatPumpCascadeICTrans"
    ):
        for lbl in ("D2",):
            if lbl in conns and lbl not in loss["inputs"]:
                loss["inputs"].append(lbl)

    return {
        "fuel": fuel,
        "product": product,
        "loss": loss,
        "scenario": scenario,
        "return_below_ambient": return_below_ambient,
        "consider_further_usage": consider_further_usage,
    }

def _scalarize_portdict(portdict):
    """
    ExerPy ports are dict-like: {"T":..., "p":..., "E_M":..., ...}
    Make common numeric fields scalar if they are lists.
    """
    if not isinstance(portdict, dict):
        return
    for k in ("T", "p", "h", "s", "E", "E_M", "E_P", "E_F", "m"):
        if k in portdict:
            portdict[k] = _scalar(portdict[k])

def _scalarize_json_value(val):
    if isinstance(val, dict):
        return {k: _scalarize_json_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, np.ndarray)):
        if len(val) == 0:
            return np.nan
        return _scalarize_json_value(val[-1])
    return val

def _scalarize_exerpy_json(data: dict) -> None:
    """In-place: collapse list/array values to scalars in ExerPy JSON export."""
    if not isinstance(data, dict):
        return
    conns = data.get("connections", {})
    if isinstance(conns, dict):
        for key, conn in conns.items():
            if isinstance(conn, dict):
                conns[key] = _scalarize_json_value(conn)
    comps = data.get("components", {})
    if isinstance(comps, dict):
        for group_key, group in comps.items():
            if not isinstance(group, dict):
                continue
            for comp_key, comp in group.items():
                if isinstance(comp, dict):
                    comps[group_key][comp_key] = _scalarize_json_value(comp)


def _is_mapping(obj):
    return isinstance(obj, MutableMapping) or (
        hasattr(obj, "keys") and hasattr(obj, "__getitem__") and hasattr(obj, "__setitem__")
    )

def _is_equations_candidate(obj):
    if isinstance(obj, dict):
        return True
    if isinstance(obj, list):
        if len(obj) == 0:
            return True
        sample = obj[0]
        if isinstance(sample, dict) or sample is None or isinstance(sample, (int, float, str)):
            return True
    return False


def _scalarize_port_container(obj):
    if _is_mapping(obj):
        _scalarize_portdict(obj)
        for k in list(getattr(obj, "keys", lambda: [])()):
            try:
                _scalarize_port_container(obj[k])
            except Exception:
                continue
        return
    if isinstance(obj, (list, tuple, np.ndarray)):
        for item in obj:
            _scalarize_port_container(item)


def _find_non_scalar(obj, path):
    issues = []
    if isinstance(obj, (list, tuple, np.ndarray)):
        sample = None
        try:
            sample = obj[-1] if len(obj) else None
        except Exception:
            sample = None
        issues.append((path, type(obj).__name__, repr(sample)))
        # still traverse nested structures if possible
        for i, item in enumerate(obj):
            issues.extend(_find_non_scalar(item, f"{path}[{i}]"))
        return issues
    if _is_mapping(obj):
        for k in list(getattr(obj, "keys", lambda: [])()):
            try:
                issues.extend(_find_non_scalar(obj[k], f"{path}.{k}"))
            except Exception:
                continue
    return issues

def _iter_portdicts(ports):
    """
    ExerPy ports can be dict-like (with .values) or list/tuple.
    Return a flat list of portdicts.
    """
    if ports is None:
        return []
    # dict-like: has .values()
    if hasattr(ports, "values"):
        try:
            return list(ports.values())
        except Exception:
            return []
    # list/tuple
    if isinstance(ports, (list, tuple, np.ndarray)):
        return list(ports)
    return []


def _scalarize_component_ports(comp):
    for attr in ("inl", "outl"):
        ports = getattr(comp, attr, None)
        _scalarize_port_container(ports)


def _scalarize_ean_ports(ean):
    for comp in (getattr(ean, "components", {}) or {}).values():
        _scalarize_component_ports(comp)


def _ensure_equations_list(equations, A):
    """Make sure `equations` is list-like and long enough for row indexing."""
    n = getattr(A, "shape", (0, 0))[0] if A is not None else 0
    if equations is None or isinstance(equations, (float, int)):
        return [None] * n
    if not isinstance(equations, list):
        try:
            equations = list(equations)
        except Exception:
            equations = [None] * n
    if len(equations) < n:
        equations = equations + [None] * (n - len(equations))
    return equations

def _grow_rows(A, b, equations, add_rows):
    """Grow A and b by add_rows rows, keep equations aligned."""
    ncols = A.shape[1]
    A2 = np.vstack([A, np.zeros((add_rows, ncols))])
    b2 = np.concatenate([b, np.zeros(add_rows)])
    equations = _ensure_equations_list(equations, A2)
    return A2, b2, equations

def patch_aux_eqs_grow_rows(ean, only_class_name=None, default_add=2):
    """
    Wrap aux_eqs so matrix rows grow if a component needs more rows than reserved.

    Why default_add=2?
      - Many aux_eqs add 1 row, but some add 2 in certain modes.
      - Growing by 2 reduces repeated grow calls while still minimal.
    """
    for name, comp in (getattr(ean, "components", {}) or {}).items():
        if not hasattr(comp, "aux_eqs") and not hasattr(comp, "dis_eqs"):
            continue
        if only_class_name and comp.__class__.__name__.lower() != only_class_name.lower():
            continue

        orig = comp.aux_eqs

        def _wrapped(A, b, counter, *args,
                     __orig=orig, __name=name, __comp=comp, __cls=comp.__class__.__name__, **kwargs):
            if __cls.lower() == "simpleheatexchanger":
                print(f"[PATCH] Skipping aux_eqs for SimpleHeatExchanger '{__name}' (not implemented)")
                return A, b, counter, _ensure_equations_list(kwargs.get("equations"), A)

            args = list(args)
            eq_idx = None
            equations = None

            if "equations" in kwargs:
                equations = _ensure_equations_list(kwargs["equations"], A)
                kwargs["equations"] = equations
            elif len(args) >= 2 and _is_equations_candidate(args[1]):
                eq_idx = 1
            else:
                for i, arg in enumerate(args):
                    if _is_equations_candidate(arg):
                        eq_idx = i
                        break

            if equations is None:
                if eq_idx is not None:
                    equations = _ensure_equations_list(args[eq_idx], A)
                    args[eq_idx] = equations
                else:
                    equations = _ensure_equations_list(None, A)

            # if no free row left, skip aux eqs to keep matrix square
            if counter >= A.shape[0]:
                print(
                    f"[PATCH] Skipping aux_eqs for {__cls} '{__name}': "
                    f"counter={counter} >= rows={A.shape[0]} (avoids overdetermined system)"
                )
                return A, b, counter, equations

            # --- scalarize inlet/outlet port values (fix list-vs-float comparisons) ---
            try:
                _scalarize_component_ports(__comp)
            except Exception:
                pass
            try:
                issues = []
                issues.extend(_find_non_scalar(getattr(__comp, "inl", None), "inl"))
                issues.extend(_find_non_scalar(getattr(__comp, "outl", None), "outl"))
                if issues:
                    print(f"[SCALAR-DBG] {__cls} '{__name}' non-scalar values:")
                    for p, t, s in issues:
                        print(f"  - {p}: {t} -> {s}")
            except Exception:
                pass
            try:
                if __cls.lower() == "compressor":
                    inl = getattr(__comp, "inl", None)
                    outl = getattr(__comp, "outl", None)
                    inl0 = inl.get(0) if hasattr(inl, "get") else None
                    outl0 = outl.get(0) if hasattr(outl, "get") else None
                    t_in = inl0.get("T") if isinstance(inl0, dict) else None
                    t_out = outl0.get("T") if isinstance(outl0, dict) else None
                    print(
                        "[SCALAR-DBG] Compressor inl/outl types:",
                        type(inl).__name__,
                        type(outl).__name__,
                        "inl0:",
                        type(inl0).__name__,
                        "outl0:",
                        type(outl0).__name__,
                    )
                    print(
                        "[SCALAR-DBG] Compressor T values:",
                        "inl0.T=",
                        t_in,
                        f"type={type(t_in).__name__}",
                        "outl0.T=",
                        t_out,
                        f"type={type(t_out).__name__}",
                    )
            except Exception:
                pass



            out = __orig(A, b, counter, *args, **kwargs)

            # If component adds too many equations, skip them to keep system square.
            try:
                A2, b2, counter2, eq2 = out
                if counter2 > A2.shape[0]:
                    print(
                        f"[PATCH] Skipping extra aux_eqs for {__cls} '{__name}': "
                        f"counter2={counter2} > rows={A2.shape[0]}"
                    )
                    return A, b, counter, equations
            except Exception:
                pass

            return out

        if hasattr(comp, "aux_eqs"):
            comp.aux_eqs = _wrapped
            print(f"[PATCH] aux_eqs GROW_ROWS enabled for {comp.__class__.__name__} '{name}'")

        if hasattr(comp, "dis_eqs"):
            orig_dis = comp.dis_eqs

            def _wrapped_dis(A, b, counter, *args,
                             __orig=orig_dis, __name=name, __comp=comp, __cls=comp.__class__.__name__, **kwargs):
                args = list(args)
                eq_idx = None
                equations = None

                if "equations" in kwargs:
                    equations = _ensure_equations_list(kwargs["equations"], A)
                    kwargs["equations"] = equations
                elif len(args) >= 2 and _is_equations_candidate(args[1]):
                    eq_idx = 1
                else:
                    for i, arg in enumerate(args):
                        if _is_equations_candidate(arg):
                            eq_idx = i
                            break

                if equations is None:
                    if eq_idx is not None:
                        equations = _ensure_equations_list(args[eq_idx], A)
                        args[eq_idx] = equations
                    else:
                        equations = _ensure_equations_list(None, A)

                if counter >= A.shape[0]:
                    print(
                        f"[PATCH] Skipping dis_eqs for {__cls} '{__name}': "
                        f"counter={counter} >= rows={A.shape[0]} (avoids overdetermined system)"
                    )
                    return A, b, counter, equations

                try:
                    _scalarize_component_ports(__comp)
                except Exception:
                    pass

                out = __orig(A, b, counter, *args, **kwargs)
                try:
                    A2, b2, counter2, eq2 = out
                    if counter2 > A2.shape[0]:
                        print(
                            f"[PATCH] Skipping extra dis_eqs for {__cls} '{__name}': "
                            f"counter2={counter2} > rows={A2.shape[0]}"
                        )
                        return A, b, counter, equations
                except Exception:
                    pass

                return out

            comp.dis_eqs = _wrapped_dis
            print(f"[PATCH] dis_eqs GROW_ROWS enabled for {comp.__class__.__name__} '{name}'")


def _exerpy_export_to_json_dict(ean) -> dict:
    """
    Exportiert das ExerPy-Objekt (aus from_tespy) in ein JSON-kompatibles dict.
    Wir nutzen bewusst private Felder, weil ExerPy diese Struktur intern hält.
    """
    data = {}

    # ExerPy hält parsed Daten typischerweise so:
    if hasattr(ean, "_component_data"):
        data["components"] = deepcopy(ean._component_data)
    if hasattr(ean, "_connection_data"):
        data["connections"] = deepcopy(ean._connection_data)

    if "components" not in data or "connections" not in data:
        raise RuntimeError(
            "ExerPy Export fehlgeschlagen: _component_data/_connection_data nicht gefunden. "
            "Bitte poste kurz `dir(ean)` und `ean.__dict__.keys()`."
        )

    _scalarize_exerpy_json(data)

    # optional
    try:
        data["ambient_conditions"] = {"Tamb": float(ean.Tamb), "pamb": float(ean.pamb)}
    except Exception:
        pass

    return data


def _exerpy_patch_condenser_group_to_heatexchanger(data: dict) -> dict:
    """
    Move ALL components in components['Condenser'] into components['HeatExchanger'].

    """
    patched = deepcopy(data)
    comps = patched.get("components", {})

    cond_group = comps.get("Condenser")
    if not isinstance(cond_group, dict) or len(cond_group) == 0:
        return patched

    hx_group = comps.setdefault("HeatExchanger", {})

    # Move everything from Condenser -> HeatExchanger
    for lbl, payload in list(cond_group.items()):
        hx_group[lbl] = payload
        del cond_group[lbl]

    # Cleanup empty group
    if len(cond_group) == 0:
        del comps["Condenser"]

    return patched



def _make_exerpy_ean_with_condenser_as_hx(*, hp, Tamb_K, pamb_Pa, condenser_label="Condenser", split_physical_exergy=True):
    """
    TESPy bleibt unverändert, aber ExerPy bekommt eine gepatchte JSON-Version,
    in der 'Condenser' als 'HeatExchanger' klassifiziert wird.
    """
    ean_raw = ExergyAnalysis.from_tespy(
        hp.nw,
        Tamb_K,
        pamb_Pa,
        split_physical_exergy=split_physical_exergy
    )

    raw_dict = _exerpy_export_to_json_dict(ean_raw)
    patched_dict = _exerpy_patch_condenser_group_to_heatexchanger(raw_dict)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(patched_dict, f, indent=2)
        patched_path = f.name

    ean = ExergyAnalysis.from_json(patched_path, split_physical_exergy=split_physical_exergy)
    return ean

def debug_dump_component_types(ean):
    print("\n[TYPE-DBG] ean.components instantiated types:")
    for name, comp in (getattr(ean, "components", {}) or {}).items():
        print(
            f"  - {name:30s} | class={comp.__class__.__name__:20s} "
            f"| module={comp.__class__.__module__}"
        )

# IC-only manual skip for PowerBus outputs.
# NOTE: PowerBus equations are between E0 (input) and each output (E1–E4).
# To test the "E0 ↔ E1" equation, set this to "E1".
IC_SKIP_POWERBUS_OUTPUT = None  # set to "E1"/"E2"/"E3"/"E4" or comma-separated list

def debug_dump_component_flags(ean):
    print("\n[FLAG-DBG] ean.components flags:")
    for name, comp in (getattr(ean, "components", {}) or {}).items():
        cls = comp.__class__.__name__
        is_diss = getattr(comp, "is_dissipative", False)
        diss_attr = getattr(comp, "dissipative", None)
        has_aux = callable(getattr(comp, "aux_eqs", None))
        has_dis = callable(getattr(comp, "dis_eqs", None))
        print(
            f"  - {name:30s} | class={cls:20s} | "
            f"is_dissipative={is_diss} | dissipative_attr={diss_attr} | "
            f"aux_eqs={has_aux} | dis_eqs={has_dis}"
        )

def debug_dump_cost_matrix_equations(exa):
    """
    Print every equation row in the cost matrix.
    PowerBus aux equations are printed explicitly as (1/E_in)*C_in - (1/E_out)*C_out = 0.
    """
    A = getattr(exa, "_A", None)
    b = getattr(exa, "_b", None)
    if A is None or b is None:
        print("[EQ-DBG] No matrix available to dump.")
        return

    vars_map = getattr(exa, "variables", {}) or {}
    eq_meta = getattr(exa, "equations", {}) or {}
    conns = getattr(exa, "connections", {}) or {}

    print("\n[EQ-DBG] Full cost-matrix equations:")
    for i in range(A.shape[0]):
        row = A[i]
        meta = eq_meta.get(i, {})
        kind = meta.get("kind")
        objects = meta.get("objects") or meta.get("object") or []

        # Special formatting for PowerBus aux equations
        if kind == "aux_power_eq" and len(objects) >= 2:
            ref = objects[0]
            out = objects[1]
            ref_E = (conns.get(ref) or {}).get("E", 0)
            out_E = (conns.get(out) or {}).get("E", 0)
            ref_term = f"(1/E_{ref})*C_{ref}" if ref_E not in (0, None) else f"1*C_{ref}"
            out_term = f"(1/E_{out})*C_{out}" if out_E not in (0, None) else f"1*C_{out}"
            rhs = b[i]
            print(f"[EQ-DBG] eq[{i:02d}] {kind}: {ref_term} - {out_term} = {rhs:.6g}")
            continue

        # Generic formatting for other equation rows
        terms = []
        for col, coef in enumerate(row):
            if coef == 0 or abs(coef) < 1e-12:
                continue
            var_name = vars_map.get(str(col), f"x{col}")
            terms.append((coef, var_name))

        if not terms:
            print(f"[EQ-DBG] eq[{i:02d}] {kind}: <empty> = {b[i]:.6g}")
            continue

        # Build expression string
        expr_parts = []
        for coef, var in terms:
            sign = "+" if coef >= 0 else "-"
            mag = abs(coef)
            if abs(mag - 1.0) < 1e-12:
                term = f"{var}"
            else:
                term = f"({mag:.6g})*{var}"
            expr_parts.append((sign, term))

        # First term keeps its sign if negative
        first_sign, first_term = expr_parts[0]
        expr = (first_term if first_sign == "+" else f"-{first_term}")
        for sign, term in expr_parts[1:]:
            expr += f" {sign} {term}"

        rhs = b[i]
        obj_txt = f" objects={objects}" if objects else ""
        print(f"[EQ-DBG] eq[{i:02d}] {kind}:{obj_txt} {expr} = {rhs:.6g}")

def patch_powerbus_skip_eqs(ean, skip_last_outputs=0, skip_output_labels=None):
    """
    Patch PowerBus.aux_eqs to skip specific output equations.
    This is a targeted workaround for IC to avoid matrix overflow or test singularity.
    """
    skip_output_labels = skip_output_labels or []
    for name, comp in (getattr(ean, "components", {}) or {}).items():
        if comp.__class__.__name__ != "PowerBus":
            continue
        if getattr(comp, "_codex_powerbus_patch", False):
            continue

        def _wrapped(A, b, counter, T0, equations, chemical_exergy_enabled, __comp=comp, __name=name):
            # Mirror original PowerBus aux_eqs but skip last outputs if requested.
            if len(__comp.inl) >= 1 and len(__comp.outl) <= 1:
                logging.info(f"PowerBus {__name} has only one output, no auxiliary equations added.")
                return A, b, counter, equations

            if len(__comp.inl) == 1 and len(__comp.outl) > 1:
                outs = list(__comp.outl.values())
                skipped = []
                if skip_output_labels:
                    outs_kept = []
                    for out in outs:
                        if out.get("name") in skip_output_labels:
                            skipped.append(out)
                        else:
                            outs_kept.append(out)
                    outs = outs_kept
                elif skip_last_outputs > 0 and len(outs) > 0:
                    skipped = outs[-skip_last_outputs:]
                    outs = outs[:-skip_last_outputs]
                if skipped:
                    __comp._codex_powerbus_skip_names = [o.get("name") for o in skipped]
                    logging.warning(
                        f"[IC-WORKAROUND] PowerBus {__name}: skipping {len(skipped)} aux equation(s): "
                        f"{', '.join(__comp._codex_powerbus_skip_names)}"
                    )
                for out in outs:
                    A[counter, __comp.inl[0]["CostVar_index"]["exergy"]] = (
                        (1 / __comp.inl[0]["E"]) if __comp.inl[0]["E"] != 0 else 1
                    )
                    A[counter, out["CostVar_index"]["exergy"]] = (-1 / out["E"]) if out["E"] != 0 else -1
                    equations[counter] = {
                        "kind": "aux_power_eq",
                        "objects": [__name, __comp.inl[0]["name"], out["name"]],
                        "property": "c_TOT",
                    }
                    b[counter] = 0
                    counter += 1
                return A, b, counter, equations

            logging.error(f"PowerBus {__name} has multiple inputs and outputs, which has not been implemented yet.")
            return A, b, counter, equations

        comp.aux_eqs = _wrapped
        comp._codex_powerbus_patch = True
        print(
            f"[PATCH] PowerBus aux_eqs skip_last_outputs={skip_last_outputs} "
            f"skip_output_labels={skip_output_labels} enabled for '{name}'"
        )


# src/heatpumps/economics/exerpy_costing.py
def debug_exerpy_equations(ean):
    print("\n[DEBUG] ExerPy component equation inventory:")
    for name, comp in (getattr(ean, "components", {}) or {}).items():
        eq = getattr(comp, "equations", None)
        n_eq = None
        try:
            n_eq = len(eq) if eq is not None else None
        except Exception:
            n_eq = "len() failed"

        hints = {}
        for attr in ("n_eq", "n_eqs", "num_eq", "num_eqs", "N_eq", "N_eqs"):
            if hasattr(comp, attr):
                hints[attr] = getattr(comp, attr)

        print(f" - {name:30s} | type={comp.__class__.__name__:20s} | equations_len={n_eq} | hints={hints}")



# ---------- small utilities ---------------------------------------------------
def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default

def eur_per_GJ_from_cent_per_kWh(cent_per_kWh: float) -> float:
    """Convert electricity price from cent/kWh → €/GJ."""
    # 1 kWh = 3.6 MJ, 1 GJ = 277.777... kWh
    return (cent_per_kWh / 100.0) * 277.7777777778

def _is_non_material(conn) -> bool:
    """
    Return True for TESPy non-material/power connections.
    TESPy prints 'Kind: power' in ExergyAnalysis; other builds show 'non-material'.
    """
    try:
        kind = str(getattr(conn, "kind", "")).lower()
        return kind in ("power", "non-material")
    except Exception:
        return False

def _class_name(obj) -> str:
    try:
        return obj.__class__.__name__.lower()
    except Exception:
        return ""

# ---------- HEX helpers (area costing) ---------------------------------------
def _lmtd(Th_in, Th_out, Tc_in, Tc_out, eps=1e-9):
    dT1 = (Th_in - Tc_out)
    dT2 = (Th_out - Tc_in)
    if abs(dT1 - dT2) < eps:
        return max(eps, (dT1 + dT2) / 2.0)
    try:
        return max(eps, abs((dT1 - dT2) / (math.log((dT1 + eps) / (dT2 + eps)))))
    except Exception:
        return max(eps, (abs(dT1) + abs(dT2)) / 2.0)

def _getQ_W(comp):
    try:
        return abs(float(comp.Q.val))
    except Exception:
        return 0.0

def _port_Ts(comp):
    Ts = []
    for side in (0, 1):
        for port in ("inl", "outl"):
            try:
                c = getattr(comp, port)[side]
                Ts.append(float(c.T.val))
            except Exception:
                Ts.append(None)
    return Ts  # [Tin1, Tout1, Tin2, Tout2]

def _lmtd_from_comp(comp, default=10.0):
    Tin1, Tout1, Tin2, Tout2 = _port_Ts(comp)
    if None in (Tin1, Tout1, Tin2, Tout2):
        return default
    Th_in, Th_out = (Tin1, Tout1)
    Tc_in, Tc_out = (Tin2, Tout2)
    try:
        if max(Tin2, Tout2) > max(Tin1, Tout1):
            Th_in, Th_out, Tc_in, Tc_out = Tin2, Tout2, Tin1, Tout1
    except Exception:
        pass
    return _lmtd(Th_in, Th_out, Tc_in, Tc_out)

def _k_for(label: str, *, k_evap, k_cond, k_inter, k_trans, k_econ, k_misc) -> float:
    L = (label or "").lower()
    if "trans" in L or "gaskühler" in L:
        return k_trans          # gas cooler / transcritical HEX
    if "economizer" in L or "econ" in L:
        return k_econ
    if "intermediate heat exchanger" in L or "condenser" in L or "inter" in L:
        return k_inter          # intermediate/condensing side between cycles
    if "evap" in L:
        return k_evap
    if "consumer" in L:
        return k_cond
    return k_misc

# ---------- pumps -------------------------------------------------------------
def _hydraulic_power_W(pump):
    """P_hyd = (m/ρ) * Δp from port data; None if not available."""
    try:
        m   = _safe_float(pump.inl[0].m.val_SI)
        rho = _safe_float(pump.inl[0].rho.val_SI)
        p_i = _safe_float(pump.inl[0].p.val_SI)
        p_o = _safe_float(pump.outl[0].p.val_SI)
        if None in (m, rho, p_i, p_o) or rho <= 0:
            return None
        return (m / rho) * max(p_o - p_i, 0.0)
    except Exception:
        return None

def _driven_equipment_of_motor(hp, motor_component):
    """
    Follow power connections whose source == motor → returns (driven_label, P_W_elec).
    If multiple, we pick the first found.
    """
    for _, c in getattr(hp, "conns", {}).items():
        try:
            if getattr(c, "source", None) is motor_component:
                tgt = getattr(c, "target", None)
                if tgt is None:
                    continue
                lbl = getattr(tgt, "label", None)
                P_W = None
                if hasattr(c, "P") and getattr(c.P, "val", None) is not None:
                    P_W = _safe_float(c.P.val)
                elif hasattr(c, "E") and getattr(c, "E", None) and getattr(c.E, "val_SI", None) is not None:
                    P_W = _safe_float(c.E.val_SI)
                if lbl:
                    return lbl, P_W
        except Exception:
            continue
    return None, None

# ============================================================================ #
#                           PUBLIC API                                         #
# ============================================================================ #
def build_exergo_boundaries(ean, hp):
    """
    Detect exergo-economic boundaries from the model:

    Returns
    -------
    (fuel_inputs, internal_zero, product, loss)
      fuel_inputs:      list[str]  -> non-material connection labels that are fed by a PowerSource (priced fuel)
      internal_zero:    list[str]  -> other non-material legs inside the plant (set cost = 0 €/GJ)
      product:          dict       -> {"inputs":[...], "outputs":[...]} for useful heat
      loss:             dict       -> {"inputs":[...], "outputs":[...]} for environment

    Notes
    -----
    - We rely on TESPy class names (PowerSource, PowerBus, Motor, etc).
    - Labels C3→C1 (sink) and B3→B1 (source) are your standard naming.
    """
    conns = getattr(hp, "conns", {}) or {}
    # fallback: some implementations store them on hp.nw
    if not conns and hasattr(hp, "nw"):
        try:
            conns = hp.nw.conns["object"].to_dict()
        except Exception:
            pass


    fuel_inputs = []
    internal_zero = []

    def _is_non_material(conn) -> bool:
        """
        TESPy 0.9 PowerConnection often does NOT expose `.kind`.
        Detect non-material links robustly.
        """
        cls = conn.__class__.__name__.lower()
        if "powerconnection" in cls:
            return True

        # Fallback heuristics:
        # - Power connections have E/P, but material connections have m/h/p/T etc.
        if hasattr(conn, "E") or hasattr(conn, "P"):
            if not (hasattr(conn, "m") or hasattr(conn, "h") or hasattr(conn, "p") or hasattr(conn, "T")):
                return True

        # Label heuristic: your model uses E0/E1/E2/E3 and e1/e2/e3
        lbl = str(getattr(conn, "label", "") or "")
        if lbl.startswith(("E", "e")):
            return True

        return False


    def _class_name(obj):
        try:
            return obj.__class__.__name__.lower()
        except Exception:
            return ""

    for lbl, c in conns.items():
        if not _is_non_material(c):
            continue
        src = getattr(c, "source", None)
        if _class_name(src) == "powersource":
            fuel_inputs.append(lbl)      # external electricity feed
        else:
            internal_zero.append(lbl)    # internal distribution → zero price

    boundaries = determine_exergy_boundaries(hp)
    fuel = boundaries["fuel"]
    product = boundaries["product"]
    loss = boundaries["loss"]

    for lbl in fuel.get("inputs", []):
        if lbl not in fuel_inputs:
            fuel_inputs.append(lbl)

    return fuel_inputs, internal_zero, product, loss

def build_costs(
    ean, hp, *,
    # default U-values [W/m²K]
    k_evap=1500.0, k_cond=3500.0, k_inter=2200.0, k_trans=60.0, k_econ=1500.0, k_misc=50.0,
    # PEC correlation choices
    hex_cost_model="ommen",
    compressor_cost_model="ommen",
    flash_cost_model="ommen",
    # flash tank costing
    flash_residence_time_s=10.0,
    flash_ref_cost=15000.0,
    flash_ref_volume_m3=1.0,
    flash_cost_exponent=0.6,
    flash_pressure_ref_bar=10.0,
    flash_pressure_exponent=0.0,
    flash_rho_default=1000.0,
    CEPCI_cur=797.9, CEPCI_ref=556.8,
    tci_factor=6.32, omc_rel=0.03, i_eff=0.08, r_n=0.02, n=20,
    tau_h_per_year=5500.0
):
    """
    Returns dicts keyed by *component label*: PEC[€], TCI[€], Z[€/h].

    Rules:
      - Compressors: selectable inlet volumetric-flow or power-based correlation.
      - Pumps: cost the pump (not the motor); prefer hydraulic power; fallback to motor electric power.
      - Motors / valves / buses / sources / cycle closers / split/merge, etc.: Z = 0 (auxiliaries).
      - Heat exchangers (Condenser, HeatExchanger, SimpleHeatExchanger, DropletSeparator/economizer):
        A from Q/(k*LMTD), with selectable area-based cost correlation.
      - CRF uses i_eff with escalation r_n.
      - Flash tank: selectable volume- or mass-flow-based correlation.
    """
    # Keep the global CEPCI_ref argument as backward-compatible fallback, but
    # use literature-specific reference years for the selectable PEC models.
    OMMEN_CEPCI_REF = 567.0
    SHAMOUSH_CEPCI_REF = 596.0
    DAI_CEPCI_REF = 555.0  # average of 2015/2016 values in the package data

    if compressor_cost_model.startswith("shamoushaki"):
        compressor_cepci = CEPCI_cur / SHAMOUSH_CEPCI_REF
    elif compressor_cost_model == "ommen":
        compressor_cepci = CEPCI_cur / OMMEN_CEPCI_REF
    else:
        compressor_cepci = CEPCI_cur / CEPCI_ref

    pump_cepci = CEPCI_cur / SHAMOUSH_CEPCI_REF

    if hex_cost_model.startswith("shamoushaki"):
        hex_cepci = CEPCI_cur / SHAMOUSH_CEPCI_REF
    elif hex_cost_model == "dai_cascade":
        hex_cepci = CEPCI_cur / DAI_CEPCI_REF
    elif hex_cost_model == "ommen":
        hex_cepci = CEPCI_cur / OMMEN_CEPCI_REF
    else:
        hex_cepci = CEPCI_cur / CEPCI_ref

    if flash_cost_model == "dai":
        flash_cepci = CEPCI_cur / DAI_CEPCI_REF
    elif flash_cost_model == "ommen":
        flash_cepci = CEPCI_cur / OMMEN_CEPCI_REF
    else:
        flash_cepci = CEPCI_cur / CEPCI_ref

    PEC = {}

    comps = getattr(hp, "comps", {}) or {}

    # --- compressors ----------------------------------------------------------
    for _, comp in comps.items():
        lbl = str(getattr(comp, "label", "") or "")
        L = lbl.lower()
        if "compressor" not in L or "motor" in L:
            continue
        try:
            m = float(comp.inl[0].m.val_SI)
            rho = float(comp.inl[0].rho.val_SI)
            VM_m3_h = (m / max(rho, 1e-9)) * 3600.0
        except Exception:
            VM_m3_h = 279.8  # safe default
        try:
            Wcomp_kW = abs(float(comp.P.val)) / 1e3
        except Exception:
            Wcomp_kW = 1.0
        if compressor_cost_model == "shamoushaki_centrifugal":
            cost = (
                math.log10(max(Wcomp_kW, 1e-9))
                + 0.03867 * Wcomp_kW ** 2
                + 4446.7 * Wcomp_kW
                + 137800.0
            ) * compressor_cepci
        elif compressor_cost_model == "shamoushaki_reciprocating":
            cost = (
                math.log10(max(Wcomp_kW, 1e-9))
                + 0.04147 * Wcomp_kW ** 2
                + 454.8 * Wcomp_kW
                + 181000.0
            ) * compressor_cepci
        else:
            cost = 19850.0 * (VM_m3_h / 279.8) ** 0.73 * compressor_cepci
        PEC[lbl] = max(cost, 0.0)

    # --- pumps (attach cost to the pump) -------------------------------------
    for _, comp in comps.items():
        lbl = str(getattr(comp, "label", "") or "")
        if "pump" not in lbl.lower():
            continue
        try:
            Wp_kW = abs(float(comp.P.val)) / 1e3
        except Exception:
            Wp_kW = 1.0  # safe default to keep log() defined
        PEC[lbl] = max(
            PEC.get(lbl, 0.0),
            max(
                math.log10(max(Wp_kW, 1e-9))
                - 0.03195 * Wp_kW ** 2
                + 467.2 * Wp_kW
                + 20480.0,
                0.0
            ) * pump_cepci
        )

    # --- heat exchangers (area-based) ----------------------------------------
    def _cost_hex(comp, default_LMTD=12.0):
        Q = _getQ_W(comp)
        if Q <= 1.0:
            return None
        k = _k_for(
            getattr(comp, "label", ""),
            k_evap=k_evap, k_cond=k_cond, k_inter=k_inter, k_trans=k_trans, k_econ=k_econ, k_misc=k_misc
        )
        L = _lmtd_from_comp(comp, default_LMTD)
        if L <= 1e-6 or k <= 1.0:
            return None
        A = Q / (k * L)
        A = max(A, 1e-3)
        if hex_cost_model == "dai_cascade":
            return 383.5 * A ** 0.65 * hex_cepci
        if hex_cost_model == "shamoushaki_shell":
            return max(
                math.log10(A) - 0.06395 * A ** 2 + 947.2 * A + 227.9,
                0.0
            ) * hex_cepci
        if hex_cost_model == "shamoushaki_plate":
            return max(
                math.log10(A) + 0.2581 * A ** 2 + 891.7 * A + 26050.0,
                0.0
            ) * hex_cepci
        return 15526.0 * (A / 42.0) ** 0.8 * hex_cepci

    HEX_TOKENS = ("condenser", "heatexchanger", "simpleheatexchanger", "dropletseparator",
                  "economizer", "transcritical", "evap", "consumer", "intermediate")
    for _, comp in comps.items():
        lbl = str(getattr(comp, "label", "") or "")
        if any(tok in lbl.lower() for tok in HEX_TOKENS):
            try:
                cost = _cost_hex(comp, default_LMTD=12.0)
                if cost:
                    PEC[lbl] = max(PEC.get(lbl, 0.0), cost)
            except Exception:
                pass

    # --- flash tank / drum (volume-based) -----------------------------------
    def _port_val(port, attr):
        obj = getattr(port, attr, None)
        if obj is None:
            return None
        return _safe_float(getattr(obj, "val_SI", getattr(obj, "val", None)))

    def _flash_volume_m3(comp, residence_time_s, rho_default):
        inlets = getattr(comp, "inl", None) or []
        best_vol_flow = None
        best_m = None
        for port in inlets:
            m = _port_val(port, "m")
            rho = _port_val(port, "rho")
            if m is None or m <= 0:
                continue
            if rho is not None and rho > 0:
                vol_flow = m / rho
                if best_vol_flow is None or vol_flow > best_vol_flow:
                    best_vol_flow = vol_flow
                    best_m = m
            else:
                if best_m is None or m > best_m:
                    best_m = m
        if best_vol_flow is None and best_m is not None and rho_default > 0:
            best_vol_flow = best_m / rho_default
        if best_vol_flow is None:
            return None
        return best_vol_flow * max(0.0, float(residence_time_s))

    def _flash_pressure_factor(comp, ref_bar, exponent):
        if exponent == 0:
            return 1.0
        inlets = getattr(comp, "inl", None) or []
        p_bar = None
        for port in inlets:
            p = _port_val(port, "p")
            if p is None:
                continue
            p = p / 1e5
            p_bar = p if p_bar is None else max(p_bar, p)
        if p_bar is None or ref_bar <= 0:
            return 1.0
        return (p_bar / ref_bar) ** exponent

    FLASH_TOKENS = ("flash",)
    for _, comp in comps.items():
        lbl = str(getattr(comp, "label", "") or "")
        if any(tok in lbl.lower() for tok in FLASH_TOKENS):
            if flash_cost_model == "dai":
                m_in = None
                for port in (getattr(comp, "inl", None) or []):
                    m_val = _port_val(port, "m")
                    if m_val is not None and m_val > 0:
                        m_in = m_val if m_in is None else max(m_in, m_val)
                if m_in:
                    PEC[lbl] = max(
                        PEC.get(lbl, 0.0),
                        280.3 * m_in ** 0.67 * flash_cepci
                    )
                else:
                    PEC.setdefault(lbl, 0.0)
            else:
                V_m3 = _flash_volume_m3(comp, flash_residence_time_s, flash_rho_default)
                if V_m3:
                    # Receiver correlation (used for flash tank): C_rec = 1444 * (V_rec/0.089)^0.63
                    # Apply CEPCI correction consistently with other correlations.
                    PEC[lbl] = max(
                        PEC.get(lbl, 0.0),
                        1444.0 * (V_m3 / 0.089) ** 0.63 * flash_cepci
                    )
                else:
                    PEC.setdefault(lbl, 0.0)

    # --- auxiliaries (never costed) ------------------------------------------
    AUX_TOKENS = ("motor", "cycle closer", "cyclecloser", "valve",
                  "splitter", "merge", "droplet", "separator",
                  "powerbus", "power bus", "power source", "powersource")
    for _, comp in comps.items():
        lbl = str(getattr(comp, "label", "") or "")
        if any(t in lbl.lower() for t in AUX_TOKENS):
            PEC.setdefault(lbl, 0.0)

    # --- CAPEX→OPEX -----------------------------------------------------------
    TCI = {k: v * tci_factor for k, v in PEC.items()}
    # capital recovery factor (CRF) with escalation r_n
    if i_eff > r_n:
        a = (i_eff - r_n) / (1.0 - ((1.0 + r_n) / (1.0 + i_eff)) ** n)
    else:
        # fallback to standard CRF to avoid division by zero/negative rates
        a = (i_eff * (1.0 + i_eff) ** n) / (((1.0 + i_eff) ** n) - 1.0)
    hours = max(1.0, float(tau_h_per_year))
    # Theoretical allocation: distribute global CCL/OMCL by PEC share
    sum_PEC = sum(PEC.values())
    sum_TCI = sum(TCI.values())
    if sum_PEC > 0:
        CCL = a * sum_TCI
        OMCL = omc_rel * sum_TCI
        Z = {
            k: ((CCL + OMCL) * (PEC[k] / sum_PEC)) / hours
            for k in TCI
        }
    else:
        # Fallback (no PEC): keep per-component formula to avoid division by zero
        Z = {k: (TCI[k] * a + omc_rel * TCI[k]) / hours for k in TCI}

    return PEC, TCI, Z

# ---------- ExerPy integration (clean public API) ----------------------------


def alias_keys_to_exerpy_components(Z_dict, ean):
    """
    Map user/TESPy label keys to ExerPy component names.
    ExerPy uses component names as created in the ExergyAnalysis.
    We normalize strings to handle spaces/underscores.
    """
    comps_dict = getattr(ean, "components", {}) or {}
    comps = list(comps_dict.keys())

    def norm(s): 
        return str(s).replace(" ", "").replace("_", "").lower()

    normmap = {norm(c): c for c in comps}

    mapped = {}
    for k, v in (Z_dict or {}).items():
        if k in comps_dict:
            mapped[k] = v
            continue
        kn = norm(k)
        mapped[normmap.get(kn, k)] = v
    return mapped


def build_Exe_Eco_Costs(
    *,
    ean,
    hp,
    boundaries,
    internal_zero_labels=None, 
    elec_price_cent_kWh,
    Z_by_component_label,
    set_product_and_loss_to_zero=True
):
    """
    Create the ExerPy user cost dict Exe_Eco_Costs.

    boundaries:
      {
        "fuel": {"inputs":[...], "outputs":[...]},
        "product": {"inputs":[...], "outputs":[...]},
        "loss": {"inputs":[...], "outputs":[...]}
      }

    - Assign component Z as "<ComponentName>_Z"
    - Assign connection prices as "<label>_c"
    - Only fuel inputs get electricity price (€/GJ).
    - By default product + loss streams get 0.
    """
    fuel = boundaries.get("fuel", {}) or {}
    product = boundaries.get("product", {}) or {}
    loss = boundaries.get("loss", {}) or {}

    # Convert price
    elec_price_eur_per_GJ = eur_per_GJ_from_cent_per_kWh(float(elec_price_cent_kWh))

    # ---- component Z ----
    # Z_by_component_label is keyed by TESPy component label
    # ExerPy expects component names in ean.components
    Z_mapped = alias_keys_to_exerpy_components(Z_by_component_label, ean)

    Exe_Eco_Costs = {}
    ean_comps = getattr(ean, "components", {}) or {}

    AUX_ZERO = (
        "cycle closer",
        "power distribution",
        "power bus",
        "powersource",
        "motor",
        "valve",
    )

    # iterate over ALL ExerPy components, not only Z_mapped keys
    for cname in ean_comps.keys():
        cn = str(cname).lower()

        # use mapped Z if available, otherwise zero
        Exe_Eco_Costs[f"{cname}_Z"] = float(Z_mapped.get(cname, 0.0))


    # ---- connection prices ----
    # only price what exists in ean.connections
    ean_conns = getattr(ean, "connections", {}) or {}


    # Seed EVERY connection cost to 0.0 (working-package pattern)
    for lbl in ean_conns.keys():
        Exe_Eco_Costs[f"{lbl}_c"] = 0.0

    internal_zero_labels = internal_zero_labels or []

    # Only the external electric fuel stream E0 gets the electricity price.
    # Thermal source-side fuel streams such as B1 are treated as free by
    # default unless explicitly overridden by the user cost dictionary.
    for lbl in (fuel.get("inputs") or []):
        if lbl in ean_conns:
            Exe_Eco_Costs[f"{lbl}_c"] = (
                elec_price_eur_per_GJ if lbl == "E0" else 0.0
            )

    # NEW: internal non-material legs explicitly set to zero price
    for lbl in internal_zero_labels:
        if lbl in ean_conns:
            Exe_Eco_Costs[f"{lbl}_c"] = 0.0

    if set_product_and_loss_to_zero:
        # Set ONLY reference (usually inlet) streams to zero (matches example: 41_c, 11_c, ...)
        for lbl in (product.get("inputs") or []):
            if lbl in ean_conns:
                Exe_Eco_Costs[f"{lbl}_c"] = 0.0

        for lbl in (loss.get("inputs") or []):
            if lbl in ean_conns:
                Exe_Eco_Costs[f"{lbl}_c"] = 0.0

    return Exe_Eco_Costs

import numpy as np

def debug_wrap_all_aux_eqs(ean):
    for name, comp in (getattr(ean, "components", {}) or {}).items():
        if not hasattr(comp, "aux_eqs"):
            continue
        orig = comp.aux_eqs

        def _wrapped(A, b, counter, equations, *args, __orig=orig, __name=name, **kwargs):
            before = counter
            try:
                out = __orig(A, b, counter, equations, *args, **kwargs)
                A2, b2, counter2, eq2 = out
                print(f"[AUX-EQS] {__name}: counter {before} -> {counter2} | A.shape={A2.shape}")
                return out
            except Exception as e:
                print(f"[AUX-EQS-FAIL] {__name}: counter={before}, A.shape={getattr(A,'shape',None)} -> {e}")
                raise

        comp.aux_eqs = _wrapped



def patch_aux_eqs_safe(ean, class_name: str):
    """
    Wrap aux_eqs of a given ExerPy component class so that:
      - equations is always list-like and indexable
      - missing self.equations does not crash older ExerPy versions
    """
    for name, comp in (getattr(ean, "components", {}) or {}).items():
        if comp.__class__.__name__.lower() != class_name.lower():
            continue
        if not hasattr(comp, "aux_eqs"):
            continue

        orig = comp.aux_eqs

        def _wrapped(A, b, counter, equations, *args, __orig=orig, __comp=comp, __name=name, **kwargs):
            equations = _ensure_equations_list(equations, A)

            # Some ExerPy versions expect comp.equations to exist
            if not hasattr(__comp, "equations") or getattr(__comp, "equations") is None:
                try:
                    __comp.equations = equations
                except Exception:
                    pass

            return __orig(A, b, counter, equations, *args, **kwargs)

        comp.aux_eqs = _wrapped
        print(f"[PATCH] aux_eqs SAFE wrapper enabled for {class_name} '{name}'")




def run_exergoeconomic_from_hp(
    *,
    hp,
    Tamb_K,
    pamb_Pa,
    boundaries,
    elec_price_cent_kWh,
    costcalcparams,
    CEPCI_cur,
    CEPCI_ref,
    tau_h_per_year,
    econ_params=None,
    print_results=False,
    debug=False
):

    """
    One clean function to be called from Streamlit.

    - Builds ExergyAnalysis from TESPy network
    - Runs exergy analysis using boundaries
    - Builds component costs Z via build_costs(...)
    - Builds Exe_Eco_Costs dict
    - Runs ExergoeconomicAnalysis and returns its DataFrames

    Returns:
      df_comp, df_mat1, df_mat2, df_non_mat, ean, Exe_Eco_Costs
    """
    #_ensure_equations_attr_for_exerpy(hp)
    econ_params = econ_params or {}
    i_eff = float(econ_params.get("i_eff", 0.08))
    r_n  = float(econ_params.get("r_n", 0.02))
    n    = int(econ_params.get("n", 20))
    omc_rel = float(econ_params.get("omc_rel", 0.03))
    tci_factor = float(econ_params.get("tci_factor", 6.32))

    # --- build exergy analysis (virgin) ---
    # Condenser workaround re-enabled (Condenser treated as HeatExchanger in ExerPy).
    ean = _make_exerpy_ean_with_condenser_as_hx(
        hp=hp,
        Tamb_K=Tamb_K,
        pamb_Pa=pamb_Pa,
        #condenser_label="Condenser",      # <- falls dein Label anders heißt, hier ändern
        split_physical_exergy=True
    )
    debug_dump_component_types(ean)
    if hp.__class__.__name__ == "HeatPumpIC":
        # IC-specific workaround: only apply if explicitly configured.
        ic_skip = IC_SKIP_POWERBUS_OUTPUT
        if isinstance(ic_skip, str) and ic_skip.strip():
            skip_labels = [s.strip() for s in ic_skip.split(",") if s.strip()]
            patch_powerbus_skip_eqs(ean, skip_output_labels=skip_labels)
    if debug:
        # Enable internal ExerPy debug logging without requiring terminal export.
        os.environ["EXERPY_DEBUG"] = "1"
        debug_dump_component_flags(ean)
        debug_wrap_all_aux_eqs(ean)
    #_ensure_equations_attr_for_exerpy(ean)
    fuel = boundaries.get("fuel") or {"inputs": [], "outputs": []}
    product = boundaries.get("product") or {"inputs": [], "outputs": []}
    loss = boundaries.get("loss") or {"inputs": [], "outputs": []}


    # run exergy analysis
    ean.analyse(E_F=fuel, E_P=product, E_L=loss)

    # Use valve dissipative equations (ExerPy expects this for throttling components).
    for _name, _comp in (getattr(ean, "components", {}) or {}).items():
        if _comp.__class__.__name__.lower() == "valve":
            _comp.is_dissipative = True
    if debug:
        # Reprint after forcing valve dissipative flag.
        debug_dump_component_flags(ean)
    

    # =======================
    # DEBUG: boundary signs
    # =======================
    def _try_get(obj, names):
        for n in names:
            if isinstance(obj, dict) and n in obj:
                return obj[n]
            if hasattr(obj, n):
                return getattr(obj, n)
        return None

    def _print_conn(lbl):
        conns = getattr(ean, "connections", {}) or {}
        c = conns.get(lbl)
        if c is None:
            print(f"[BOUNDARY-DBG] {lbl}: NOT FOUND in ean.connections")
            return

        # Try common field names across ExerPy versions
        candidates = {}
        for n in ["E", "E_dot", "Ex", "Ex_dot", "exergy", "exergy_flow", "power", "P", "P_dot"]:
            v = _try_get(c, [n])
            if v is not None:
                candidates[n] = v

        print(f"\n[BOUNDARY-DBG] {lbl} type={type(c)}")
        if hasattr(c, "__dict__"):
            print(f"[BOUNDARY-DBG] {lbl} __dict__ keys:", list(c.__dict__.keys()))
        print(f"[BOUNDARY-DBG] {lbl} candidate fields:")
        for k, v in candidates.items():
            try:
                vv = float(v)
            except Exception:
                vv = v
            print(f"  - {k}: {vv}")

    print("\n[BOUNDARY-DBG] boundaries used for ean.analyse:")
    print("  fuel   :", fuel)
    print("  product:", product)
    print("  loss   :", loss)

    labels = []
    labels += (fuel.get("inputs") or []) + (fuel.get("outputs") or [])
    labels += (product.get("inputs") or []) + (product.get("outputs") or [])
    labels += (loss.get("inputs") or []) + (loss.get("outputs") or [])
    # unique
    labels = list(dict.fromkeys(labels))

    for lbl in labels:
        _print_conn(lbl)

    print("\n[BOUNDARY-DBG] ean totals (if available):")
    for attr in ["E_F", "E_P", "E_L", "E_D", "epsilon"]:
        if hasattr(ean, attr):
            print(f"  - {attr}: {getattr(ean, attr)}")

    # =======================
    # end DEBUG
    # =======================


    df_comp, df_mat, df_non_mat = ean.exergy_results(print_results=False)
    df_mat1 = df_mat
    df_mat2 = None


    if debug:
        debug_exerpy_equations(ean)
        import pandas as pd
        print("\n[DEBUG] boundaries passed in:")
        print(" fuel:", fuel)
        print(" product:", product)
        print(" loss:", loss)

        print("\n[DEBUG] ean.connections keys (first 50):")
        print(list(getattr(ean, "connections", {}).keys())[:50])

        print("\n[DEBUG] ean.components keys (first 50):")
        print(list(getattr(ean, "components", {}).keys())[:50])

        # Check exergy result tables for NaNs (IMPORTANT!)
        try:
            df_comp, df_mat, df_nonmat = ean.exergy_results(print_results=False)
            print("\n[DEBUG] Exergy component table columns:", list(df_comp.columns))
            if "E_F [kW]" in df_comp.columns:
                bad = df_comp[df_comp["E_F [kW]"].isna() | df_comp.get("E_P [kW]", pd.Series(False)).isna()]
                if len(bad) > 0:
                    print("\n[DEBUG] Components with NaNs in E_F/E_P:")
                    print(bad.to_string(index=False))
        except Exception as e:
            print("[DEBUG] Could not read exergy_results:", e)

    # --- costs Z from your existing function ---
    _, _, Z = build_costs(
        ean, hp,
        CEPCI_cur=float(CEPCI_cur),
        CEPCI_ref=float(CEPCI_ref),
        k_evap=float(costcalcparams.get("k_evap", 1500.0)),
        k_cond=float(costcalcparams.get("k_cond", 3500.0)),
        k_inter=float(costcalcparams.get("k_inter", 2200.0)) if "k_inter" in costcalcparams else 2200.0,
        k_trans=float(costcalcparams.get("k_trans", 60.0)) if "k_trans" in costcalcparams else 60.0,
        k_econ=float(costcalcparams.get("k_econ", 1500.0)) if "k_econ" in costcalcparams else 1500.0,
        k_misc=float(costcalcparams.get("k_misc", 50.0)),
        hex_cost_model=str(costcalcparams.get("hex_cost_model", "ommen")),
        compressor_cost_model=str(costcalcparams.get("compressor_cost_model", "ommen")),
        flash_cost_model=str(costcalcparams.get("flash_cost_model", "ommen")),
        flash_residence_time_s=float(costcalcparams.get("residence_time", 10.0)),
        tci_factor=tci_factor,
        omc_rel=omc_rel,
        i_eff=i_eff,
        r_n=r_n,
        n=n,
        tau_h_per_year=float(tau_h_per_year)
    )

    fuel_inputs, internal_zero, product_b, loss_b = build_exergo_boundaries(ean, hp)
    boundaries_detected = determine_exergy_boundaries(hp)
    boundaries_detected["product"] = product_b
    boundaries_detected["loss"] = loss_b
    for lbl in fuel_inputs:
        if lbl not in boundaries_detected["fuel"]["inputs"]:
            boundaries_detected["fuel"]["inputs"].append(lbl)
    if debug:
        print("\n[BOUNDARY-DBG] UI boundaries:")
        print(" fuel   :", fuel)
        print(" product:", product)
        print(" loss   :", loss)

        print("\n[BOUNDARY-DBG] detected boundaries for costing:")
        print(" fuel   :", boundaries_detected["fuel"])
        print(" product:", boundaries_detected["product"])
        print(" loss   :", boundaries_detected["loss"])
    # --- after boundaries_detected = {...} ---
    if len(boundaries_detected["fuel"].get("inputs", [])) == 0 and len(fuel.get("inputs", [])) > 0:
        print("[BOUNDARY-DBG] WARNING: detected fuel empty -> fallback to UI fuel inputs")
        boundaries_detected["fuel"]["inputs"] = list(fuel["inputs"])
    if debug:
        print("[BOUNDARY-DBG] final costing fuel:", boundaries_detected["fuel"])

    # QUICK CHECK: remove Consumer CAPEX from system balance
    if "Consumer" in Z:
        Z["Consumer"] = 0.0


    Exe_Eco_Costs = build_Exe_Eco_Costs(
        ean=ean,
        hp=hp,
        boundaries=boundaries_detected,   # ✅ USE DETECTED
        internal_zero_labels=internal_zero,
        elec_price_cent_kWh=float(elec_price_cent_kWh),
        Z_by_component_label=Z,
        set_product_and_loss_to_zero=True
    )


    if debug:
        print("\n[DEBUG] Exe_Eco_Costs keys (sorted, first 80):")
        keys = sorted(Exe_Eco_Costs.keys())
        print(keys[:80])
        # verify boundary connection labels exist in ean.connections
        ean_conns = set(getattr(ean, "connections", {}).keys())
        needed = []
        for d in (fuel, product, loss):
            needed += (d.get("inputs") or []) + (d.get("outputs") or [])
        missing = [k for k in sorted(set(needed)) if k not in ean_conns]
        print("[DEBUG] boundary labels missing in ean.connections:", missing)

    _scalarize_ean_ports(ean)

    if debug:
        # --- QUICK TEST (nur Debug): fuel Richtung drehen ---
        fuel_flip = {"inputs": [], "outputs": fuel.get("inputs", [])}
        print("\n[BOUNDARY-DBG] QUICK TEST: fuel flipped:", fuel_flip)
        try:
            ean.analyse(E_F=fuel_flip, E_P=product, E_L=loss)
            if hasattr(ean, "E_F"):
                print("[BOUNDARY-DBG] after flip: E_F =", ean.E_F)
        except Exception as _e:
            print("[BOUNDARY-DBG] flip test failed:", _e)

        # WICHTIG: Zustand zurücksetzen
        ean.analyse(E_F=fuel, E_P=product, E_L=loss)

    # --- Run exergoeconomic analysis (compat across ExerPy versions) ---
    print("\n[COST-DBG] user costs (wichtig):")
    for k in sorted(Exe_Eco_Costs.keys()):
        if k.endswith("_c") or k.endswith("_Z"):
            if k.startswith(("E", "e", "C", "B")) or k.endswith("_Z"):
                print(f"  {k:25s} = {Exe_Eco_Costs[k]}")

    exa = ExergoeconomicAnalysis(ean)

    try:
        if debug:
            print("[DEBUG] Using stepwise API for tracing")

            exa.initialize_cost_variables()
            exa.assign_user_costs(Exe_Eco_Costs)

            # optional quick sanity prints:
            print("[MATRIX-DBG] number of components:", len(getattr(ean, "components", {}) or {}))
            print("[MATRIX-DBG] number of connections:", len(getattr(ean, "connections", {}) or {}))

            # Build matrix once to dump all equations in readable form.
            try:
                exa.construct_matrix()
            except TypeError:
                # Older ExerPy expected Tamb as positional arg.
                exa.construct_matrix(ean.Tamb)
            debug_dump_cost_matrix_equations(exa)
            # Solve (construct_matrix will be called again internally).
            try:
                exa.solve_exergoeconomic_analysis(Tamb=ean.Tamb)
            except TypeError:
                # Newer ExerPy signature takes no kwargs.
                exa.solve_exergoeconomic_analysis()
        else:
            exa.run(Exe_Eco_Costs=Exe_Eco_Costs, Tamb=ean.Tamb)

    # (keep the result-collection code here)



        # ==========================================================
        # Collect exergoeconomic results (Matrix 1 / Matrix 2 / f ...)
        # ==========================================================
        df_execo_comp = None
        df_mat1_ex = None
        df_mat2_ex = None
        df_non_mat_ex = None

        # Try common ExerPy APIs (version differences)
        for meth in ("exergoeconomic_results", "results", "get_results"):
            if hasattr(exa, meth):
                try:
                    out = getattr(exa, meth)(print_results=False)
                except TypeError:
                    out = getattr(exa, meth)()
                except Exception:
                    out = None

                if isinstance(out, tuple):
                    if len(out) >= 4:
                        df_execo_comp, df_mat1_ex, df_mat2_ex, df_non_mat_ex = out[:4]
                        break
                    elif len(out) == 3:
                        df_execo_comp, df_mat1_ex, df_non_mat_ex = out
                        df_mat2_ex = None
                        break

        # Fallback (don’t overwrite the exergy df_mat1 you already have)
        if df_execo_comp is None:
            df_execo_comp = df_comp
            df_mat1_ex = df_mat1
            df_mat2_ex = None
            df_non_mat_ex = df_non_mat            


    except Exception as e:
        import traceback
        print("\n[DEBUG] Exergoeconomic step failed:", e)
        print(traceback.format_exc())
        raise


    return df_execo_comp, df_mat1_ex, df_mat2_ex, df_non_mat_ex, ean, Exe_Eco_Costs
