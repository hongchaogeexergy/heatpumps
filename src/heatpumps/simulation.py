from copy import deepcopy

import numpy as np

import variables as var


_A_TARGET_CALIBRATION_MODELS = {
    'flash': (0.5, 3.0),
    'econ_closed': (0.5, 2.0),
}


def _instantiate_hp(hp_model_name, params):
    """Create a heat pump instance for the requested model."""
    if 'econ' in hp_model_name:
        return var.hp_model_classes[hp_model_name](
            params, econ_type=var.hp_models[hp_model_name]['econ_type']
        )
    return var.hp_model_classes[hp_model_name](params)


def _run_design_once(hp_model_name, params, **run_kwargs):
    """Run a single TESPy design simulation without extra calibration logic."""
    hp = _instantiate_hp(hp_model_name, params)
    hp.run_model(**run_kwargs)
    return hp


def _supports_a_target_calibration(hp_model_name, params):
    """Return whether A-target calibration is requested and supported."""
    setup = params.get('setup', {})
    return (
        setup.get('calibration_mode') == 'a_target'
        and hp_model_name in _A_TARGET_CALIBRATION_MODELS
        and 'A_target' in setup
    )


def _evaluate_a_at_rip(hp_model_name, params, rip_factor):
    """Run one design solve at a specific RIP and return A_inj."""
    candidate_params = deepcopy(params)
    candidate_params.setdefault('setup', {})
    candidate_params['setup']['rip_factor'] = float(rip_factor)
    hp = _run_design_once(
        hp_model_name, candidate_params, exergy_analysis=False
    )
    injection = hp.get_injection_metrics()
    a_inj = float(injection.get('A_inj', np.nan))
    if not np.isfinite(a_inj):
        raise RuntimeError(
            f'Kein gültiger Einspritzmassenanteil A für RIP={rip_factor:.4f}.'
        )
    return a_inj


def _calibrate_rip_to_target_a(hp_model_name, params):
    """Calibrate RIP so the solved model approaches the requested A_target."""
    target_a = float(params['setup']['A_target'])
    rip_lower, rip_upper = _A_TARGET_CALIBRATION_MODELS[hp_model_name]
    coarse_rips = np.linspace(rip_lower, rip_upper, 17)

    successful = []
    failed_rips = []
    for rip in coarse_rips:
        try:
            successful.append((float(rip), _evaluate_a_at_rip(hp_model_name, params, rip)))
        except Exception:
            failed_rips.append(float(rip))

    if not successful:
        raise RuntimeError(
            'Die A-Kalibrierung konnte keinen stabilen RIP-Stützpunkt finden.'
        )

    successful.sort(key=lambda item: item[0])
    best_rip, best_a = min(
        successful, key=lambda item: abs(item[1] - target_a)
    )

    bracket = None
    for left, right in zip(successful, successful[1:]):
        left_err = left[1] - target_a
        right_err = right[1] - target_a
        if left_err == 0 or right_err == 0 or left_err * right_err < 0:
            bracket = [left, right]
            break

    if bracket is not None:
        (rip_left, a_left), (rip_right, a_right) = bracket
        for _ in range(12):
            rip_mid = 0.5 * (rip_left + rip_right)
            a_mid = _evaluate_a_at_rip(hp_model_name, params, rip_mid)
            if abs(a_mid - target_a) < abs(best_a - target_a):
                best_rip, best_a = rip_mid, a_mid

            left_err = a_left - target_a
            mid_err = a_mid - target_a
            if abs(mid_err) <= 5e-4:
                best_rip, best_a = rip_mid, a_mid
                break
            if left_err == 0 or left_err * mid_err < 0:
                rip_right, a_right = rip_mid, a_mid
            else:
                rip_left, a_left = rip_mid, a_mid
    elif len(successful) > 1:
        best_index = min(
            range(len(successful)),
            key=lambda idx: abs(successful[idx][1] - target_a)
        )
        left_index = max(best_index - 1, 0)
        right_index = min(best_index + 1, len(successful) - 1)
        rip_left = successful[left_index][0]
        rip_right = successful[right_index][0]
        if rip_right > rip_left:
            for rip_mid in np.linspace(rip_left, rip_right, 9):
                try:
                    a_mid = _evaluate_a_at_rip(hp_model_name, params, rip_mid)
                except Exception:
                    continue
                if abs(a_mid - target_a) < abs(best_a - target_a):
                    best_rip, best_a = float(rip_mid), a_mid

    calibrated_params = deepcopy(params)
    calibrated_params.setdefault('setup', {})
    calibrated_params['setup']['rip_factor'] = float(best_rip)
    hp = _run_design_once(hp_model_name, calibrated_params)

    achieved_a = float(hp.get_injection_metrics().get('A_inj', np.nan))
    hp.calibration_result = {
        'mode': 'a_target',
        'target_A': target_a,
        'achieved_A': achieved_a,
        'rip_factor': float(best_rip),
        'matched': np.isfinite(achieved_a) and abs(achieved_a - target_a) <= 0.01,
        'failed_rips': failed_rips,
        'search_bounds': (rip_lower, rip_upper),
    }
    return hp


def run_design(hp_model_name, params):
    """Run TESPy design simulation of heat pump."""
    if _supports_a_target_calibration(hp_model_name, params):
        return _calibrate_rip_to_target_a(hp_model_name, params)
    return _run_design_once(hp_model_name, params)


def run_partload(hp):
    """Run TESPy offdesign simulation of heat pump."""
    if (
        hasattr(hp, 'supports_partload_boundary_modes')
        and not hp.supports_partload_boundary_modes()
    ):
        raise NotImplementedError(hp.get_partload_mode_reason())
    hp.offdesign_simulation()
    partload_char = hp.calc_partload_char()

    return hp, partload_char
