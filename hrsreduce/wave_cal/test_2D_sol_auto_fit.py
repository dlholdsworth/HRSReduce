import io
import struct
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ============================================================================
# WAVELENGTH CODE (UNCHANGED)
# ============================================================================

# --------- Robust loader for "object scalar" .npy saved with NumPy 2.x ---------
class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # NumPy 2.x may pickle objects under numpy._core.*, NumPy 1.x uses numpy.core.*
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_arc_lines_npy(npy_path, order_numbers=None):
    """
    Load arc line picks from a .npy file that stores a pickled python object.

    Parameters
    ----------
    npy_path : str
        Path to the .npy file.
    order_numbers : array-like or None
        If the dict keys are 0..N-1 (indices), map them onto these real order numbers.
        If None, tries to infer whether keys are already real order numbers.

    Returns
    -------
    pix_all, ord_all, wav_all : 1D numpy arrays
        Concatenated calibration points.
    meta : dict
        Contains mapping information and per-order counts.
    """
    # Manually parse .npy header (v1.0) and unpickle payload
    with open(npy_path, "rb") as f:
        magic = f.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError("Not a .npy file")
        ver = f.read(2)
        if ver != b"\x01\x00":
            raise ValueError(f"Unsupported .npy version: {ver!r} (expected 1.0)")

        header_len = struct.unpack("<H", f.read(2))[0]
        header = f.read(header_len)  # noqa: F841
        payload = f.read()

    obj = _CompatUnpickler(io.BytesIO(payload)).load()

    # The file you attached is a 0-d object array containing a dict
    if isinstance(obj, np.ndarray) and obj.shape == () and obj.dtype == object:
        obj = obj.item()

    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict-like object, got {type(obj)}")

    keys = list(obj.keys())
    n_orders_in_file = len(keys)

    # Determine whether keys are already real order numbers or just 0..N-1 indices
    keys_int = np.array([int(k) for k in keys], dtype=int)
    looks_like_indexed = (np.min(keys_int) == 0) and (np.max(keys_int) == n_orders_in_file - 1)

    if order_numbers is None:
        if looks_like_indexed:
            # Default for your case: real orders 125..84 (descending), 42 orders
            order_numbers = np.arange(85, 52, -1, dtype=int)  # 125..84 inclusive
        else:
            order_numbers = np.unique(keys_int)

    order_numbers = np.asarray(order_numbers, dtype=int)

    if looks_like_indexed:
        if len(order_numbers) != n_orders_in_file:
            raise ValueError(
                f"File contains {n_orders_in_file} orders (keys 0..{n_orders_in_file-1}), "
                f"but order_numbers has length {len(order_numbers)}."
            )
        key_to_order = {int(k): int(order_numbers[int(k)]) for k in keys_int}
    else:
        # assume keys are the actual order numbers
        key_to_order = {int(k): int(k) for k in keys_int}

    pix_list, ord_list, wav_list = [], [], []
    per_order_counts = {}

    for k in sorted(keys_int):
        d = obj[int(k)]
        pix = np.asarray(d["line_positions"], dtype=float).ravel()
        wav = np.asarray(d["known_wavelengths_air"], dtype=float).ravel()
        if pix.size != wav.size:
            raise ValueError(f"Order key {k}: line_positions and known_wavelengths_air differ in length")

        ordnum = key_to_order[int(k)]
        pix_list.append(pix)
        wav_list.append(wav)
        ord_list.append(np.full(pix.size, ordnum, dtype=float))
        per_order_counts[ordnum] = int(pix.size)

    pix_all = np.concatenate(pix_list)
    ord_all = np.concatenate(ord_list)
    wav_all = np.concatenate(wav_list)

    meta = {
        "n_orders_in_file": n_orders_in_file,
        "looks_like_indexed_keys": bool(looks_like_indexed),
        "key_to_order": key_to_order,
        "per_order_counts": per_order_counts,
    }
    return pix_all, ord_all, wav_all, meta


# --------- 2D polynomial surface fit (pix, order) -> wavelength ----------
def _poly_design_matrix(pix_s, ord_s, deg_pix=6, deg_ord=3, cross=True):
    pix_s = np.asarray(pix_s, float).ravel()
    ord_s = np.asarray(ord_s, float).ravel()
    if pix_s.shape != ord_s.shape:
        raise ValueError("pix and order must have the same shape")

    cols = []
    exps = []
    if cross:
        for i in range(deg_pix + 1):
            for j in range(deg_ord + 1):
                cols.append((pix_s**i) * (ord_s**j))
                exps.append((i, j))
    else:
        for i in range(deg_pix + 1):
            cols.append(pix_s**i)
            exps.append((i, 0))
        for j in range(1, deg_ord + 1):
            cols.append(ord_s**j)
            exps.append((0, j))

    A = np.vstack(cols).T
    return A, exps


def fit_wavelength_surface(pix, order, wave,
                           deg_pix=6, deg_ord=3,
                           cross=True,
                           robust=True, n_iter=7, clip_sigma=4.0,
                           ridge=1e-10):
    """
    Fit wavelength(pix, order) with a 2D polynomial and robust reweighting.
    (No per-point uncertainties.)
    """
    pix = np.asarray(pix, float).ravel()
    order = np.asarray(order, float).ravel()
    wave = np.asarray(wave, float).ravel()
    if not (pix.shape == order.shape == wave.shape):
        raise ValueError("pix, order, wave must be same length")

    good = np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave)
    pix, order, wave = pix[good], order[good], wave[good]

    # Scale inputs to ~[-1,1] for numerical stability
    pix_mu = np.mean(pix)
    pix_span = np.max(pix) - np.min(pix)
    pix_span = pix_span if pix_span > 0 else 1.0

    ord_mu = np.mean(order)
    ord_span = np.max(order) - np.min(order)
    ord_span = ord_span if ord_span > 0 else 1.0

    pix_s = (pix - pix_mu) / (0.5 * pix_span)
    ord_s = (order - ord_mu) / (0.5 * ord_span)

    A, exps = _poly_design_matrix(pix_s, ord_s, deg_pix=deg_pix, deg_ord=deg_ord, cross=cross)

    def solve(A, y, w):
        W = w[:, None]
        ATA = A.T @ (W * A)
        ATy = A.T @ (w * y)
        if ridge and ridge > 0:
            ATA = ATA + ridge * np.eye(ATA.shape[0])
        return np.linalg.solve(ATA, ATy)

    coeff = solve(A, wave, np.ones_like(wave))

    if robust:
        for _ in range(n_iter):
            resid = wave - (A @ coeff)
            med = np.median(resid)
            mad = np.median(np.abs(resid - med))
            sigma = 1.4826 * mad if mad > 0 else (np.std(resid) + 1e-12)

            u = resid / (clip_sigma * sigma)
            w = np.zeros_like(wave)
            m = np.abs(u) < 1
            w[m] = (1 - u[m]**2)**2  # Tukey biweight

            if np.sum(w > 0) < A.shape[1]:
                break
            coeff = solve(A, wave, w)

    def predict(pix_new, order_new):
        pix_new = np.asarray(pix_new, float)
        order_new = np.asarray(order_new, float)
        if pix_new.shape != order_new.shape:
            raise ValueError("pix_new and order_new must have same shape")

        pix_s_new = (pix_new - pix_mu) / (0.5 * pix_span)
        ord_s_new = (order_new - ord_mu) / (0.5 * ord_span)
        A_new, _ = _poly_design_matrix(pix_s_new.ravel(), ord_s_new.ravel(),
                                       deg_pix=deg_pix, deg_ord=deg_ord, cross=cross)
        return (A_new @ coeff).reshape(pix_new.shape)

    return {
        "coeff": coeff,
        "exps": exps,
        "deg_pix": deg_pix,
        "deg_ord": deg_ord,
        "cross": cross,
        "pix_mu": pix_mu,
        "pix_span": pix_span,
        "ord_mu": ord_mu,
        "ord_span": ord_span,
        "predict": predict,
    }


def expand_to_full_grid(model, n_pix=2048, order_min=84, order_max=125):
    """
    Produce wavelength grid shape (n_orders, n_pix) with orders descending (125->84).
    """
    orders = np.arange(order_min, order_max + 1, dtype=float)[::-1]
    pix = np.arange(n_pix, dtype=float)
    pix_grid, ord_grid = np.meshgrid(pix, orders)
    wave_grid = model["predict"](pix_grid, ord_grid)
    return wave_grid, orders.astype(int)


# ============================================================================
# PLOTTING (ADDED) — DOES NOT CHANGE THE WAVELENGTH CODE ABOVE
# ============================================================================

def _compute_used_mask_for_plotting(pix, order, wave, model, n_iter=7, clip_sigma=4.0, ridge=1e-10):
    """
    Recompute the robust weights/mask for diagnostics ONLY.
    This leaves the wavelength-fitting code untouched.
    """
    pix = np.asarray(pix, float).ravel()
    order = np.asarray(order, float).ravel()
    wave = np.asarray(wave, float).ravel()

    good = np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave)
    pix_g, ord_g, wav_g = pix[good], order[good], wave[good]

    deg_pix = model["deg_pix"]
    deg_ord = model["deg_ord"]
    cross = model["cross"]

    pix_mu = model["pix_mu"]
    pix_span = model["pix_span"]
    ord_mu = model["ord_mu"]
    ord_span = model["ord_span"]

    pix_s = (pix_g - pix_mu) / (0.5 * pix_span)
    ord_s = (ord_g - ord_mu) / (0.5 * ord_span)

    A, _ = _poly_design_matrix(pix_s, ord_s, deg_pix=deg_pix, deg_ord=deg_ord, cross=cross)

    # Start from the model coefficients
    coeff = model["coeff"]

    def solve(A, y, w):
        W = w[:, None]
        ATA = A.T @ (W * A)
        ATy = A.T @ (w * y)
        if ridge and ridge > 0:
            ATA = ATA + ridge * np.eye(ATA.shape[0])
        return np.linalg.solve(ATA, ATy)

    # Iterate weights (and allow coeff to update, matching the original algorithm behaviour)
    w = np.ones_like(wav_g)
    coeff = solve(A, wav_g, w)

    for _ in range(n_iter):
        resid = wav_g - (A @ coeff)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med))
        sigma = 1.4826 * mad if mad > 0 else (np.std(resid) + 1e-12)

        u = resid / (clip_sigma * sigma)
        w = np.zeros_like(wav_g)
        m = np.abs(u) < 1
        w[m] = (1 - u[m]**2)**2

        if np.sum(w > 0) < A.shape[1]:
            break
        coeff = solve(A, wav_g, w)

    used_good = w > 0
    used_full = np.zeros_like(good, dtype=bool)
    used_full[np.where(good)[0]] = used_good
    return used_full


def plot_diagnostics_like_example(pix, order, wave, model,
                                  n_iter=7, clip_sigma=4.0,
                                  order_min=84, order_max=125,
                                  n_pix_plot=None):
    """
    Make a diagnostic plot matching the example layout:
      - Left big: wavelength curves per order + points (used filled, rejected open)
      - Top-right: residuals vs pixel with RMS and N used/total
      - Bottom-right: residuals vs aperture index with fit + clipping + iter annotation
    """
    pix = np.asarray(pix, float).ravel()
    order = np.asarray(order, float).ravel()
    wave = np.asarray(wave, float).ravel()

    used_mask = _compute_used_mask_for_plotting(
        pix, order, wave, model, n_iter=n_iter, clip_sigma=clip_sigma
    )
    good = np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave)
    used = used_mask & good
    rej = (~used_mask) & good

    # Residuals (at measured line points)
    wave_fit = model["predict"](pix[good], order[good])
    resid = wave[good] - wave_fit

    # Used/rejected counts
    n_used = int(np.sum(used))
    n_tot = int(np.sum(good))

    # RMS on used points only
    if np.any(used[good]):
        rms = float(np.sqrt(np.mean((wave[used] - model["predict"](pix[used], order[used]))**2)))
    else:
        rms = np.nan

    # Define "aperture" indices 0..Norders-1 (like the example plot)
    orders_unique = np.unique(order[good]).astype(int)
    orders_unique = np.sort(orders_unique)  # ascending to make index stable
    order_to_ap = {o: i for i, o in enumerate(orders_unique)}
    aperture = np.array([order_to_ap[int(o)] for o in order[good]], dtype=int)

    # Pixel range for curves in the left panel
    if n_pix_plot is None:
        x_min = float(np.nanmin(pix[good]))
        x_max = float(np.nanmax(pix[good]))
        x = np.linspace(x_min, x_max, 400)
    else:
        x = np.linspace(0, n_pix_plot - 1, 400)

    # Orders shown on the left panel (descending like your example)
    orders_desc = np.arange(order_min, order_max + 1, dtype=float)[::-1]

    # ---- Layout: big left + 2 right panels ----
    fig = plt.figure(figsize=(14, 7), dpi=150)
    gs = gridspec.GridSpec(2, 2, width_ratios=[2.25, 1.0], height_ratios=[1, 1], wspace=0.25, hspace=0.25)

    axL = fig.add_subplot(gs[:, 0])
    axTR = fig.add_subplot(gs[0, 1])
    axBR = fig.add_subplot(gs[1, 1])

    # ---- Left: wavelength curves per order + points ----
    for o in orders_desc:
        w_curve = model["predict"](x, np.full_like(x, o))
        axL.plot(x, w_curve, linewidth=0.8, alpha=0.9)

    # Points: used filled, rejected open (use same color mapping as order)
    sc_used = axL.scatter(pix[used], wave[used], c=order[used], s=10, edgecolors="none")
    axL.scatter(pix[rej], wave[rej], facecolors="none", edgecolors="0.3", s=10, linewidths=0.7)

    axL.set_xlabel("Pixel")
    axL.set_ylabel("λ (Å)")
    axL.grid(True, alpha=0.25)
    cbar = fig.colorbar(sc_used, ax=axL, pad=0.01)
    cbar.set_label("Order")

    # ---- Top-right: residuals vs pixel ----
    axTR.scatter(pix[good], resid, c=order[good], s=8, alpha=0.9, edgecolors="none")
    axTR.set_xlabel("Pixel")
    axTR.set_ylabel("Residual on λ (Å)")
    axTR.set_title(f"R.M.S. = {rms:.5f}, N = {n_used}/{n_tot}", fontsize=9)
    axTR.grid(True, axis="y", linestyle="--", alpha=0.35)
    axTR.axhline(0, linewidth=0.8, alpha=0.7)

    # ---- Bottom-right: residuals vs aperture ----
    axBR.scatter(aperture, resid, c=order[good], s=8, alpha=0.9, edgecolors="none")
    axBR.set_xlabel("Aperture")
    axBR.set_ylabel("Residual on λ (Å)")
    axBR.set_title(
        f"Xorder = {model['deg_pix']}, Yorder = {model['deg_ord']}, clipping = ±{clip_sigma:g}, Niter = {n_iter}",
        fontsize=9
    )
    axBR.grid(True, axis="y", linestyle="--", alpha=0.35)
    axBR.axhline(0, linewidth=0.8, alpha=0.7)

    plt.show()

def compute_rms_per_order(pix, order, wave, model,
                          n_iter=7, clip_sigma=4.0):
    """
    Compute RMS residual for each order (used lines only).

    Returns
    -------
    orders : ndarray
        Sorted unique order numbers (ascending).
    rms_per_order : ndarray
        RMS for each order.
    n_used_per_order : ndarray
        Number of used lines per order.
    """

    pix = np.asarray(pix, float).ravel()
    order = np.asarray(order, float).ravel()
    wave = np.asarray(wave, float).ravel()

    # Recompute which lines were used (same logic as plotting helper)
    used_mask = _compute_used_mask_for_plotting(
        pix, order, wave, model,
        n_iter=n_iter, clip_sigma=clip_sigma
    )

    good = np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave)
    used = used_mask & good

    # Residuals
    wave_fit = model["predict"](pix[good], order[good])
    resid = wave[good] - wave_fit

    order_good = order[good]
    used_good = used[good]

    orders_unique = np.unique(order_good).astype(int)
    orders_unique = np.sort(orders_unique)

    rms_per_order = np.zeros_like(orders_unique, dtype=float)
    n_used_per_order = np.zeros_like(orders_unique, dtype=int)

    for i, o in enumerate(orders_unique):
        m = (order_good == o) & used_good
        if np.any(m):
            rms_per_order[i] = np.sqrt(np.mean(resid[m]**2))
            n_used_per_order[i] = np.sum(m)
        else:
            rms_per_order[i] = np.nan
            n_used_per_order[i] = 0

    return orders_unique, rms_per_order, n_used_per_order
    

def _overall_rms_used(pix, order, wave, model, used_mask):
    """Overall RMS residual on used lines only."""
    good = np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave)
    used = used_mask & good
    if np.sum(used) == 0:
        return np.nan
    resid = wave[used] - model["predict"](pix[used], order[used])
    return float(np.sqrt(np.mean(resid**2)))


def _flatness_metric(rms_per_order, n_used_per_order, min_lines_per_order=3):
    """
    Flatness metric: std(rms_order)/median(rms_order), ignoring orders with too few lines.
    Lower is better.
    """
    rms_per_order = np.asarray(rms_per_order, float)
    n_used_per_order = np.asarray(n_used_per_order, int)

    m = np.isfinite(rms_per_order) & (n_used_per_order >= min_lines_per_order)
    if np.sum(m) < 3:
        return np.nan

    vals = rms_per_order[m]
    med = np.median(vals)
    if med <= 0:
        return np.nan
    return float(np.std(vals) / med)


def select_best_degrees(
    pix, order, wave,
    deg_pix_grid=range(3, 9),        # try 3..8
    deg_ord_grid=range(1, 6),        # try 1..5
    cross=True,
    robust=True,
    n_iter=7,
    clip_sigma=4.0,
    ridge=1e-10,
    min_lines_per_order=3,
    alpha=1.0,
):
    """
    Grid-search polynomial degrees. Returns:
      - results: list of dicts (one per degree pair)
      - best_by_rms: dict
      - best_by_flatness: dict
      - best_by_score: dict (score = rms * (1 + alpha*flatness))

    Notes
    -----
    - 'rms' uses used lines only (robust-clipped).
    - 'flatness' is std(per-order-rms)/median(per-order-rms) over orders with >=min_lines_per_order used lines.
    - score combines both: smaller is better.
    """
    pix = np.asarray(pix, float).ravel()
    order = np.asarray(order, float).ravel()
    wave = np.asarray(wave, float).ravel()

    results = []

    for dp in deg_pix_grid:
        for do in deg_ord_grid:
            model = fit_wavelength_surface(
                pix, order, wave,
                deg_pix=dp, deg_ord=do,
                cross=cross,
                robust=robust, n_iter=n_iter, clip_sigma=clip_sigma,
                ridge=ridge
            )

            # Determine which points were used (same robust logic as in your plotting helper)
            used_mask = _compute_used_mask_for_plotting(
                pix, order, wave, model,
                n_iter=n_iter, clip_sigma=clip_sigma, ridge=ridge
            )

            # Per-order RMS using used lines only
            orders_u, rms_u, n_used_u = compute_rms_per_order(
                pix, order, wave, model,
                n_iter=n_iter, clip_sigma=clip_sigma
            )

            rms = _overall_rms_used(pix, order, wave, model, used_mask)
            flat = _flatness_metric(rms_u, n_used_u, min_lines_per_order=min_lines_per_order)

            # Combine metrics into a single score (dimensionless multiplier on rms)
            # If flat is NaN (not enough orders), score becomes NaN.
            score = float(rms * (1.0 + alpha * flat)) if (np.isfinite(rms) and np.isfinite(flat)) else np.nan

            results.append({
                "deg_pix": dp,
                "deg_ord": do,
                "rms": rms,
                "flatness": flat,
                "score": score,
                "n_used_total": int(np.sum(used_mask)),
                "n_total": int(np.sum(np.isfinite(pix) & np.isfinite(order) & np.isfinite(wave))),
                "orders": orders_u,
                "rms_per_order": rms_u,
                "n_used_per_order": n_used_u,
            })

    # Choose bests (ignore NaNs)
    finite_rms = [r for r in results if np.isfinite(r["rms"])]
    finite_flat = [r for r in results if np.isfinite(r["flatness"])]
    finite_score = [r for r in results if np.isfinite(r["score"])]

    best_by_rms = min(finite_rms, key=lambda r: r["rms"]) if finite_rms else None
    best_by_flatness = min(finite_flat, key=lambda r: r["flatness"]) if finite_flat else None
    best_by_score = min(finite_score, key=lambda r: r["score"]) if finite_score else None

    # Sort results by score then rms (NaNs go last)
    def _sort_key(r):
        s = r["score"]
        return (np.inf if not np.isfinite(s) else s, r["rms"] if np.isfinite(r["rms"]) else np.inf)

    results_sorted = sorted(results, key=_sort_key)

    return results_sorted, best_by_rms, best_by_flatness, best_by_score


# --------------------------- Run it on your file ---------------------------
if __name__ == "__main__":
    npy_path = "./Intermediate_files/MR_R_linelist_L_NEW.npy"

    # If your file keys are 0..41, this maps them to real orders 125..84 by default.
    pix, order, wave, meta = load_arc_lines_npy(npy_path)
    
    results, best_rms, best_flat, best_score = select_best_degrees(
    pix, order, wave,
    deg_pix_grid=range(1,9),
    deg_ord_grid=range(1, 9),
    cross=True,
    robust=True,
    n_iter=5,
    clip_sigma=4.0,
    ridge=1e-10,
    min_lines_per_order=1,
    alpha=1.0,          # increase alpha to prioritise flatness more
)

    print("Best by RMS:", best_rms["deg_pix"], best_rms["deg_ord"], "rms=", best_rms["rms"], "flat=", best_rms["flatness"])
    print("Best by flatness:", best_flat["deg_pix"], best_flat["deg_ord"], "rms=", best_flat["rms"], "flat=", best_flat["flatness"])
    print("Best by combined score:", best_score["deg_pix"], best_score["deg_ord"], "rms=", best_score["rms"], "flat=", best_score["flatness"], "score=", best_score["score"])

    print("\nTop 10 degree pairs by score:")
    for r in results[:10]:
        print(f"dp={r['deg_pix']:d}, do={r['deg_ord']:d} | rms={r['rms']:.6f} | flat={r['flatness']:.4f} | score={r['score']:.6f} | used={r['n_used_total']}/{r['n_total']}")
        
    winner = best_score
    model = fit_wavelength_surface(
    pix, order, wave,
    deg_pix=4, deg_ord=8,
    cross=True, robust=True, n_iter=5, clip_sigma=4.0, ridge=1e-10
    )

    plot_diagnostics_like_example(pix, order, wave, model, n_iter=5, clip_sigma=4.0, order_min=53, order_max=85, n_pix_plot=4096)
    
    
#
#
#    print("Loaded arc points:", pix.size)
#    print("Per-order line counts (sample):", dict(list(meta["per_order_counts"].items())[:5]))
#    print("Key mapping type:", "indexed->real orders" if meta["looks_like_indexed_keys"] else "keys are orders")
#
#    model = fit_wavelength_surface(
#        pix, order, wave,
#        deg_pix=6, deg_ord=3,
#        cross=True,
#        robust=True,
#        clip_sigma=4.0,
#        ridge=1e-10
#    )
#
#    wave_grid, orders_desc = expand_to_full_grid(model, n_pix=2048, order_min=84, order_max=125)
#    print("Expanded wavelength grid:", wave_grid.shape, "orders:", orders_desc[0], "->", orders_desc[-1])
#
#    # Produce the diagnostic plot (like your example)
#    plot_diagnostics_like_example(
#        pix, order, wave, model,
#        n_iter=7, clip_sigma=4.0,
#        order_min=84, order_max=125,
#        n_pix_plot=2048
#    )
#
#    
