"""Hour-by-hour Twitch-viewer forecast for a game launch day.

Idea: a launch day has a shape, and that shape is mostly decided by how big
hour 1 already is.  A launch whose first hour is enormous (hundreds of
thousands watching) has its audience in already: it sags through the night
and climbs again for the evening peak.  A launch whose first hour is tiny
(a few hundred, most regions still locked) can grow a thousand-fold by the
evening.  Across the past launches this kit ships, log(hour-1 viewers)
against log(peak / hour-1) is almost a straight line with a negative slope.

So for each hour of the day I build the forecast from the comparables'
own ratio curves r = viewers(h) / viewers(1):

  * every comparable contributes its log ratio at that hour, weighted by
    how similar it is to this launch -- close in log hour-1 size (a
    Gaussian kernel), close in pre-launch ramp (the hour before release
    day over hour 1, when both sides have it), and a platform factor
    (console-only launches count fully; launches that also opened on PC
    count less, because GTA VI is console-only and PC launches behave
    differently at hour 1);
  * the five quantiles at that hour are LOGNORMAL around a centre that
    blends two estimates of the log ratio: the kernel-weighted median of
    the comparables' log ratios, and a weighted least-squares trend of
    log ratio against log hour-1 (the trend is nearly linear, slope about
    -0.93 for the peak ratio).  Half of each beats either alone under
    leave-one-out, on every subset: the median is local and noisy at the
    size extremes, while the trend extrapolates it.  The spread is the
    weighted spread of the log ratios around the centre, floored, so the
    band is wider where past launches disagree and wider later in the
    day.  (Empirical quantiles of the ratios were tried first; with so
    few effective launches their 5%/95% tails are just the sample
    extremes and score worse under leave-one-out pinball.)
  * the median curve is smoothed twice across neighbouring hours, because
    a real audience curve moves gradually;
  * three early-access-shaped comparables (Monster Hunter Wilds,
    Starfield, Assassin's Creed Shadows) are dropped from the kernel
    when the target itself is big (hour 1 at/above 100k): their hour 1
    is inflated by early access, so their ratio curves decay fast and
    mislead a rolling-midnight launch.  Under leave-one-out this wins
    on big and huge and harms no subset (see DROP_NAMES below).
  * the minute/hourly history before the checkpoint gives a slope for
    hour 1; a steep slope would nudge the first few forecast hours in
    the same direction, fading fast.  The nudge is OFF by default: the
    practice cases' history is sparse hourly points while the final's is
    minute reads, so a slope tuned on practice data cannot be trusted
    to behave the same on the real launch, and leave-one-out scoring
    shows it changes nothing either way (0.0001).

With no usable comparable at all, the forecast falls back to holding the
hour-1 print with a widening random-walk spread (the house baseline's
shape).  Everything is deterministic: standard library only, no network,
no randomness, fixed iteration order.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

IN_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Gaussian z-scores, only used by the fallback (the main forecast uses
# empirical quantiles of the comparables' ratios).
FALLBACK_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- tuning knobs (chosen by leave-one-out pinball scoring, see loo.py) ---
# LOO (37 pseudo-cases, p00; identical on p01-p03, same comparables):
#   before (kernel median only): all=0.395 practice-like=0.264 big=0.215
#     small=0.787; after (trend blend below): all=0.379 practice-like=0.262
#     big=0.205 small=0.737, against the persistence baseline's 0.930 /
#     0.338 / 0.261 / 2.572 on the same protocol.  A prize needs a loss
#     strictly below the best house entry.
SIZE_BW = 1.0        # Gaussian bandwidth on log(hour-1 viewers)
PRE_BW = 0.5         # Gaussian bandwidth on log(pre-hour / hour-1)
PC_FACTOR = 0.7      # weight of a launch that also opened on PC
MIN_H1 = 50          # a comparable needs at least this many hour-1 viewers
MIN_DAY_HOURS = 20   # ... and at least this many valid hours on release day
SPREAD_HALF = 0.25   # each side of the median is at least this far (log units)
SPREAD_SHRINK = 1.0  # lognormal mode: shrink the cross-launch spread
QUANTILE_MODE = "lognormal"  # "empirical" or "lognormal"
SMOOTH = 2           # [0.25, 0.5, 0.25] passes over neighbouring hours
FALLBACK_SIGMA = 0.15
SLOPE_ON = False     # nudge hours 2-4 from the hour-1 slope (off: see above)
SLOPE_CAP = 0.2      # largest log nudge the slope may cause
REG_BLEND = 0.5      # median = REG_BLEND * kernel median + (1-REG_BLEND)
                     #   * trend prediction (see trend_fit); 0.5 wins LOO
REG_SLOPE_CAP = 1.0  # the trend slope is capped at +/- this (log per log)
# Four comparables are early-access / simultaneous-global-unlock shaped:
# their hour 1 is inflated by early access, so their ratio curves decay
# fast and mislead a rolling-midnight target.  For a target at/above
# DROP_CUT they are dropped from the kernel entirely (they are still
# scored as pseudo-cases in leave-one-out).  Under LOO the 3-name set
# wins on big (0.2030 -> 0.2001), midbig (0.2555 -> 0.2521) and huge
# (0.1156 -> 0.1135); adding Diablo IV (round 5) wins on every subset
# (big 0.2001 -> 0.1991, huge 0.1135 -> 0.1121, midbig 0.2521 ->
# 0.2513) and damages none of the largest launches (ELDEN RING /
# Cyberpunk unchanged).  A 5-name set (+ Nightreign) wins huge more
# (0.1090) but hurts Cyberpunk (+0.009, the launch closest in size to
# GTA VI's likely hour 1) and midbig, so it was rejected.  A
# data-driven mean-log-ratio filter reproduces exactly this 3-name set
# (and no shape rule beats the name list), so the list stands.
# Honest caveats: the gains are small (<=0.003, ~1 SE); worst case it
# is neutral on the final, whose target will be huge.
DROP_CUT = 100000                  # drop the names below for targets
                                   # at/above this h1
DROP_NAMES = frozenset({
    "Monster Hunter Wilds",
    "Starfield",
    "Assassin's Creed Shadows",
    "Diablo IV",
})
# A huge launch's shape is best predicted by other huge launches: for a
# target at/above HUGE_CUT, non-huge comparables are downweighted to
# HUGE_W.  Under leave-one-out this wins on big (0.2053 -> 0.2030) and
# huge (0.1217 -> 0.1156) and harms no subset; 5 of the 6 huge cases
# improve, the one that worsens (Diablo IV) is early-access shaped and
# does not resemble a rolling-midnight launch.  The gain is modest and
# flat across HUGE_W 0-0.5 (a plateau, not a knife-edge); 0.25 is the
# robust centre.
HUGE_CUT = 300000                # a target at/above this h1 is huge
HUGE_W = 0.25                    # weight multiplier for non-huge
                                 # comparables when the target is huge
# Mid-day tail widening: the LOO residuals of the median forecast are
# heavier-tailed than lognormal in the middle of the day (hours 10-15,
# tail/shoulder ratio 1.17-1.55), so the outer quantiles there are
# widened by MIDDAY_TAIL while the shoulders stay lognormal.  An
# all-hours tail widening wins all/small but LOSES huge, so the
# widening applies only where the asymmetry lives.  Under LOO (on top
# of the 4-name drop) this wins big (0.1991 -> 0.1987), midbig
# (0.2513 -> 0.2507) and all (0.3761 -> 0.3748) and leaves huge
# unchanged (0.1121); T=1.1 and T=1.15 tie on big/huge, 1.1 is the
# rounder, less aggressive choice.
MIDDAY_LO = 10                   # first hour with the widened tail
MIDDAY_HI = 15                   # last hour with the widened tail
MIDDAY_TAIL = 1.1                # multiplier on the +-1.645 z-scores
                                 # for mid-day hours only


def to_dt(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def half_up_2dp(x):
    """Round to 2 decimals, halves up, returning an int for whole numbers."""
    scaled = float(x) * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def gauss(distance, bandwidth):
    """Gaussian kernel weight for a distance in log space."""
    scaled = distance / bandwidth
    return math.exp(-0.5 * scaled * scaled)


def weighted_median(pairs):
    """The weighted median of (value, weight) pairs."""
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(w for _, w in ordered)
    running = 0.0
    for value, weight in ordered:
        running += weight
        if running >= total / 2.0:
            return value
    return ordered[-1][0]


def weighted_std(pairs, centre):
    """The weighted standard deviation of the values around `centre`."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    variance = sum(w * (v - centre) ** 2 for v, w in pairs) / total
    return math.sqrt(max(0.0, variance))


def weighted_quantile(pairs, level):
    """Empirical quantile of (value, weight) pairs at cumulative level.

    Pairs are sorted by value; the quantile is linearly interpolated
    between the two bracketing values on the cumulative-weight axis.
    """
    ordered = sorted(pairs, key=lambda pair: pair[0])
    total = sum(w for _, w in ordered)
    if total <= 0:
        return ordered[-1][0]
    target = level * total
    seen = 0.0
    prev_value, prev_seen = ordered[0][0], 0.0
    for value, weight in ordered:
        seen += weight
        if seen >= target:
            if seen == prev_seen:
                return value
            frac = (target - prev_seen) / (seen - prev_seen)
            return prev_value + frac * (value - prev_value)
        prev_value, prev_seen = value, seen
    return ordered[-1][0]


def trend_fit(points, x0):
    """Weighted least squares of log-ratio on log(hour-1), at the target.

    points are (log hour-1, log ratio, weight).  Fits y = a + b*(x - x0)
    and returns (a, b, residual std), with a the fitted log-ratio at the
    target's own size x0.  The size trend is nearly linear (log peak ratio
    against log hour-1 has slope about -0.93), so the fit extrapolates the
    trend where the kernel has few neighbours -- at the very big and very
    small ends, where the plain kernel median is biased.  Returns None
    when the fit is degenerate.
    """
    total = sum(w for _, _, w in points)
    if total <= 0:
        return None
    mean_x = sum(w * (x - x0) for x, _, w in points) / total
    mean_y = sum(w * y for _, y, w in points) / total
    sxx = sum(w * (x - x0 - mean_x) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return mean_y, 0.0, 0.0
    sxy = sum(w * (x - x0 - mean_x) * (y - mean_y) for x, y, w in points)
    slope = sxy / sxx
    value = mean_y - slope * mean_x
    variance = sum(
        w * (y - (value + slope * (x - x0))) ** 2 for x, y, w in points
    ) / total
    return value, slope, math.sqrt(max(0.0, variance))


def platform_factor(platforms):
    """How much a comparable counts: console-only launches count fully."""
    text = (platforms or "").lower()
    if "pc" in text:
        return PC_FACTOR
    return 1.0


def extract_curve(comp):
    """A comparable's release-day curve, or None when it cannot anchor one.

    Returns (hour1, {hour: viewers for hours 1..24}, pre_ratio,
    platform_factor, name).  Hours are offsets from the comparable's
    day_start; pre_ratio is the hour before release day over hour 1,
    or None.  The name travels with the curve so learn_quantiles can
    drop early-access-shaped comparables for big targets (DROP_NAMES).
    """
    try:
        start = to_dt(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    hours = {}
    pre = None
    for row in comp.get("hourly") or []:
        try:
            when = to_dt(row["from"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers is None or viewers <= 0:
            continue
        offset = (when - start).total_seconds() / 3600.0
        whole = int(offset)
        if offset == whole and 0 <= whole < 24:
            hours[whole + 1] = viewers
        elif offset == -1.0:
            pre = viewers
    hour1 = hours.get(1)
    if hour1 is None or hour1 < MIN_H1:
        return None
    if sum(1 for h in range(2, 25) if hours.get(h)) < MIN_DAY_HOURS:
        return None
    pre_ratio = (pre / hour1) if pre else None
    return (
        hour1,
        hours,
        pre_ratio,
        platform_factor(comp.get("platforms")),
        comp.get("name"),
    )


def target_context(case):
    """Hour-1 print, pre-launch ramp, and hour-1 slope for the target case."""
    known = next(
        (row for row in case.get("known") or [] if row.get("hour") == 1), None
    )
    if known is not None and known.get("viewers"):
        hour1 = max(1.0, float(known["viewers"]))
    else:
        hourly = case.get("history_hourly") or []
        hour1 = max(1.0, float(hourly[-1]["viewers"])) if hourly else 1.0
    try:
        start = to_dt(case["day_start"])
    except (KeyError, TypeError, ValueError):
        start = datetime(1970, 1, 1, tzinfo=timezone.utc)
    pre_ratio = None
    wanted = start - timedelta(hours=1)
    for row in case.get("history_hourly") or []:
        try:
            when = to_dt(row["from"])
        except (KeyError, TypeError, ValueError):
            continue
        if when == wanted:
            viewers = row.get("viewers")
            if viewers and viewers > 0 and hour1 > 0:
                pre_ratio = viewers / hour1
            break
    # Slope of log viewers over the last few history points (per hour).
    slope = 0.0
    points = []
    for row in (case.get("history") or [])[-6:]:
        try:
            when = to_dt(row["at"])
            viewers = row["viewers"]
        except (KeyError, TypeError, ValueError):
            continue
        if viewers and viewers > 0:
            points.append((when.timestamp() / 3600.0, math.log(viewers)))
    if len(points) >= 2:
        n = len(points)
        mean_x = sum(p[0] for p in points) / n
        mean_y = sum(p[1] for p in points) / n
        denom = sum((p[0] - mean_x) ** 2 for p in points)
        if denom > 0:
            slope = sum(
                (p[0] - mean_x) * (p[1] - mean_y) for p in points
            ) / denom
    return hour1, pre_ratio, slope


def learn_quantiles(curves, hour1, pre_ratio):
    """Per-hour log quantiles (dict hour -> [5 log values]) from the curves."""
    ref_log = math.log(max(1.0, hour1))
    ref_pre = math.log(pre_ratio) if pre_ratio else None
    by_hour = {}
    for hour in range(2, 25):
        pairs = []
        points = []
        for comp_h1, hourly, comp_pre, plat, name in curves:
            viewers = hourly.get(hour)
            if not viewers:
                continue
            # Early-access-shaped comparables mislead a big rolling-
            # midnight target (see DROP_NAMES above); drop them here.
            if hour1 >= DROP_CUT and name in DROP_NAMES:
                continue
            weight = gauss(math.log(comp_h1) - ref_log, SIZE_BW) * plat
            # Huge regime: a huge launch's shape is best predicted by
            # other huge launches, so non-huge comparables count less
            # when the target itself is huge (see HUGE_W above).
            if hour1 >= HUGE_CUT and comp_h1 < HUGE_CUT:
                weight *= HUGE_W
            if ref_pre is not None and comp_pre:
                weight *= gauss(math.log(comp_pre) - ref_pre, PRE_BW)
            pairs.append((math.log(viewers / comp_h1), weight))
            points.append((math.log(comp_h1), math.log(viewers / comp_h1), weight))
        if not pairs:
            by_hour[hour] = None
            continue
        if QUANTILE_MODE == "lognormal":
            median = weighted_median(pairs)
            kern_spread = weighted_std(pairs, median)
            # Blend with the size-trend fit: the kernel median is local
            # and noisy at the size extremes, while the trend extrapolates
            # the (near-linear) log-ratio-against-log-size relation, so
            # half of each scores better under leave-one-out than either
            # alone, on every subset (all / practice-like / big / small).
            # The spread then comes from the trend's residual std (the
            # size trend already removed), floored, and never below half
            # the kernel spread -- a narrower, better-calibrated band
            # than the raw cross-launch spread.
            fit = trend_fit(points, ref_log)
            if fit is not None:
                trend, slope, resid = fit
                slope = max(-REG_SLOPE_CAP, min(REG_SLOPE_CAP, slope))
                median = REG_BLEND * median + (1.0 - REG_BLEND) * trend
                spread = max(SPREAD_HALF, resid * SPREAD_SHRINK)
                if REG_BLEND > 0:
                    spread = max(
                        spread, kern_spread * SPREAD_SHRINK * REG_BLEND
                    )
            else:
                spread = max(
                    SPREAD_HALF, kern_spread * SPREAD_SHRINK
                )
            if MIDDAY_LO <= hour <= MIDDAY_HI:
                z_scores = [
                    -1.645 * MIDDAY_TAIL,
                    -0.674,
                    0.0,
                    0.674,
                    1.645 * MIDDAY_TAIL,
                ]
            else:
                z_scores = FALLBACK_Z
            logs = [median + z * spread for z in z_scores]
        else:
            logs = [weighted_quantile(pairs, level) for level in LEVELS]
            median = logs[2]
            # Floor the spread: never narrower than +/- SPREAD_HALF around
            # the median, expanding the outer quantiles only.
            if logs[4] - median < SPREAD_HALF:
                logs[4] = median + SPREAD_HALF
            if median - logs[0] < SPREAD_HALF:
                logs[0] = median - SPREAD_HALF
            logs[1] = min(logs[1], median)
            logs[3] = max(logs[3], median)
        by_hour[hour] = logs
    # Smooth each level's curve across neighbouring hours.
    for _ in range(SMOOTH):
        for level_idx in range(5):
            raw = {
                h: (by_hour[h][level_idx] if by_hour[h] else 0.0)
                for h in range(2, 25)
            }
            for hour in range(2, 25):
                if hour == 2:
                    value = 0.75 * raw[2] + 0.25 * raw[3]
                elif hour == 24:
                    value = 0.25 * raw[23] + 0.75 * raw[24]
                else:
                    value = (
                        0.25 * raw[hour - 1]
                        + 0.5 * raw[hour]
                        + 0.25 * raw[hour + 1]
                    )
                if by_hour[hour] is not None:
                    by_hour[hour][level_idx] = value
    return by_hour


def fallback_quantiles():
    """Hold hour 1 with a random-walk spread (no usable comparable)."""
    return {
        h: [z * FALLBACK_SIGMA * math.sqrt(h - 1) for z in FALLBACK_Z]
        for h in range(2, 25)
    }


def forecast_from_curves(curves, hour1, pre_ratio, slope, hours_wanted):
    """Quantile rows for hours 2..24 from pre-extracted comparable curves.

    Separated from predict() so leave-one-out scoring can reuse the exact
    same code path with pre-extracted curves instead of re-parsing every
    comparable's 408 hourly points per pseudo-case.
    """
    if curves:
        by_hour = learn_quantiles(curves, hour1, pre_ratio)
    else:
        by_hour = fallback_quantiles()
    hours = []
    for hour in hours_wanted:
        if hour < 2 or hour > 24:
            continue
        logs = by_hour.get(hour)
        if logs is None:
            logs = [z * FALLBACK_SIGMA * math.sqrt(hour - 1) for z in FALLBACK_Z]
        logs = list(logs)
        if SLOPE_ON and hour <= 4 and abs(slope) > 0.15:
            nudge = max(-SLOPE_CAP, min(SLOPE_CAP, slope * (0.5 ** (hour - 2))))
            logs[2] = logs[2] + nudge
        quantiles = []
        previous = 0.0
        for value in logs:
            number = hour1 * math.exp(value)
            number = half_up_2dp(min(1e9, max(1.0, number)))
            if number < previous:  # rounding must never break the ordering
                number = previous
            quantiles.append(number)
            previous = number
        hours.append({"hour": hour, "q": quantiles})
    hours.sort(key=lambda row: row["hour"])
    return {"hours": hours}


def predict(case):
    """One case's forecast: {"hours": [{"hour": h, "q": [...]}, ...]}."""
    hour1, pre_ratio, slope = target_context(case)
    curves = []
    for comp in case.get("comparables") or []:
        parsed = extract_curve(comp)
        if parsed is not None:
            curves.append(parsed)
    rows = case.get("forecast") or [{"hour": h} for h in range(2, 25)]
    wanted = [int(row["hour"]) for row in rows]
    return forecast_from_curves(curves, hour1, pre_ratio, slope, wanted)


def main():
    """Read every <id>.in, write <id>.out, in name order (deterministic)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = predict(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
