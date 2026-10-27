"""Compare old and latest AI against updated dietician relevance labels.

The merged workbook supplies human and latest AI labels; the old results folder
supplies old AI labels matched by participant and screenshot ID.
"""
import argparse
from pathlib import Path

import pandas as pd
from screenshot_filtering.export_dieticians import ROOT, identifier, build_lookup, annotation


def read_merged(path, sheet="Coding"):
    data = pd.read_excel(path, sheet_name=sheet, header=1, dtype=str, keep_default_na=False)
    columns = {"User": "participant", "Id_number": "screenshot",
               "Is this a relevant ad?": "human", "AI ad detected": "detected",
               "AI relevance": "relevance", "AI processing status": "status"}
    missing = set(columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing merged workbook columns: {sorted(missing)}")
    data = data[list(columns)].rename(columns=columns).copy()
    for column in ("participant", "screenshot"):
        data[column] = data[column].map(identifier)
    data = data[data.participant.str.fullmatch(r"\d+") & data.screenshot.str.fullmatch(r"\d+")].copy()
    if data.empty or data.duplicated(["participant", "screenshot"]).any():
        raise ValueError("Labels are empty or contain duplicate participant/screenshot pairs")
    for column in ("human", "detected", "relevance", "status"):
        data[column] = data[column].str.strip().str.upper()
    data["prediction"] = ""
    ok = data.status.eq("OK")
    data.loc[ok & data.detected.eq("NO") & data.relevance.isin(["", "NO"]), "prediction"] = "NO"
    data.loc[ok & data.detected.eq("YES"), "prediction"] = data.relevance
    return data.set_index(["participant", "screenshot"])


def scores(human, predicted):
    tp = int(((human == "YES") & (predicted == "YES")).sum())
    fn = int(((human == "YES") & (predicted == "NO")).sum())
    fp = int(((human == "NO") & (predicted == "YES")).sum())
    tn = int(((human == "NO") & (predicted == "NO")).sum())
    total = tp + fn + fp + tn
    kappa = ac1 = "N/A"
    if total:
        observed = (tp + tn) / total
        human_positive = (tp + fn) / total
        ai_positive = (tp + fp) / total
        chance_kappa = (human_positive * ai_positive
                        + (1 - human_positive) * (1 - ai_positive))
        # Binary, unweighted AC1: pooled category prevalence across both raters.
        # Gwet (2008), doi:10.1348/000711006X126600.
        pooled_positive = (human_positive + ai_positive) / 2
        chance_ac1 = 2 * pooled_positive * (1 - pooled_positive)
        if chance_kappa < 1:
            kappa = f"{(observed - chance_kappa) / (1 - chance_kappa):.3f}"
        ac1 = f"{(observed - chance_ac1) / (1 - chance_ac1):.3f}"
    def percent(numerator, denominator):
        return f"{100 * numerator / denominator:.1f}%" if denominator else "N/A"
    return {"Correct relevant-ad screenshots": tp, "Missed relevant-ad screenshots": fn,
            "False positives": fp, "Correct negatives": tn,
            "Recall (real ads found)": percent(tp, tp + fn),
            "Precision (AI positives correct)": percent(tp, tp + fp),
            "Agreement": percent(tp + tn, tp + tn + fp + fn),
            "F1": percent(2 * tp, 2 * tp + fp + fn),
            "Cohen's kappa": kappa, "Gwet's AC1": ac1}


def old_predictions(keys, results_dir, validation_dir):
    lookup = build_lookup(results_dir, validation_dir)
    rows = []
    for key in keys:
        result = annotation(key, lookup)
        prediction = ""
        if result["AI processing status"] == "OK":
            if result["AI ad detected"] == "NO":
                prediction = "NO"
            elif result["AI ad detected"] == "YES":
                prediction = result["AI relevance"]
        rows.append(dict(prediction=prediction, status=result["AI processing status"]))
    return pd.DataFrame(rows, index=keys)


def compare(data, old):
    if not data.index.equals(old.index):
        raise ValueError("Old predictions must match the merged participant/screenshot index")
    human_clear = data.human.isin(["YES", "NO"])
    included = human_clear.copy()
    print(f"Human-labelled rows: {len(data)} across {data.index.get_level_values(0).nunique()} participants")
    print(f"Excluded uncertain/blank human labels: {int((~human_clear).sum())}")
    for name, run in (("Old / try 2", old), ("Latest / merged workbook", data)):
        usable = run.status.eq("OK") & run.prediction.isin(["YES", "NO"])
        included &= usable
        print(f"{name} processing statuses: {run.status.value_counts().to_dict()}")
        print(f"Unresolved among clear human labels: {int((human_clear & ~usable).sum())}")
    print(f"Evaluated screenshots usable in BOTH runs: {int(included.sum())}")
    human = data.loc[included, "human"]
    table = pd.DataFrame({"Old vs dieticians": scores(human, old.loc[included, "prediction"]),
                          "Latest vs dieticians": scores(human, data.loc[included, "prediction"])})
    print(table.to_string())
    was_correct = old.loc[included, "prediction"].eq(human)
    now_correct = data.loc[included, "prediction"].eq(human)
    print(f"Corrected screenshots: {int((~was_correct & now_correct).sum())}")
    print(f"Previously correct, now incorrect: {int((was_correct & ~now_correct).sum())}")
    print("Positive means detected AND relevant in Round 2. Counts are screenshots, not independent ad groups.")
    print("Failed, missing, inconsistent and unsure AI results are excluded, never counted as negatives.")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/aggregated validation/dieticians_with_ai_final_NLS.xlsx")
    parser.add_argument("--old", type=Path, default=ROOT / "data/aggregated validation (try 2)")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/validation 1")
    parser.add_argument("--sheet", default="Coding")
    args = parser.parse_args()
    print(f"Merged source (updated human + latest AI): {args.input}\nOld AI: {args.old}")
    data = read_merged(args.input, args.sheet)
    compare(data, old_predictions(data.index, args.old, args.validation))


if __name__ == "__main__":
    main()
