"""
End-to-end test of the Job Market Intelligence System on REAL resumes pulled
from the internet (HuggingFace: Sachinkelenjaguri/Resume_dataset, 962 real CVs).

Replicates the production path used by streamlit_app.py / the FastAPI backend:
    resume_text  -> extract_skills()         (scripts.extract_skills)
                 -> extract_experience_hint() (mirrors streamlit logic)
                 -> predict_category()         (tech-11 ensemble classifier)
                 -> recommend_jobs()           (scripts.recommend, hybrid)

Part A: classifier accuracy on real CVs vs their dataset category.
Part B: full resume -> job recommendation for a few representative CVs.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.extract_skills import extract_skills
from scripts.predict import predict_category
from scripts.recommend import recommend_jobs, get_available_roles

CSV = Path(__file__).resolve().parent / "resume_dataset.csv"

# dataset Category -> project role (tech-11 for the classifier; also a valid
# recommender target role). Only mappable tech categories are tested for the
# classifier accuracy read.
CAT2ROLE = {
    "Data Science": "Data Scientist",
    "DevOps Engineer": "DevOps Engineer",
    "Python Developer": "Software Engineer",
    "Java Developer": "Software Engineer",
    "DotNet Developer": "Software Engineer",
    "ETL Developer": "Data Engineer",
    "Database": "Data Engineer",
    "Hadoop": "Data Engineer",
    "Network Security Engineer": "Cybersecurity Analyst",
    "Business Analyst": "Business Analyst",
    "Testing": "QA Engineer",
    "Automation Testing": "QA Engineer",
}


def experience_hint(text: str) -> str:
    """Mirror of streamlit_app.extract_experience_hint (years detection)."""
    compact = re.sub(r"\s+", " ", text or "").strip()
    low = compact.lower()
    years = [int(n) for n in re.findall(r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", low)]
    if years:
        return f"{max(years)} years of experience"
    if any(t in low for t in ("fresh graduate", "recent graduate", "entry level")):
        return "Entry-level / recent graduate experience"
    return ""


def main() -> None:
    df = pd.read_csv(CSV)
    roles = set(get_available_roles())
    print(f"Loaded {len(df)} real resumes | recommender exposes {len(roles)} roles\n")

    # ---- Part A: classifier on real CVs (N per mapped category) -------------
    PER_CAT = 5
    rows, correct, total = [], 0, 0
    print("=" * 78)
    print("PART A — tech-11 ensemble classifier on REAL resumes")
    print("=" * 78)
    for cat, role in CAT2ROLE.items():
        sample = df[df["Category"] == cat].head(PER_CAT)
        for _, r in sample.iterrows():
            res = predict_category(r["Resume"])
            ok = res["predicted_category"] == role
            correct += ok; total += 1
            rows.append({"true_cat": cat, "expected_role": role,
                         "predicted": res["predicted_category"],
                         "confidence": round(res["confidence"], 3), "match": ok})
    by_cat = {}
    for x in rows:
        by_cat.setdefault((x["true_cat"], x["expected_role"]), []).append(x)
    for (cat, role), xs in by_cat.items():
        hits = sum(x["match"] for x in xs)
        preds = ", ".join(sorted({x["predicted"] for x in xs}))
        print(f"  {cat:26s} (-> {role:21s}) {hits}/{len(xs)} correct | preds seen: {preds}")
    print(f"\n  Classifier accuracy on {total} real CVs: {correct}/{total} = {correct/total:.1%}")

    # ---- Part B: full resume -> recommendation pipeline --------------------
    print("\n" + "=" * 78)
    print("PART B — resume -> job recommendation (hybrid strategy)")
    print("=" * 78)
    demo_cats = ["Data Science", "DevOps Engineer", "ETL Developer",
                 "Network Security Engineer", "Java Developer"]
    report = []
    for cat in demo_cats:
        r = df[df["Category"] == cat].iloc[0]
        resume = r["Resume"]
        skills = extract_skills(resume)
        exp = experience_hint(resume) or resume[:300]
        role = CAT2ROLE[cat]
        pred = predict_category(resume)
        recs = recommend_jobs(skills, [role], exp, strategy="hybrid", top_k=3)
        print(f"\n--- REAL CV: category='{cat}'  (resume {len(resume)} chars) ---")
        print(f"  detected skills ({len(skills)}): {', '.join(skills[:12])}"
              + (" ..." if len(skills) > 12 else ""))
        print(f"  experience hint: {exp[:80]!r}")
        print(f"  classifier says: {pred['predicted_category']} ({pred['confidence']:.2f})"
              f"   | recommending within role: {role}")
        print(f"  top {len(recs)} job matches:")
        for j in recs:
            print(f"    - job {str(j['job_id'])[:12]} | score {j['score']:.3f} "
                  f"| have {len(j.get('matched_skills', []))} / "
                  f"gap {len(j.get('missing_skills', []))} skills")
            print(f"        reason: {j['reason'][:100]}")
            if j.get("missing_skills"):
                print(f"        skill gap: {', '.join(j['missing_skills'][:6])}")
        report.append({"cv_category": cat, "detected_skills": skills,
                       "experience": exp, "classifier_prediction": pred["predicted_category"],
                       "classifier_confidence": round(pred["confidence"], 3),
                       "recommend_role": role,
                       "recommendations": [{"job_id": str(j["job_id"]), "score": j["score"],
                                            "matched_skills": j.get("matched_skills", []),
                                            "missing_skills": j.get("missing_skills", []),
                                            "reason": j["reason"]} for j in recs]})

    out = Path(__file__).resolve().parent / "real_cv_results.json"
    out.write_text(json.dumps({"classifier": {"accuracy": correct / total, "n": total,
                                               "detail": rows},
                               "recommendations": report}, indent=2, default=str), "utf-8")
    print(f"\nFull results saved -> {out}")


if __name__ == "__main__":
    main()
