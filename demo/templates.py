"""Synthetic CMR report templates, one per class.

Written from scratch for this demo. They are not derived from, and do not
paraphrase, any real report. No names, dates, or identifiers.
"""

SYNTHETIC_NOTE = "SYNTHETIC EXAMPLE. NOT A REAL PATIENT."

TEMPLATES = {
    "ischemic": f"""{SYNTHETIC_NOTE}

PROCEDURE: Cardiac MRI with gadolinium contrast.

INDICATION: Exertional chest pain. Assess myocardial viability.

TECHNIQUE: Cine SSFP, T1 mapping, and phase-sensitive inversion recovery imaging after intravenous gadolinium.

FINDINGS:
Left ventricle: Mildly dilated with akinesis of the basal to mid anterior wall. LVEF 41%.
Right ventricle: Normal size and systolic function.
Late gadolinium enhancement: Subendocardial delayed enhancement of the basal to mid anterior and anteroseptal walls, involving 50 to 75 percent of wall thickness, in the LAD territory.

IMPRESSION:
1. Prior myocardial infarction in the LAD territory with limited viability of the anterior wall.
2. Mildly reduced left ventricular systolic function.""",

    "non-ischemic": f"""{SYNTHETIC_NOTE}

PROCEDURE: Cardiac MRI with gadolinium contrast.

INDICATION: Palpitations and an abnormal ECG.

TECHNIQUE: Cine SSFP, T2 mapping, and phase-sensitive inversion recovery imaging after intravenous gadolinium.

FINDINGS:
Left ventricle: Normal size with mild global hypokinesis. LVEF 50%.
Right ventricle: Normal size and systolic function.
Late gadolinium enhancement: Mid-wall delayed enhancement of the basal inferolateral wall, with a separate subepicardial focus in the mid lateral wall. The subendocardium is spared.

IMPRESSION:
1. Mid-wall and subepicardial fibrosis of the lateral wall, in a pattern most consistent with prior myocarditis.
2. Low-normal left ventricular systolic function.""",

    "mixed": f"""{SYNTHETIC_NOTE}

PROCEDURE: Cardiac MRI with gadolinium contrast.

INDICATION: Heart failure with reduced ejection fraction.

TECHNIQUE: Cine SSFP and phase-sensitive inversion recovery imaging after intravenous gadolinium.

FINDINGS:
Left ventricle: Moderately dilated with inferior akinesis and global hypokinesis. LVEF 32%.
Right ventricle: Mildly reduced systolic function.
Late gadolinium enhancement: Transmural delayed enhancement of the basal to mid inferior wall in the RCA territory. Separately, there is linear mid-wall delayed enhancement of the basal to mid septum, which does not follow a coronary territory.

IMPRESSION:
1. Prior inferior myocardial infarction in the RCA territory.
2. Additional septal mid-wall fibrosis, a separate non-ischemic pattern.
3. Moderately reduced left ventricular systolic function.""",

    "no-scar": f"""{SYNTHETIC_NOTE}

PROCEDURE: Cardiac MRI with gadolinium contrast.

INDICATION: Dyspnea on exertion. Evaluate for cardiomyopathy.

TECHNIQUE: Cine SSFP, T1 mapping, and phase-sensitive inversion recovery imaging after intravenous gadolinium.

FINDINGS:
Left ventricle: Normal size, wall thickness, and systolic function. LVEF 62%.
Right ventricle: Normal size and systolic function.
Late gadolinium enhancement: No late gadolinium enhancement is seen in the left ventricular myocardium.

IMPRESSION:
1. Normal biventricular size and systolic function.
2. No late gadolinium enhancement.""",

    "not-specified": f"""{SYNTHETIC_NOTE}

PROCEDURE: Cardiac MRI with gadolinium contrast.

INDICATION: Ventricular ectopy.

TECHNIQUE: Cine SSFP and phase-sensitive inversion recovery imaging after intravenous gadolinium.

FINDINGS:
Left ventricle: Normal size with mildly reduced systolic function. LVEF 48%.
Right ventricle: Normal size and systolic function.
Late gadolinium enhancement: A small focus of delayed enhancement is present in the basal inferior wall.

IMPRESSION:
1. Small focal myocardial scar in the basal inferior wall. Its cause is uncertain on this study.
2. Mildly reduced left ventricular systolic function.""",
}

# Extra synthetic case: negated scar words. In the paper, a keyword rule that ignored negation turned
# correct no-scar answers like this one into not-specified, so the LLM results use the model's own answer.
EXTRA_CASES = {
    "No scar, negated scar words": f"""{SYNTHETIC_NOTE}

PROCEDURE: Cardiac MRI with gadolinium contrast.

INDICATION: Family history of cardiomyopathy.

TECHNIQUE: Cine SSFP and phase-sensitive inversion recovery imaging after intravenous gadolinium.

FINDINGS:
Left ventricle: Normal size, wall thickness, and systolic function. LVEF 59%.
Right ventricle: Normal size and systolic function.
Late gadolinium enhancement: No myocardial delayed enhancement is present. There is no evidence of scar or fibrosis.

IMPRESSION:
1. Normal biventricular size and systolic function.
2. No evidence of myocardial scar.""",
}
