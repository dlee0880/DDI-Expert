from typing import Literal

import pandas as pd

PromptStyle = Literal[
    "multimodal_context",
    "mechanism_stub",
    "interaction_explanation",
    "human_explanation",
    "regression",
]


def build_prompt(row: pd.Series, style: PromptStyle) -> str:
    drug_a = row.get("Drug_A", row.get("DrugName", ""))
    drug_b = row.get("Drug_B", row.get("DrugName_2", ""))

    if style == "multimodal_context":
        return f"""
### Drug Names ###
#Drug1 Name: {drug_a}
#Drug2 Name: {drug_b}

### Proteins associated with each drug ###
Proteins associated with #Drug1: {row.get('trim_dicts_aggregated_A', '')}
Proteins associated with #Drug2: {row.get('trim_dicts_aggregated_B', '')}

### Diseases associated with each drug ###
Diseases associated with #Drug1: {row.get('disease_A', '')}
Diseases associated with #Drug2: {row.get('disease_B', '')}

### Pathways associated with each drug ###
Pathways associated with #Drug1: {row.get('pathway_A', '')}
Pathways associated with #Drug2: {row.get('pathway_B', '')}

### Side effects associated with each drug ###
Side effects associated with #Drug1: {row.get('se_A', '')}
Side effects associated with #Drug2: {row.get('se_B', '')}

### Drug Structures ###
#Drug1 Structure: {row.get('smiles_A', '')}
#Drug2 Structure: {row.get('smiles_B', '')}
""".strip()

    if style == "mechanism_stub":
        return f"""
### Drug Names ###
#Drug1 Name: {drug_a}
#Drug2 Name: {drug_b}

### Question ###
Explain the mechanism of pharmacokinetic drug interaction between the above two drugs.

### Explanation ###
#Drug1 may
""".strip()

    if style == "interaction_explanation":
        return f"""
### Question ###
Explain if there would be a drug interaction between {drug_a} and {drug_b}.
If so, explain the mechanism.

### Answer and Explanation ###
""".strip()

    if style == "human_explanation":
        return f"""
### Question ###
Explain whether there would be a significant drug interaction between {drug_a} and {drug_b},
and if so, explain the mechanism of pharmacokinetic drug interaction between {drug_a} and {drug_b}.

Now, explain.
### Explanation ###
""".strip()

    if style == "regression":
        return f"""
### Question ###
Estimate the fold change of the area under the concentration time curve due to the drug interaction between {drug_a} and {drug_b}.

### Answer ###
""".strip()

    raise ValueError(f"Unsupported prompt style: {style}")
