"""
parse_medquad.py
Walks the MedQuAD-master folder tree, parses every Document XML file,
and produces one row per Question/Answer pair with all associated metadata.

Usage:
    python parse_medquad.py /path/to/MedQuAD-master output.csv
"""
import os
import sys
import glob
import xml.etree.ElementTree as ET
import pandas as pd


def parse_document(path, folder_name):
    """Parse a single MedQuAD Document XML file into a list of row dicts (one per QA pair)."""
    rows = []
    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        return rows, f"PARSE_ERROR: {path} -> {e}"

    root = tree.getroot()

    # Two schema variants appear in a small number of legacy files:
    #   Standard:  <Document id source url> <Focus> <FocusAnnotations> <QAPairs><QAPair><Question qid qtype>...
    #   Legacy:    <doc docid corpus url> <doctitle-focus> <umls> <qaPairs><pair><question qid qtype>...
    #   Legacy2:   <DiseaseFile fid source url> <Focus> <UMLS> <QAPairs><QAPair><Question qid qtype>...
    if root.tag == "Document":
        doc_id = root.attrib.get("id", "")
        source = root.attrib.get("source", "")
    elif root.tag == "doc":
        doc_id = root.attrib.get("docid", "")
        source = root.attrib.get("corpus", "")
    elif root.tag == "DiseaseFile":
        doc_id = root.attrib.get("fid", "")
        source = root.attrib.get("source", "")
    else:
        return rows, f"UNEXPECTED_ROOT: {path} -> <{root.tag}>"

    url = root.attrib.get("url", "")

    focus_el = root.find("Focus")
    if focus_el is None:
        focus_el = root.find("doctitle-focus")
    focus = focus_el.text.strip() if focus_el is not None and focus_el.text else ""

    category = ""
    cuis = []
    semantic_types = []
    semantic_group = ""
    synonyms = []

    fa = root.find("FocusAnnotations")
    umls = fa.find("UMLS") if fa is not None else root.find("UMLS")
    if umls is None:
        umls = root.find("umls")  # legacy lowercase, flat CUI/semanticType text

    if fa is not None:
        cat_el = fa.find("Category")
        if cat_el is not None and cat_el.text:
            category = cat_el.text.strip()
        synonyms = [s.text.strip() for s in fa.findall("./Synonyms/Synonym") if s.text]

    if umls is not None:
        cuis = [c.text.strip() for c in umls.findall("./CUIs/CUI") if c.text] or \
               ([umls.find("cui").text.strip()] if umls.find("cui") is not None and umls.find("cui").text else []) or \
               ([umls.find("CUI").text.strip()] if umls.find("CUI") is not None and umls.find("CUI").text else [])
        semantic_types = [s.text.strip() for s in umls.findall("./SemanticTypes/SemanticType") if s.text] or \
               ([umls.find("semanticType").text.strip()] if umls.find("semanticType") is not None and umls.find("semanticType").text else []) or \
               ([umls.find("SemanticType").text.strip()] if umls.find("SemanticType") is not None and umls.find("SemanticType").text else [])
        sg_el = umls.find("SemanticGroup")
        if sg_el is None:
            sg_el = umls.find("semanticGroup")
        if sg_el is not None and sg_el.text:
            semantic_group = sg_el.text.strip()

    qapairs = root.find("QAPairs")
    if qapairs is None:
        qapairs = root.find("qaPairs")
    if qapairs is None:
        return rows, None  # document with no QA pairs (rare but valid)

    pair_els = qapairs.findall("QAPair") or qapairs.findall("pair")

    for qap in pair_els:
        pid = qap.attrib.get("pid", "")
        q_el = qap.find("Question")
        if q_el is None:
            q_el = qap.find("question")
        a_el = qap.find("Answer")
        if a_el is None:
            a_el = qap.find("answer")

        question_text = (q_el.text or "").strip() if q_el is not None else ""
        qid = q_el.attrib.get("qid", "") if q_el is not None else ""
        qtype = q_el.attrib.get("qtype", "") if q_el is not None else ""
        answer_text = (a_el.text or "").strip() if a_el is not None else ""

        rows.append({
            "source_folder": folder_name,
            "source_site": source,
            "document_id": doc_id,
            "url": url,
            "focus": focus,
            "focus_category": category,
            "umls_cuis": "|".join(cuis),
            "umls_semantic_types": "|".join(semantic_types),
            "semantic_group": semantic_group,
            "synonyms": "|".join(synonyms),
            "pair_id": pid,
            "question_id": qid,
            "qtype": qtype,
            "question": question_text,
            "answer": answer_text,
        })

    return rows, None


def main(root_dir, out_csv):
    all_rows = []
    errors = []

    folders = sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        xml_files = glob.glob(os.path.join(folder_path, "**", "*.xml"), recursive=True)
        for fp in xml_files:
            rows, err = parse_document(fp, folder)
            all_rows.extend(rows)
            if err:
                errors.append(err)

    df = pd.DataFrame(all_rows)

    # basic cleanup
    df["answer"] = df["answer"].fillna("")
    df["question"] = df["question"].fillna("")
    df["answer_is_empty"] = df["answer"].str.strip().eq("")
    df["question_len_words"] = df["question"].str.split().str.len()
    df["answer_len_words"] = df["answer"].str.split().str.len()
    df["num_cuis"] = df["umls_cuis"].apply(lambda x: 0 if x == "" else len(x.split("|")))
    df["num_semantic_types"] = df["umls_semantic_types"].apply(lambda x: 0 if x == "" else len(x.split("|")))
    df["num_synonyms"] = df["synonyms"].apply(lambda x: 0 if x == "" else len(x.split("|")))

    df.to_csv(out_csv, index=False)

    print(f"Parsed {len(df):,} QA pairs from {len(folders)} folders.")
    print(f"Saved to: {out_csv}")
    if errors:
        print(f"\n{len(errors)} parse warnings/errors (first 10 shown):")
        for e in errors[:10]:
            print(" -", e)

    return df


if __name__ == "__main__":
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "MedQuAD-master"
    out_csv = sys.argv[2] if len(sys.argv) > 2 else "medquad_full.csv"
    main(root_dir, out_csv)
