from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


REPO_URL = "https://github.com/tanishq170205/Fashionpedia.git"
OUTPUT_PATH = "Glance_ML_Assignment_Tanishq.pdf"


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="TitleCenter",
            parent=base["Title"],
            alignment=1,
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=4,
        )
    )
    base.add(
        ParagraphStyle(
            name="SubtitleCenter",
            parent=base["Normal"],
            alignment=1,
            fontSize=11,
            leading=14,
            spaceAfter=16,
        )
    )
    base.add(
        ParagraphStyle(
            name="Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=10,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=colors.HexColor("#111827"),
            spaceBefore=6,
            spaceAfter=3,
        )
    )
    base.add(
        ParagraphStyle(
            name="BodyTextClean",
            parent=base["BodyText"],
            fontSize=9.6,
            leading=13,
            spaceAfter=6,
        )
    )
    return base


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def approach_table(style):
    data = [
        ["Approach", "Strength", "Tradeoff", "Best use"],
        [
            "Plain CLIP",
            "Very simple; fast ANN search; strong zero-shot baseline.",
            "Weak at binding colors to garments; scene and attributes compete in one vector.",
            "Broad vibe search where strict garment/color composition is not required.",
        ],
        [
            "Supervised detector + classifiers",
            "Precise when taxonomy is fixed and labels are available.",
            "Limited zero-shot vocabulary; new garment terms need new labels or training data.",
            "Catalog search with a controlled product taxonomy.",
        ],
        [
            "Open-vocabulary retrieve + rerank",
            "Keeps zero-shot recall while checking region-level fashion constraints.",
            "More moving parts and slower than a single CLIP lookup.",
            "Fashion queries with colors, garments, context, and composition.",
        ],
    ]
    table = Table(data, colWidths=[3.0 * cm, 4.1 * cm, 5.2 * cm, 4.6 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.3),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_story():
    s = styles()
    story = [
        p("Glance ML Internship Assignment", s["TitleCenter"]),
        p("Multimodal Fashion and Context Retrieval", s["SubtitleCenter"]),
        p("Project Overview", s["Section"]),
        p(
            "The assignment asks for an image retrieval system that understands both fashion "
            "attributes and scene context. A plain CLIP baseline is a useful starting point, "
            "but it often fails on compositional prompts such as 'red tie and white shirt' "
            "because the whole query is compressed into one vector. I built a two-stage "
            "retrieve-and-rerank system that keeps the fast zero-shot benefits of CLIP while "
            "adding garment-level checks for color, category, and person association.",
            s["BodyTextClean"],
        ),
        p("1. Approaches and Tradeoffs", s["Section"]),
        approach_table(s),
        Spacer(1, 8),
        p("2. Chosen Approach", s["Section"]),
        p("Indexer", s["Subsection"]),
        p(
            "The indexer processes each image offline. First, it stores a full-image embedding "
            "from Marqo FashionCLIP (hf-hub:Marqo/marqo-fashionCLIP). Then Grounding DINO "
            "detects garments and person boxes. For every garment crop, the pipeline stores a "
            "local CLIP embedding and a dominant RGB color extracted with K-means. Garments are "
            "assigned to the most-overlapping person box, so later retrieval can check whether "
            "multiple requested garments belong to the same person. ChromaDB stores one vector "
            "document per image, with region metadata serialized alongside the full-image embedding.",
            s["BodyTextClean"],
        ),
        p("Retriever", s["Subsection"]),
        p(
            "At query time, Llama 3.3 70B through Groq parses the text into garment/color "
            "constraints and an optional setting phrase. The raw query is embedded with "
            "FashionCLIP and ChromaDB returns the top 300 candidates. These candidates are "
            "reranked with a weighted combination of global similarity, garment/color attribute "
            "score, and setting similarity.",
            s["BodyTextClean"],
        ),
        p("How it handles fashion queries", s["Subsection"]),
        p(
            "For garment matching, the reranker compares query label embeddings against stored "
            "region embeddings rather than relying only on detector labels. It expands synonyms, "
            "so 'raincoat' can match regions that look like coats, jackets, or anoraks. Color "
            "phrases are normalized before RGB distance checks; for example, 'bright yellow' "
            "becomes 'yellow' and 'navy blue' becomes 'navy'. For multi-garment prompts, the "
            "score is computed per person instance. This prevents a red tie on one person and a "
            "white shirt on another from being treated as a full compositional match.",
            s["BodyTextClean"],
        ),
        p("3. Codebase Link", s["Section"]),
        p(
            f"Repository: {REPO_URL}<br/><br/>"
            "The repository separates indexing, retrieval, evaluation, and the optional FastAPI "
            "demo app. The main files are indexer/main.py for offline indexing, retriever/main.py "
            "for command-line search, retriever/reranker.py for the compositional logic, and "
            "eval/run_eval.py for the five benchmark queries.",
            s["BodyTextClean"],
        ),
        p("4. Limitations and Future Work", s["Section"]),
        p("Dataset coverage", s["Subsection"]),
        p(
            "Fashionpedia val_test2020 is mostly runway, editorial, and fashion-event imagery. "
            "Office interiors and park-bench scenes are not common, so context-heavy queries are "
            "partly limited by dataset coverage rather than only retrieval logic. The code still "
            "handles the query structure, but no retrieval method can return many correct office "
            "or park images if they are not present in the corpus.",
            s["BodyTextClean"],
        ),
        p("Adding locations and weather", s["Subsection"]),
        p(
            "The practical extension is metadata enrichment. If GPS and timestamps are available, "
            "index city, region, season, temperature, and weather through EXIF and a historical "
            "weather API. If metadata is missing, add image-based classifiers for coarse location "
            "type and weather class. The query parser can then emit optional fields such as "
            "location='Tokyo' or weather='rainy', and ChromaDB can apply these as metadata filters "
            "before ANN retrieval. This is preferable to hoping CLIP can infer real geography or "
            "weather from pixels alone.",
            s["BodyTextClean"],
        ),
        p("Improving precision", s["Subsection"]),
        p(
            "Precision can be improved in four directions. First, fine-tune the region encoder on "
            "Fashionpedia train masks and attributes so garment concepts are better separated. "
            "Second, replace hand-set fusion weights with a small learned reranker trained from "
            "relevance judgments. Third, add pattern and material recognition for striped, plaid, "
            "leather, denim, or silk prompts. Fourth, replace bounding-box person association with "
            "pose or segmentation for crowded and occluded scenes.",
            s["BodyTextClean"],
        ),
        p("Scalability", s["Subsection"]),
        p(
            "The architecture scales because only stage 1 touches the full corpus. With one "
            "million images, HNSW still returns a fixed candidate set and stage 2 only reranks "
            "those candidates. The main production change would be storing garment regions in a "
            "separate region-level index instead of embedding all region vectors inside image metadata.",
            s["BodyTextClean"],
        ),
    ]
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawCentredString(A4[0] / 2, 0.8 * cm, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    doc = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="Glance ML Assignment - Multimodal Fashion Retrieval",
    )
    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(f"PDF created: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
