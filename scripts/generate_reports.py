"""Build the branded Business Understanding and Data Preparation PDF reports."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from prepare_data import create_numeric_export


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
IMAGE_DIR = ROOT / "images"
REPORT_DIR = ROOT / "reports"
BANNER_PATH = IMAGE_DIR / "wilson-moses-banner.png"
LOGO_PATH = IMAGE_DIR / "wilson-moses-logo.png"

PAGE_WIDTH, PAGE_HEIGHT = A4
DARK = colors.HexColor("#0d0d0d")
OFF_WHITE = colors.HexColor("#f4f0e8")
PAPER = colors.HexColor("#fbfaf7")
GREY = colors.HexColor("#6f6d68")
LIGHT_GREY = colors.HexColor("#dedbd3")
GOLD = colors.HexColor("#c69a4b")
SOFT_GOLD = colors.HexColor("#ead8b4")


def register_fonts() -> None:
    """Embed open fonts so report layout stays stable across PDF viewers."""
    fonts = {
        "BrandSans": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "BrandSansBold": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "BodySerif": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "BodySerifBold": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "BodySerifItalic": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    }
    for name, path in fonts.items():
        pdfmetrics.registerFont(TTFont(name, path))


def prepare_logo_asset() -> None:
    """Extract the current gold WM emblem from the approved brand banner."""
    with PillowImage.open(BANNER_PATH) as banner:
        logo = banner.crop((1180, 45, banner.width, 350))
        logo.save(LOGO_PATH)


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_eyebrow": ParagraphStyle(
            "cover_eyebrow",
            parent=base["Normal"],
            fontName="BrandSansBold",
            fontSize=10,
            leading=14,
            textColor=GOLD,
            spaceAfter=6,
            tracking=1.2,
        ),
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontName="BrandSansBold",
            fontSize=28,
            leading=33,
            textColor=OFF_WHITE,
            alignment=TA_LEFT,
            spaceAfter=12,
        ),
        "cover_subtitle": ParagraphStyle(
            "cover_subtitle",
            parent=base["Normal"],
            fontName="BodySerif",
            fontSize=12,
            leading=18,
            textColor=colors.HexColor("#c7c4bd"),
            spaceAfter=18,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            parent=base["Normal"],
            fontName="BrandSans",
            fontSize=9,
            leading=14,
            textColor=OFF_WHITE,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="BrandSansBold",
            fontSize=16.5,
            leading=20,
            textColor=DARK,
            spaceBefore=4,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="BrandSansBold",
            fontSize=12.2,
            leading=15.5,
            textColor=colors.HexColor("#8b6426"),
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["BodyText"],
            fontName="BodySerif",
            fontSize=10.2,
            leading=15,
            textColor=colors.HexColor("#252422"),
            alignment=TA_LEFT,
            spaceAfter=7,
        ),
        "body_bold": ParagraphStyle(
            "body_bold",
            parent=base["BodyText"],
            fontName="BodySerifBold",
            fontSize=10.2,
            leading=15,
            textColor=colors.HexColor("#252422"),
            spaceAfter=7,
        ),
        "callout": ParagraphStyle(
            "callout",
            parent=base["BodyText"],
            fontName="BodySerifItalic",
            fontSize=10.2,
            leading=15,
            leftIndent=11,
            rightIndent=11,
            borderColor=GOLD,
            borderWidth=0,
            borderPadding=9,
            backColor=colors.HexColor("#f1e8d8"),
            textColor=colors.HexColor("#3a3329"),
            spaceBefore=4,
            spaceAfter=12,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontName="BrandSans",
            fontSize=8.5,
            leading=11,
            textColor=GREY,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "table_header": ParagraphStyle(
            "table_header",
            parent=base["Normal"],
            fontName="BrandSansBold",
            fontSize=8.5,
            leading=11,
            textColor=OFF_WHITE,
        ),
        "table_cell": ParagraphStyle(
            "table_cell",
            parent=base["Normal"],
            fontName="BodySerif",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#252422"),
        ),
        "small": ParagraphStyle(
            "small",
            parent=base["Normal"],
            fontName="BrandSans",
            fontSize=8.3,
            leading=11,
            textColor=GREY,
        ),
    }


def draw_cover(canvas, _doc) -> None:
    canvas.saveState()
    canvas.setFillColor(DARK)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
    canvas.setFillColor(GOLD)
    canvas.rect(18 * mm, 18 * mm, 1.6 * mm, PAGE_HEIGHT - 36 * mm, stroke=0, fill=1)
    canvas.restoreState()


def draw_body_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(PAPER)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
    canvas.setFillColor(DARK)
    canvas.rect(0, PAGE_HEIGHT - 21 * mm, PAGE_WIDTH, 21 * mm, stroke=0, fill=1)
    canvas.drawImage(str(LOGO_PATH), PAGE_WIDTH - 39 * mm, PAGE_HEIGHT - 18 * mm, 28 * mm, 13 * mm, preserveAspectRatio=True, mask="auto")
    canvas.setFont("BrandSansBold", 8.7)
    canvas.setFillColor(OFF_WHITE)
    canvas.drawString(18 * mm, PAGE_HEIGHT - 10.2 * mm, "WILSON MOSES")
    canvas.setFont("BrandSans", 7.6)
    canvas.setFillColor(GOLD)
    canvas.drawString(18 * mm, PAGE_HEIGHT - 15.2 * mm, "DATA SCIENCE FIELD NOTES")
    canvas.setStrokeColor(GOLD)
    canvas.setLineWidth(0.8)
    canvas.line(18 * mm, 15 * mm, PAGE_WIDTH - 18 * mm, 15 * mm)
    canvas.setFont("BrandSans", 7.2)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 9.5 * mm, doc.report_short_title)
    canvas.drawRightString(PAGE_WIDTH - 18 * mm, 9.5 * mm, f"{doc.page - 1:02d}")
    canvas.restoreState()


def report_document(path: Path, short_title: str) -> BaseDocTemplate:
    body_frame = Frame(
        18 * mm,
        17 * mm,
        PAGE_WIDTH - 36 * mm,
        PAGE_HEIGHT - 42 * mm,
        id="body_frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=2 * mm,
        bottomPadding=2 * mm,
    )
    cover_frame = Frame(
        27 * mm,
        22 * mm,
        PAGE_WIDTH - 49 * mm,
        PAGE_HEIGHT - 40 * mm,
        id="cover_frame",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    document = BaseDocTemplate(
        str(path),
        pagesize=A4,
        title=path.stem.replace("_", " ").title(),
        author="Wilson Moses",
        subject="AnalystLab Africa data science portfolio report",
        creator="Wilson Moses portfolio report generator",
    )
    document.report_short_title = short_title
    document.addPageTemplates(
        [
            PageTemplate(id="Cover", frames=[cover_frame], onPage=draw_cover),
            PageTemplate(id="Body", frames=[body_frame], onPage=draw_body_page),
        ]
    )
    return document


def cover_story(styles, number: str, title: str, subtitle: str) -> list:
    banner = Image(str(BANNER_PATH), width=155 * mm, height=38.75 * mm)
    metadata = Table(
        [
            [Paragraph("AUTHOR", styles["small"]), Paragraph("Wilson Moses", styles["cover_meta"])],
            [Paragraph("PROGRAMME", styles["small"]), Paragraph("AnalystLab Africa - Data Science Internship", styles["cover_meta"])],
            [Paragraph("PROJECT", styles["small"]), Paragraph("Loan Approval Data Preparation", styles["cover_meta"])],
            [Paragraph("SUBMISSION", styles["small"]), Paragraph("16 August 2026 | Week 2 | Batch D", styles["cover_meta"])],
        ],
        colWidths=[34 * mm, 105 * mm],
        hAlign="LEFT",
    )
    metadata.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#171717")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#333333")),
                ("INNERGRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#333333")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return [
        Spacer(1, 7 * mm),
        banner,
        Spacer(1, 31 * mm),
        Paragraph(f"DATA SCIENCE FIELD NOTES / {number}", styles["cover_eyebrow"]),
        Paragraph(title, styles["cover_title"]),
        Paragraph(subtitle, styles["cover_subtitle"]),
        HRFlowable(width="100%", thickness=1.2, color=GOLD, spaceBefore=1, spaceAfter=14),
        metadata,
        Spacer(1, 25 * mm),
        Paragraph(
            "Educational portfolio report. This work does not constitute a production lending model or lending policy.",
            styles["cover_subtitle"],
        ),
        NextPageTemplate("Body"),
        PageBreak(),
    ]


def heading(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["h1"])


def subheading(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["h2"])


def body(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["body"])


def callout(text: str, styles) -> Paragraph:
    return Paragraph(text, styles["callout"])


def bullet_list(items: list[str], styles) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["body"]), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
        bulletFontName="BodySerif",
        bulletFontSize=7,
        spaceAfter=8,
    )


def styled_table(rows: list[list[str]], widths: list[float], styles) -> Table:
    formatted = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else styles["table_cell"]
        formatted.append([Paragraph(str(value), style) for value in row])
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), OFF_WHITE),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0eee8")]),
                ("GRID", (0, 0), (-1, -1), 0.35, LIGHT_GREY),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def report_image(path: Path, width: float, styles, caption: str) -> list:
    with PillowImage.open(path) as source:
        ratio = source.height / source.width
    image = Image(str(path), width=width, height=width * ratio)
    return [image, Paragraph(caption, styles["caption"])]


def generate_supporting_visuals(cleaned: pd.DataFrame) -> None:
    numeric = create_numeric_export(cleaned)
    predictors = numeric.drop(columns=["loan_status"])
    target = numeric["loan_status"]
    train_x, _, train_y, _ = train_test_split(
        predictors, target, test_size=0.20, random_state=42, stratify=target
    )
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(train_x, train_y)
    importance = pd.Series(model.feature_importances_, index=train_x.columns).sort_values().tail(10)

    fig, axis = plt.subplots(figsize=(8.6, 5.2), facecolor="#0d0d0d")
    axis.set_facecolor("#0d0d0d")
    axis.barh([label.replace("_", " ") for label in importance.index], importance.values, color="#c69a4b")
    axis.set_title("Exploratory feature importance", color="#f4f0e8", fontsize=16, weight="bold", pad=16)
    axis.set_xlabel("Random Forest importance", color="#a9a7a2")
    axis.tick_params(colors="#f4f0e8", labelsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="x", color="#333333", alpha=0.6)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "feature-importance.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    features = ["applicant_income", "coapplicant_income", "loan_amount", "total_income", "loan_income_ratio"]
    counts = {}
    for column in features:
        q1 = cleaned[column].quantile(0.25)
        q3 = cleaned[column].quantile(0.75)
        spread = q3 - q1
        counts[column] = int(((cleaned[column] < q1 - 1.5 * spread) | (cleaned[column] > q3 + 1.5 * spread)).sum())
    outliers = pd.Series(counts).sort_values()
    fig, axis = plt.subplots(figsize=(8.6, 4.8), facecolor="#0d0d0d")
    axis.set_facecolor("#0d0d0d")
    bars = axis.barh([label.replace("_", " ") for label in outliers.index], outliers.values, color="#c69a4b")
    axis.set_title("Potential IQR outliers retained", color="#f4f0e8", fontsize=16, weight="bold", pad=16)
    axis.set_xlabel("Flagged records", color="#a9a7a2")
    axis.tick_params(colors="#f4f0e8", labelsize=9)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(axis="x", color="#333333", alpha=0.6)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, outliers.values):
        axis.text(value + 0.8, bar.get_y() + bar.get_height() / 2, str(value), va="center", color="#f4f0e8")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "outlier-counts.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def build_business_report(styles) -> None:
    story = cover_story(
        styles,
        "02A",
        "BUSINESS UNDERSTANDING REPORT",
        "Loan Approval Data Preparation - defining the problem before modelling",
    )
    story += [
        heading("Executive summary", styles),
        body(
            "This report frames the loan-approval practice project before any predictive model is developed. The supplied dataset records applicant characteristics, requested loan information, credit-history flags and historical approval outcomes. The project asks whether these records can support an educational binary-classification workflow while preserving responsible-use boundaries.",
            styles,
        ),
        callout(
            "Core question: Can historical application information support a transparent learning exercise that estimates recorded loan-approval outcomes, without presenting the result as a real lending decision system?",
            styles,
        ),
        heading("Business context", styles),
        body(
            "Lending decisions require consistent assessment of risk, affordability and policy requirements. Historical application data can help analysts identify patterns associated with earlier outcomes, but those patterns may also contain undocumented policy choices or social bias. Predictive analysis should therefore support investigation and decision design, not replace responsible lending practice or qualified human oversight.",
            styles,
        ),
        subheading("Problem statement", styles),
        body(
            "The educational lender is assumed to receive many applications containing different forms of applicant and loan information. Manual review alone can be slow and inconsistent. The analytical task is to prepare the supplied data and later test whether information available at application time can estimate the historical target, <b>Loan_Status</b>.",
            styles,
        ),
        subheading("Decision and intended use", styles),
        bullet_list(
            [
                "Primary educational decision: determine whether the available data is sufficiently structured for a later classification experiment.",
                "Potential demonstration use: produce a probability or class estimate for the historical approval outcome.",
                "Excluded use: automatically approving, rejecting or ranking real applicants.",
                "Human role: interpret evidence, review policy and fairness concerns, and challenge model errors before any operational proposal.",
            ],
            styles,
        ),
        heading("Project objectives", styles),
        bullet_list(
            [
                "Inspect data structure, completeness, types and duplication.",
                "Document the target and predictor variables.",
                "Resolve missing values using transparent educational rules.",
                "Create interpretable features for total income and a simple loan-income ratio.",
                "Encode and scale variables where required for later modelling.",
                "Identify redundant features and initial associations without treating them as causal evidence.",
                "Publish reproducible outputs and clearly state responsible-use limitations.",
            ],
            styles,
        ),
        heading("Stakeholders", styles),
        styled_table(
            [
                ["Stakeholder", "Interest", "Required safeguard"],
                ["Applicants", "Fair and understandable treatment", "No automated real-world use; review subgroup errors and harmful proxies"],
                ["Lending analysts", "Consistent evidence and efficient review", "Transparent features, documented limitations and human oversight"],
                ["Risk and compliance teams", "Policy, legal and risk alignment", "Independent validation, governance and auditability"],
                ["Data-science team", "Reliable and reproducible modelling", "Leakage-safe pipelines, versioned data and monitored performance"],
                ["Project reviewer", "Evidence of methodological learning", "Clear separation between completed work and future modelling"],
            ],
            [34 * mm, 57 * mm, 76 * mm],
            styles,
        ),
        heading("Analytical approach", styles),
        body(
            "The later modelling problem is a supervised binary-classification task. The positive and negative class labels are derived from the historical <b>Loan_Status</b> field. The current repository completes the data-preparation phase and uses descriptive statistics, missing-value analysis, correlation and exploratory feature importance to understand the data before a final model is selected.",
            styles,
        ),
        styled_table(
            [
                ["Element", "Definition"],
                ["Unit of analysis", "One loan application"],
                ["Target", "Historical Loan_Status: Y or N"],
                ["Proposed task", "Supervised binary classification"],
                ["Current phase", "Inspection, cleaning, feature engineering and preprocessing"],
                ["Not yet claimed", "Validated predictive performance, deployment or operational lending value"],
            ],
            [43 * mm, 124 * mm],
            styles,
        ),
        heading("Data requirements", styles),
        body(
            "A valid future experiment requires variables that are available at the intended decision time, consistently defined and ethically defensible. The current dataset provides applicant demographics, employment and education categories, income fields, requested loan amount, term, credit-history flag, property area and the historical outcome.",
            styles,
        ),
        bullet_list(
            [
                "Application identifiers must not be treated as ordinary predictors.",
                "Missingness and category definitions must be documented.",
                "Financial units and collection procedures should be confirmed before business interpretation.",
                "Sensitive personal variables require necessity, legal and fairness review.",
                "All transformations used for model evaluation must be fitted on training data only.",
            ],
            styles,
        ),
        heading("Success criteria", styles),
        styled_table(
            [
                ["Dimension", "Provisional criterion"],
                ["Data quality", "Published outputs contain no unexpected missing values, duplicate rows or non-numeric fields in the numeric export"],
                ["Reproducibility", "A clean environment can recreate the processed outputs from the raw CSV"],
                ["Predictive evaluation", "Future work compares against a transparent baseline using class-sensitive metrics, calibration and error analysis"],
                ["Fairness", "Future work evaluates subgroup performance and excludes unjustified sensitive attributes or proxies"],
                ["Communication", "Claims remain limited to the evidence and do not imply production readiness"],
            ],
            [40 * mm, 127 * mm],
            styles,
        ),
        heading("Expected value", styles),
        body(
            "If later validated with suitable data and governance, predictive analysis could help structure risk review, prioritise manual investigation and reveal inconsistencies in historical decisions. In this portfolio project, the immediate value is methodological: creating a transparent, reproducible foundation for responsible model development.",
            styles,
        ),
        heading("Risks, assumptions and limitations", styles),
        bullet_list(
            [
                "The 614-record practice dataset is small and may not represent any actual lending population.",
                "Historical approval labels can reflect undocumented policy choices or bias.",
                "The source does not fully document financial units, collection processes or decision timing.",
                "The loan-income ratio omits expenses, debt, interest rates and repayment capacity.",
                "Associations and feature importance do not establish causation or legitimate policy relevance.",
                "A model trained on this dataset must not be used to make real lending decisions.",
            ],
            styles,
        ),
        heading("Conclusion", styles),
        body(
            "The project is suitable as an educational classification and preprocessing exercise, provided its limitations remain explicit. The next phase should convert the current transformations into a leakage-safe scikit-learn pipeline, establish a baseline, evaluate performance and calibration, and review fairness before considering any operational scenario.",
            styles,
        ),
    ]
    document = report_document(REPORT_DIR / "business_understanding_report.pdf", "LOAN APPROVAL / BUSINESS UNDERSTANDING")
    document.build(story)


def build_preparation_report(styles, raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    missing = raw.isna().sum()
    missing = missing[missing.gt(0)]
    story = cover_story(
        styles,
        "02B",
        "DATA PREPARATION REPORT",
        "Inspection, cleaning, feature engineering, encoding and feature screening",
    )
    story += [
        heading("Executive summary", styles),
        body(
            "This report documents the preparation of 614 historical loan-application records for later machine-learning study. The workflow inspects data quality, treats missing observations, removes the application identifier, creates two analytical features, encodes categorical variables, standardises selected numerical variables, assesses potential outliers and screens features for redundancy and exploratory relevance.",
            styles,
        ),
        callout(
            "Verified result: the cleaned export contains 614 rows and 14 columns; the corrected numeric export contains 614 rows and 14 fully numeric columns. Both contain zero missing values and zero exact duplicate rows.",
            styles,
        ),
        heading("Dataset overview", styles),
        styled_table(
            [
                ["Measure", "Verified value"],
                ["Applications", f"{len(raw):,}"],
                ["Original variables", f"{raw.shape[1]}"],
                ["Approved outcomes", f"{int(raw['Loan_Status'].eq('Y').sum())} (68.73%)"],
                ["Not-approved outcomes", f"{int(raw['Loan_Status'].eq('N').sum())} (31.27%)"],
                ["Exact duplicate rows", f"{int(raw.duplicated().sum())}"],
                ["Missing cells", f"{int(raw.isna().sum().sum())}"],
                ["Fields with missing data", f"{len(missing)}"],
            ],
            [60 * mm, 107 * mm],
            styles,
        ),
        *report_image(IMAGE_DIR / "approval-outcomes.png", 148 * mm, styles, "Figure 1. Historical outcome distribution in the source dataset."),
        heading("Initial inspection", styles),
        body(
            "Inspection covered shape, column names, data types, summary statistics, target balance, missingness and exact duplicate rows. The application identifier is unique but does not provide an interpretable applicant characteristic, so it is removed before analytical preparation.",
            styles,
        ),
        *report_image(IMAGE_DIR / "missing-values.png", 152 * mm, styles, "Figure 2. Missing-value counts before treatment."),
        styled_table(
            [["Field", "Missing", "Treatment"]]
            + [[field, str(int(value)), "Median" if field == "LoanAmount" else "Mode"] for field, value in missing.items()],
            [67 * mm, 30 * mm, 70 * mm],
            styles,
        ),
        heading("Missing-value treatment", styles),
        body(
            "Low-frequency categorical missing values were filled using each field's most common category. <b>LoanAmount</b> was filled using the median because it is less sensitive to extreme values than the mean. <b>Loan_Amount_Term</b> and <b>Credit_History</b> were filled using their modes because both are discrete repeated fields in this dataset.",
            styles,
        ),
        callout(
            "These are transparent educational choices, not universal defaults. In future model evaluation, every imputation rule must be learned from training data only and compared with suitable alternatives.",
            styles,
        ),
        heading("Feature engineering", styles),
        styled_table(
            [
                ["Transformation", "Definition", "Reason"],
                ["Remove Loan_ID", "Drop the unique identifier", "Avoid learning an arbitrary application code"],
                ["total_income", "applicant_income + coapplicant_income", "Represent combined recorded household income"],
                ["loan_income_ratio", "loan_amount / total_income", "Create a simple affordability-oriented indicator"],
                ["snake_case names", "Standardise all analytical column names", "Improve readability and consistency in Python"],
            ],
            [38 * mm, 57 * mm, 72 * mm],
            styles,
        ),
        body(
            "No application had zero total income, so the ratio calculation did not create infinite values. The ratio remains a limited proxy because it excludes expenses, interest, debt and repayment capacity.",
            styles,
        ),
        heading("Categorical encoding", styles),
        styled_table(
            [
                ["Field", "Encoding"],
                ["gender", "Male = 1; Female = 0"],
                ["married", "Yes = 1; No = 0"],
                ["education", "Graduate = 1; Not Graduate = 0"],
                ["self_employed", "Yes = 1; No = 0"],
                ["credit_history", "Convert the 0/1 flag to integer"],
                ["dependents", "Convert 3+ to 3, then store as integer"],
                ["property_area", "One-hot encode with one reference category removed"],
                ["loan_status", "Y = 1; N = 0"],
            ],
            [60 * mm, 107 * mm],
            styles,
        ),
        body(
            "Binary coding is used for compactness in this learning exercise. The numeric values do not imply that one social category is inherently better than another. Sensitive fields require necessity and fairness review before any real modelling proposal.",
            styles,
        ),
        heading("Numerical scaling", styles),
        body(
            "StandardScaler was applied to applicant income, co-applicant income, loan amount, loan term, total income and loan-income ratio. The corrected numeric export has means approximately equal to zero and population standard deviations equal to one for the retained scaled fields.",
            styles,
        ),
        callout(
            "The published numeric file records the full-sample preprocessing exercise. It must not be used to claim unbiased predictive performance because the scaler was fitted before a final evaluation split. Future work should place imputation, encoding and scaling inside a training-only pipeline.",
            styles,
        ),
        heading("Outlier assessment", styles),
        body(
            "Boxplots and the 1.5 x IQR rule were used to flag unusual values. The flagged observations were retained because the extreme income and loan values appeared plausible rather than clear data-entry errors. Removing valid extremes without domain evidence could erase important cases and distort the population represented by the practice data.",
            styles,
        ),
        *report_image(IMAGE_DIR / "outlier-counts.png", 150 * mm, styles, "Figure 3. Potential outliers identified by the IQR rule and retained for analysis."),
        heading("Feature screening", styles),
        body(
            "Feature screening combined correlation, target relationships, business logic and exploratory Random Forest importance. Applicant income and engineered total income showed a correlation of approximately 0.89. Applicant income was removed from the published numeric export to reduce redundancy while retaining the combined-income feature.",
            styles,
        ),
        *report_image(IMAGE_DIR / "feature-importance.png", 150 * mm, styles, "Figure 4. Exploratory Random Forest importance after the published feature reduction."),
        body(
            "Credit history remains the strongest initial signal in the exploratory result. This importance is neither causal evidence nor proof that the field is sufficient, fair or suitable for real lending policy. Feature importance can change across splits, algorithms and preprocessing choices.",
            styles,
        ),
        heading("Published outputs", styles),
        styled_table(
            [
                ["Output", "Shape", "Contract"],
                ["loan_prediction_cleaned.csv", "614 x 14", "Human-readable, engineered, no missing values, no duplicate rows"],
                ["loan_prediction_ml_ready.csv", "614 x 14", "Fully numeric, encoded, scaled full-sample demonstration artifact"],
            ],
            [58 * mm, 28 * mm, 81 * mm],
            styles,
        ),
        body(
            "The original notebook mistakenly exported the earlier pre-encoding dataframe under the ML-ready filename. The standardized repository corrects this by exporting the encoded and scaled dataframe and validating that every resulting column is numeric.",
            styles,
        ),
        heading("Reproducibility and verification", styles),
        bullet_list(
            [
                "The raw CSV is preserved unchanged in data/raw.",
                "scripts/prepare_data.py recreates both processed datasets.",
                "The notebook uses repository-relative paths and executes all code cells successfully.",
                "The preparation script validates shapes, missingness and numeric data types before publication.",
                "README charts and reports can be regenerated from versioned scripts.",
            ],
            styles,
        ),
        heading("Limitations and responsible use", styles),
        bullet_list(
            [
                "The practice dataset is small and does not establish representativeness.",
                "Mode imputation can reinforce majority categories and reduce variation.",
                "Historical approval labels may contain policy or social bias.",
                "Sensitive demographic variables require legal, ethical and fairness review.",
                "The full-sample numeric artifact is not suitable for unbiased performance evaluation.",
                "Feature importance and correlation describe association, not causation.",
                "No production model, deployment or real-world lending decision is claimed.",
            ],
            styles,
        ),
        heading("Conclusion", styles),
        body(
            "The data-preparation phase produced transparent, reproducible and internally consistent outputs suitable for continued educational modelling. The next phase should rebuild these operations in a scikit-learn Pipeline and ColumnTransformer, split before fitting any transformation, establish a transparent baseline, evaluate class-sensitive and calibration metrics, and examine subgroup errors before drawing practical conclusions.",
            styles,
        ),
    ]
    document = report_document(REPORT_DIR / "data_preprocessing_report.pdf", "LOAN APPROVAL / DATA PREPARATION")
    document.build(story)


def main() -> None:
    register_fonts()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    prepare_logo_asset()
    raw = pd.read_csv(DATA_DIR / "raw" / "loan_approval_train.csv")
    cleaned = pd.read_csv(DATA_DIR / "processed" / "loan_prediction_cleaned.csv")
    generate_supporting_visuals(cleaned)
    styles = build_styles()
    build_business_report(styles)
    build_preparation_report(styles, raw, cleaned)
    print("Created two branded Wilson Moses Data Science Field Notes reports.")


if __name__ == "__main__":
    main()
