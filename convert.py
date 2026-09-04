"""
PDF → DITA conversion pipeline for BNY Hackathon.

Usage (single file):
    python convert.py report.pdf -o dita_out

Usage (batch):
    python convert.py doc1.pdf doc2.pdf doc3.pdf -o dita_out
    python convert.py *.pdf -o dita_out

Requirements:
    pip install pymupdf pillow lxml groq

Environment:
    export GROQ_API_KEY="your-key"
"""

import os
import re
import json
import argparse
import io
from pathlib import Path

from lxml import etree
import fitz                     # PyMuPDF
from PIL import Image
from groq import Groq

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL  = "llama-3.3-70b-versatile"

def _get_client() -> Groq:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY environment variable is not set.\n"
            "  export GROQ_API_KEY='your-key-here'"
        )
    return Groq(api_key=key)

# DOCTYPE strings required by DITA-OT for validation
DOCTYPES = {
    "concept": (
        '<!DOCTYPE concept PUBLIC "-//OASIS//DTD DITA Concept//EN" "concept.dtd">'
    ),
    "task": (
        '<!DOCTYPE task PUBLIC "-//OASIS//DTD DITA General Task//EN" "generalTask.dtd">'
    ),
    "reference": (
        '<!DOCTYPE reference PUBLIC "-//OASIS//DTD DITA Reference//EN" "reference.dtd">'
    ),
    "map": (
        '<!DOCTYPE map PUBLIC "-//OASIS//DTD DITA Map//EN" "map.dtd">'
    ),
}

# Max image dimensions / size per hackathon spec
IMG_MAX_WIDTH  = 1000   # px
IMG_MAX_BYTES  = 200 * 1024  # 200 KB


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class DitaPipeline:
    def __init__(self, output_dir: str = "dita_out"):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.images_path = self.output_path / "images"
        self.images_path.mkdir(exist_ok=True)

        self.product_name  = "ABC Accounting"   # replaced by keyref
        self.global_img_idx = 1
        self.all_topic_files: list[str] = []    # accumulated across batch

    # ------------------------------------------------------------------
    # Text helpers
    # ------------------------------------------------------------------
    def clean_grammar(self, text: str) -> str:
        """Replace Latin abbreviations and fix known typos."""
        if not text:
            return ""
        subs = [
            (re.compile(r'\bi\.e\.,?\s*', re.I), 'that is, '),
            (re.compile(r'\be\.g\.,?\s*', re.I), 'for example, '),
            (re.compile(r'\betc\.\s*',    re.I), ''),
            (re.compile(r'\bvia\b',       re.I), 'through'),
            (re.compile(r'\bliqidity\b',  re.I), 'liquidity'),
            (re.compile(r'\bwit the\b',   re.I), 'within the'),
        ]
        for pattern, repl in subs:
            text = pattern.sub(repl, text)
        return text.strip()

    def set_text_with_keyref(self, el, text: str):
        """
        Write text into *el*, replacing every occurrence of the product name
        with a <ph keyref="product-name"/> element so DITA can resolve it via
        the keydef in the map.
        """
        if not text:
            return
        parts = re.split(r'\bABC\s+Accounting\b|\bABC\b', text)
        el.text = self.clean_grammar(parts[0])
        for part in parts[1:]:
            ph = etree.SubElement(el, "ph")
            ph.set("keyref", "product-name")
            ph.tail = self.clean_grammar(part)

    # ------------------------------------------------------------------
    # Page filtering
    # ------------------------------------------------------------------
    def is_toc_or_title_page(self, text: str) -> bool:
        """Skip table-of-contents and near-empty cover pages."""
        lines = text.splitlines()
        dot_lines = sum(1 for l in lines if re.search(r'\.{4,}', l))
        return (
            dot_lines >= 2
            or len(text) < 200
            or bool(re.search(r'table of contents', text, re.I))
        )

    # ------------------------------------------------------------------
    # Image extraction
    # ------------------------------------------------------------------
    def extract_images_from_page(self, page, doc) -> list[dict]:
        """
        Extract all raster images from a PyMuPDF page.
        Resize to ≤1000 px wide and compress to ≤200 KB (PNG).
        Returns list of dicts: {filename, alt_text}.
        """
        results = []
        image_list = page.get_images(full=True)

        for xref, *_ in image_list:
            try:
                base_image = doc.extract_image(xref)
                img_bytes  = base_image["image"]

                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

                # Resize if wider than target
                if pil_img.width > IMG_MAX_WIDTH:
                    ratio  = IMG_MAX_WIDTH / pil_img.width
                    new_h  = int(pil_img.height * ratio)
                    pil_img = pil_img.resize(
                        (IMG_MAX_WIDTH, new_h), Image.LANCZOS
                    )

                # Save as PNG, reducing quality until under 200 KB
                # PNG is lossless so we use optimize; if still too big,
                # convert to JPEG-mode by dropping alpha.
                out_buf = io.BytesIO()
                pil_img.save(out_buf, format="PNG", optimize=True)

                if out_buf.tell() > IMG_MAX_BYTES:
                    # Flatten alpha → white, save as JPEG inside .png name
                    flat = Image.new("RGB", pil_img.size, (255, 255, 255))
                    flat.paste(pil_img, mask=pil_img.split()[3])
                    out_buf = io.BytesIO()
                    quality = 85
                    flat.save(out_buf, format="JPEG", quality=quality, optimize=True)
                    while out_buf.tell() > IMG_MAX_BYTES and quality > 30:
                        quality -= 10
                        out_buf = io.BytesIO()
                        flat.save(out_buf, format="JPEG", quality=quality, optimize=True)

                fname = f"img_{self.global_img_idx:04d}.png"
                self.global_img_idx += 1
                (self.images_path / fname).write_bytes(out_buf.getvalue())

                results.append(
                    {"filename": f"images/{fname}", "alt_text": f"Figure {self.global_img_idx - 1}"}
                )
            except Exception as exc:
                print(f"  [!] Image extraction failed (xref={xref}): {exc}")

        return results

    # ------------------------------------------------------------------
    # Hyperlink extraction
    # ------------------------------------------------------------------
    def extract_links_from_page(self, page) -> list[dict]:
        """Return list of {text, uri} dicts for all external links on the page."""
        links = []
        for link in page.get_links():
            uri = link.get("uri", "")
            if uri and uri.startswith("http"):
                # Try to find display text near the link rect
                rect  = fitz.Rect(link["from"])
                words = page.get_text("words", clip=rect)
                text  = " ".join(w[4] for w in words).strip() or uri
                links.append({"text": text, "uri": uri})
        return links

    # ------------------------------------------------------------------
    # AI call
    # ------------------------------------------------------------------
    def get_structured_data(self, raw_content: str, images: list, links: list) -> dict | None:
        """
        Call the LLM and return a parsed JSON dict with a 'topics' list.
        """
        images_hint = ""
        if images:
            images_hint = (
                "\n\nIMAGES IN THIS SECTION (already extracted to disk):\n"
                + "\n".join(f'  - filename: {i["filename"]}' for i in images)
                + "\nInclude each image in the appropriate topic body_elements as "
                  '{{"type": "image", "filename": "<filename>", "alt": "<meaningful alt text>"}}.'
            )

        links_hint = ""
        if links:
            links_hint = (
                "\n\nHYPERLINKS FOUND IN THIS SECTION:\n"
                + "\n".join(f'  - text: "{l["text"]}", uri: {l["uri"]}' for l in links)
                + "\nInclude them in body_elements as "
                  '{{"type": "xref", "text": "<link text>", "href": "<uri>"}}.'
            )

        prompt = f"""You are an expert technical writer converting documentation to DITA XML.

RULES:
1. TOPIC DETECTION: Split content into separate topics where the type changes
   (concept = explanatory, task = step-by-step, reference = tables/data).
2. QUALITY: Remove ALL passive voice. Fix grammar. Keep technical meaning intact.
3. CODE BLOCKS: Return any code as {{"type": "codeblock", "text": "..."}}.
4. SHORT DESC: One clear sentence summarising the topic purpose.
5. KEYWORDS: 3-4 relevant keywords per topic.
6. Do NOT invent content not present in the source text.

Return ONLY a JSON object, no markdown fences, no preamble:
{{
  "topics": [
    {{
      "type": "concept" | "task" | "reference",
      "title": "string",
      "parent_title": "string or null",
      "shortdesc": "string",
      "keywords": ["kw1", "kw2"],
      "body_elements": [
        {{"type": "p",         "text": "..."}},
        {{"type": "codeblock", "text": "..."}},
        {{"type": "table",     "headers": ["Col1"], "rows": [["val"]]}},
        {{"type": "image",     "filename": "images/img_0001.png", "alt": "description"}},
        {{"type": "xref",      "text": "Link text", "href": "https://..."}}
      ],
      "steps": [
        {{"cmd": "Do this.", "info": "Expected result."}}
      ]
    }}
  ]
}}

HIERARCHY RULE: Set "parent_title" to the title of the topic that is this topic's logical
parent in the document structure, matching the source PDF section/subsection hierarchy.
Set "parent_title" to null for top-level topics only. Example: a setup task sits under
its overview concept; a reference settings table sits under the task it supports.
{images_hint}{links_hint}

SOURCE TEXT:
{raw_content}
"""
        try:
            resp = _get_client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=6000,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            return json.loads(raw)
        except Exception as exc:
            print(f"  [!] API/JSON error: {exc}")
            return None

    # ------------------------------------------------------------------
    # DITA XML builder
    # ------------------------------------------------------------------
    def _serialize(self, root, topic_type: str) -> bytes:
        """
        Serialize an lxml element to bytes with the correct XML declaration
        AND DOCTYPE required by DITA-OT.
        """
        xml_body = etree.tostring(root, pretty_print=True, encoding="unicode")
        doctype  = DOCTYPES[topic_type]
        full_doc = f'<?xml version="1.0" encoding="UTF-8"?>\n{doctype}\n{xml_body}'
        return full_doc.encode("utf-8")

    def create_topic(self, topic_data: dict, topic_id: str) -> bytes:
        topic_type = topic_data.get("type", "concept")
        if topic_type not in ("task", "concept", "reference"):
            topic_type = "concept"

        root = etree.Element(topic_type, id=topic_id)
        etree.SubElement(root, "title").text = topic_data.get("title", "Untitled")

        # <shortdesc>
        # FIX 2: sanitise shortdesc wording so it matches the actual topic type.
        raw_sd = topic_data.get("shortdesc", "")
        if raw_sd:
            if topic_type == "concept" and re.match("this task", raw_sd, re.I):
                raw_sd = re.sub("^this task", "This topic", raw_sd, flags=re.I)
            elif topic_type == "reference" and re.match("this task", raw_sd, re.I):
                raw_sd = re.sub("^this task", "This reference", raw_sd, flags=re.I)
            sd = etree.SubElement(root, "shortdesc")
            self.set_text_with_keyref(sd, raw_sd)

        # <prolog> / <keywords>
        kws_list = topic_data.get("keywords", [])
        if kws_list:
            prolog   = etree.SubElement(root, "prolog")
            metadata = etree.SubElement(prolog, "metadata")
            kws      = etree.SubElement(metadata, "keywords")
            for kw in kws_list:
                etree.SubElement(kws, "keyword").text = kw

        # body
        body_tag = {"task": "taskbody", "concept": "conbody", "reference": "refbody"}[topic_type]
        body = etree.SubElement(root, body_tag)

        # Split body_elements: for tasks, codeblocks are NOT valid inside
        # <context>. Separate them so we can place them in <example> after steps.
        all_body = topic_data.get("body_elements", [])
        if topic_type == "task":
            context_elems  = [e for e in all_body if e.get("type") != "codeblock"]
            example_elems  = [e for e in all_body if e.get("type") == "codeblock"]
        else:
            context_elems  = all_body
            example_elems  = []

        # For tasks, put non-codeblock body_elements inside <context>
        parent_el = body
        if topic_type == "task" and context_elems:
            parent_el = etree.SubElement(body, "context")
        # FIX: Wszystko wewnątrz 'reference' musi być owinięte w <section>
        elif topic_type == "reference" and context_elems:
            parent_el = etree.SubElement(body, "section")

        for elem in context_elems:
            etype = elem.get("type")

            if etype == "p" and elem.get("text"):
                pel = etree.SubElement(parent_el, "p")
                self.set_text_with_keyref(pel, elem["text"])

            elif etype == "codeblock" and elem.get("text"):
                # <codeblock> is not a valid direct child of <refbody> — wrap in <section>.
                target = parent_el
                if topic_type == "reference" and parent_el.tag == "refbody":
                    target = etree.SubElement(parent_el, "section")
                cb = etree.SubElement(target, "codeblock")
                cb.text = elem["text"]

            elif etype == "table":
                self._build_table(parent_el, elem)

            elif etype == "image":
                fig = etree.SubElement(parent_el, "fig")
                img_el = etree.SubElement(fig, "image")
                img_el.set("href", elem.get("filename", ""))
                img_el.set("width", "1000px")
                alt_el = etree.SubElement(img_el, "alt")
                alt_el.text = elem.get("alt", "Figure")

            elif etype == "xref":
                # FIX 3: skip xrefs with empty href — they are broken links
                href = elem.get("href", "").strip()
                if not href:
                    print(f"  [!] Skipping <xref> with empty href (text: '{elem.get('text','')}')")
                    continue
                p_el = etree.SubElement(parent_el, "p")
                xr   = etree.SubElement(p_el, "xref")
                xr.set("href", href)
                xr.set("format", "html")
                xr.set("scope", "external")
                xr.text = elem.get("text", href)

        # steps (task only)
        if topic_type == "task" and topic_data.get("steps"):
            steps_el = etree.SubElement(body, "steps")
            for s in topic_data["steps"]:
                step = etree.SubElement(steps_el, "step")
                cmd  = etree.SubElement(step, "cmd")
                self.set_text_with_keyref(cmd, s.get("cmd", "Perform this step."))
                if s.get("info"):
                    info = etree.SubElement(step, "info")
                    self.set_text_with_keyref(info, s["info"])

        # FIX 1: place any codeblocks deferred from <context> into <example>
        # <example> is valid in taskbody and accepts <codeblock>.
        if example_elems:
            example_el = etree.SubElement(body, "example")
            for elem in example_elems:
                cb = etree.SubElement(example_el, "codeblock")
                cb.text = elem["text"]

        return self._serialize(root, topic_type)

    def _build_table(self, parent_el, elem: dict):
        """Build a DITA <table>/<tgroup> from a dict with headers/rows."""
        headers = elem.get("headers", [])
        rows    = elem.get("rows", [])
        cols    = max(len(headers), max((len(r) for r in rows), default=0), 1)

        table  = etree.SubElement(parent_el, "table")
        tgroup = etree.SubElement(table, "tgroup", cols=str(cols))

        # colspec elements (required by strict DITA-OT)
        for i in range(1, cols + 1):
            etree.SubElement(tgroup, "colspec", colname=f"col{i}")

        if headers:
            thead = etree.SubElement(tgroup, "thead")
            row   = etree.SubElement(thead, "row")
            for h in headers:
                self.set_text_with_keyref(etree.SubElement(row, "entry"), str(h))

        tbody = etree.SubElement(tgroup, "tbody")
        for r in rows:
            row = etree.SubElement(tbody, "row")
            for cell in r:
                self.set_text_with_keyref(etree.SubElement(row, "entry"), str(cell))
            # Pad short rows so column count matches
            for _ in range(cols - len(r)):
                etree.SubElement(row, "entry")

    # ------------------------------------------------------------------
    # Image injection helper
    # ------------------------------------------------------------------
    def _inject_images(self, fname: str, images: list[dict]):
        """
        Parse an already-written .dita file and append <fig><image>
        elements for any images not yet referenced, then rewrite the file.
        Works for concept (conbody), reference (refbody), and task (taskbody).
        """
        import re as _re
        raw = (self.output_path / fname).read_bytes().decode("utf-8")

        # Strip DOCTYPE so lxml can parse it, remember it for re-insertion
        doctype_match = _re.search(r'<!DOCTYPE[^>]*>', raw)
        doctype_str   = doctype_match.group(0) if doctype_match else ""
        clean         = _re.sub(r'<!DOCTYPE[^>]*>', '', raw)

        tree = etree.fromstring(clean.encode("utf-8"))

        # Find the body element (conbody / refbody / taskbody)
        body = None
        for tag in ("conbody", "refbody", "taskbody"):
            body = tree.find(tag)
            if body is not None:
                break
                
        if body is None:
            return  # can't inject — bail silently

        # FIX: Zabezpieczenie przed bezpośrednim wstrzykiwaniem obrazków do <refbody>
        target_el = body
        if body.tag == "refbody":
            target_el = body.find("section")
            if target_el is None:
                target_el = etree.SubElement(body, "section")

        for img in images:
            fig    = etree.SubElement(target_el, "fig")
            img_el = etree.SubElement(fig, "image")
            img_el.set("href",  img["filename"])
            img_el.set("width", "1000px")
            alt_el = etree.SubElement(img_el, "alt")
            alt_el.text = img.get("alt_text", "Figure")

        xml_body = etree.tostring(tree, pretty_print=True, encoding="unicode")
        full_doc = '<?xml version="1.0" encoding="UTF-8"?>\n' + doctype_str + '\n' + xml_body
        (self.output_path / fname).write_bytes(full_doc.encode("utf-8"))

    # ------------------------------------------------------------------
    # Page grouping
    # ------------------------------------------------------------------
    def group_pages(self, doc) -> list[dict]:
        """Split PDF pages into logical sections based on numbered headings."""
        sections: list[dict] = []
        current_texts: list[str] = []
        current_title = "Introduction"
        heading_re = re.compile(r"^\d+[\.\d]*\s+.{5,}", re.MULTILINE)

        for page in doc:
            text = page.get_text().strip()
            if not text or self.is_toc_or_title_page(text):
                continue

            m = heading_re.search(text)
            page_title = m.group(0).strip() if m else None

            if page_title and page_title != current_title:
                if current_texts:
                    sections.append({
                        "title": current_title,
                        "text": "\n".join(current_texts),
                    })
                current_title  = page_title
                current_texts  = [text]
            else:
                current_texts.append(text)

        if current_texts:
            sections.append({"title": current_title, "text": "\n".join(current_texts)})

        return sections

    # ------------------------------------------------------------------
    # Per-PDF processing
    # ------------------------------------------------------------------
    def process_pdf(self, pdf_path: str | Path) -> list[str]:
        """
        Convert a single PDF.  Returns list of generated .dita filenames
        (relative to output_dir).
        """
        pdf_path = Path(pdf_path)
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print('='*60)

        doc = fitz.open(str(pdf_path))

        # Collect per-page images and links keyed by page index
        page_images: dict[int, list] = {}
        page_links:  dict[int, list] = {}
        for page_num, page in enumerate(doc):
            imgs = self.extract_images_from_page(page, doc)
            lnks = self.extract_links_from_page(page)
            if imgs: page_images[page_num] = imgs
            if lnks: page_links[page_num]  = lnks

        sections = self.group_pages(doc)
        doc.close()

        generated: list[str] = []
        # {fname: {"title": str, "parent_title": str|None}} for map hierarchy
        topic_meta: list[dict] = []
        topic_idx = len(self.all_topic_files) + 1  # global counter across batch
        
        seen_titles: set[str] = set()              # dedup guard
        title_mapping: dict[str, str] = {}         # original -> deduplicated map

        for section in sections:
            print(f"\n  Section: {section['title'][:60]}")

            # Attach images/links for pages likely belonging to this section
            all_imgs  = [i for imgs in page_images.values() for i in imgs]
            all_links = [l for lnks in page_links.values()  for l in lnks]

            data = self.get_structured_data(section["text"], all_imgs, all_links)
            if not data or "topics" not in data:
                print("  [!] No valid 'topics' returned from API.")
                continue

            for topic_data in data["topics"]:
                # Tasks without steps → demote to concept
                if topic_data.get("type") == "task" and not topic_data.get("steps"):
                    print(f"  [!] Task '{topic_data.get('title')}' has no steps → demoted to concept.")
                    topic_data["type"] = "concept"

                has_body  = bool(topic_data.get("body_elements"))
                has_steps = bool(topic_data.get("steps"))
                if not has_body and not has_steps:
                    print(f"  [!] Skipping empty topic: {topic_data.get('title')}")
                    continue

                # Deduplicate titles — append a counter suffix if already seen
                raw_title = topic_data.get("title", "Untitled")
                original_title = raw_title
                
                if raw_title in seen_titles:
                    suffix = 2
                    while f"{raw_title} ({suffix})" in seen_titles:
                        suffix += 1
                    new_title = f"{raw_title} ({suffix})"
                    print(f"  [!] Duplicate title '{raw_title}' → renamed to '{new_title}'")
                    topic_data["title"] = new_title
                    raw_title = new_title
                
                seen_titles.add(raw_title)
                title_mapping[original_title] = raw_title
                
                # Fix map hierarchy breakage: update parent_title if it was deduplicated earlier
                parent_title = topic_data.get("parent_title")
                if parent_title and parent_title in title_mapping:
                    topic_data["parent_title"] = title_mapping[parent_title]

                topic_id = f"topic_{topic_idx:03d}"
                xml_bytes = self.create_topic(topic_data, topic_id)
                fname = f"{topic_id}.dita"
                (self.output_path / fname).write_bytes(xml_bytes)

                t_type = topic_data.get("type", "?")
                t_title = raw_title[:50]
                print(f"  ✓  {fname}  [{t_type}]  {t_title}")

                generated.append(fname)
                topic_meta.append({
                    "fname":        fname,
                    "title":        raw_title,
                    "parent_title": topic_data.get("parent_title"),
                })
                topic_idx += 1

        # FIX 4: auto-inject any images the AI ignored into the first
        # topic generated for this document (must be concept or reference).
        # Moved outside the section loop to avoid infinite duplication!
        all_imgs = [i for imgs in page_images.values() for i in imgs]
        if all_imgs and generated:
            # Collect which image filenames already appear in written files
            used = set()
            for gf in generated:
                txt = (self.output_path / gf).read_text(encoding="utf-8")
                for img in all_imgs:
                    if img["filename"] in txt:
                        used.add(img["filename"])

            missing = [i for i in all_imgs if i["filename"] not in used]
            if missing:
                # Find the first concept (or any) topic in this section
                target_fname = None
                for gf in generated:
                    txt = (self.output_path / gf).read_text(encoding="utf-8")
                    if "concept.dtd" in txt or "reference.dtd" in txt:
                        target_fname = gf
                        break
                if target_fname is None:
                    target_fname = generated[0]

                self._inject_images(target_fname, missing)
                for img in missing:
                    print(f"  ✓  Auto-injected {img['filename']} → {target_fname}")

        # Store meta alongside file list so generate_map can build hierarchy
        if not hasattr(self, "all_topic_meta"):
            self.all_topic_meta: list[dict] = []
        self.all_topic_meta.extend(topic_meta)

        return generated

    # ------------------------------------------------------------------
    # Map generation
    # ------------------------------------------------------------------
    def generate_map(self, topic_files: list[str]):
        """
        Generate main.ditamap with nested <topicref> hierarchy derived from
        the parent_title fields collected during process_pdf calls.
        Falls back to a flat list if no hierarchy metadata is available.
        """
        if not topic_files:
            print("\n[!] No DITA files generated — skipping map.")
            return

        root = etree.Element("map")
        etree.SubElement(root, "title").text = "Converted Documentation"

        # Product-name keydef
        keydef = etree.SubElement(root, "keydef", keys="product-name")
        tm     = etree.SubElement(keydef, "topicmeta")
        kws    = etree.SubElement(tm, "keywords")
        etree.SubElement(kws, "keyword").text = self.product_name

        meta = getattr(self, "all_topic_meta", [])

        if meta:
            # Build title → lxml element map so children can find their parent
            title_to_el: dict[str, etree._Element] = {}

            for item in meta:
                tr = etree.Element("topicref", href=item["fname"])
                title_to_el[item["title"]] = tr

            for item in meta:
                tr     = title_to_el[item["title"]]
                parent = item.get("parent_title")
                if parent and parent in title_to_el:
                    title_to_el[parent].append(tr)
                else:
                    # Top-level: attach directly to map root
                    root.append(tr)
        else:
            # Fallback: flat list (no hierarchy metadata)
            for fname in topic_files:
                etree.SubElement(root, "topicref", href=fname)

        xml_body = etree.tostring(root, pretty_print=True, encoding="unicode")
        doctype  = DOCTYPES["map"]
        full_doc = f'<?xml version="1.0" encoding="UTF-8"?>\n{doctype}\n{xml_body}'

        (self.output_path / "main.ditamap").write_bytes(full_doc.encode("utf-8"))
        print(
            f"\n✓ Done. Generated {len(topic_files)} topic(s) + main.ditamap"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="PDF → DITA conversion pipeline (BNY Hackathon)"
    )
    parser.add_argument(
        "pdfs",
        nargs="+",
        metavar="PDF",
        help="One or more PDF files to convert (supports shell globs)",
    )
    parser.add_argument(
        "-o", "--output",
        default="dita_out",
        help="Output directory (default: dita_out)",
    )
    args = parser.parse_args()

    pipeline = DitaPipeline(output_dir=args.output)

    for pdf in args.pdfs:
        pdf_path = Path(pdf)
        if not pdf_path.is_file():
            print(f"[!] Not a file, skipping: {pdf}")
            continue
        topics = pipeline.process_pdf(pdf_path)
        pipeline.all_topic_files.extend(topics)

    pipeline.generate_map(pipeline.all_topic_files)


if __name__ == "__main__":
    main()