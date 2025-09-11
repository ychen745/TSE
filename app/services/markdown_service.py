import markdown as md
import bleach
from bleach.linkifier import Linker

_ALLOWED_TAGS = bleach.sanitizer.ALLOWED_TAGS.union({
    "p", "pre", "code",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "ul", "ol", "li",
    "strong", "em",
    "table", "thead", "tbody", "tr", "th", "td",
    "hr", "blockquote"
})

_ALLOWED_ATTRS = {"*": ["class"], "a": ["href","title","name","target"]}

def md_to_safe_html(md_text: str) -> str:
    html = md.markdown(
        md_text or "",
        extensions=["fenced_code", "tables", "codehilite"]
    )
    clean = bleach.clean(html, tags=_ALLOWED_TAGS, attributes=_ALLOWED_ATTRS, strip=True)
    return Linker().linkify(clean)