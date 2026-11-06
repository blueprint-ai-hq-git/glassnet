"""lit_selfimprove.py — verify the self-improvement / self-distillation literature via the
arXiv API. Every paper cited must come back from a real lookup with matching title and authors.
Nothing is cited from memory or from a search-result snippet.
"""
import re
import sys
import urllib.parse
import urllib.request

UA = {"User-Agent": "glassnet-litcheck/0.1"}


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45).read().decode()


def arxiv_search(q, n=6):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{q}", "start": 0, "max_results": n, "sortBy": "relevance"})
    return _get(url)


def entries(xml):
    out = []
    for e in re.findall(r"<entry>.*?</entry>", xml, re.S):
        aid = re.search(r"<id>(.*?)</id>", e, re.S)
        ti = re.search(r"<title>(.*?)</title>", e, re.S)
        au = re.findall(r"<name>(.*?)</name>", e, re.S)
        pub = re.search(r"<published>(.*?)</published>", e, re.S)
        out.append({"id": aid.group(1).strip() if aid else "?",
                    "title": " ".join(ti.group(1).split()) if ti else "?",
                    "authors": au[:3],
                    "published": pub.group(1)[:10] if pub else "?"})
    return out


def by_id(aid):
    xml = _get(f"http://export.arxiv.org/api/query?id_list={aid}")
    e = entries(xml)
    return e[0] if e else None


QUERIES = [
    ("star", "STaR self-taught reasoner bootstrapping reasoning rationales"),
    ("collapse", "model collapse recursive training generative models"),
    ("bornagain", "born again neural networks self distillation"),
    ("surpass", "student surpasses teacher self distillation knowledge"),
    ("opsd", "on-policy self-distillation language model privileged teacher"),
]

print("=" * 100)
print("LITERATURE CHECK — self-improvement / self-distillation, verified via the arXiv API")
print("=" * 100)
found = {}
for tag, q in QUERIES:
    print(f"\n--- query [{tag}]: {q}")
    try:
        es = entries(arxiv_search(q, 6))
    except Exception as exc:
        print(f"    QUERY FAILED: {exc}")
        continue
    for e in es:
        print(f"    {e['id'].split('/abs/')[-1]:<14} {e['published']}  "
              f"{e['title'][:74]}")
        print(f"        {', '.join(e['authors'])}")
    found[tag] = es

# direct id verification of the classic anchors, so nothing is cited from memory
print("\n" + "=" * 100)
print("DIRECT ID VERIFICATION of the anchor papers")
print("=" * 100)
ANCHORS = {"STaR (Zelikman et al.)": "2203.14465",
           "Model collapse (Shumailov et al.)": "2305.17493",
           "Born-again networks (Furlanello et al.)": "1805.04770",
           "Self-distillation (Zhang et al.)": "1905.08094"}
for label, aid in ANCHORS.items():
    try:
        e = by_id(aid)
        if e:
            print(f"  {label:<42} arXiv:{aid}")
            print(f"      title  : {e['title'][:88]}")
            print(f"      authors: {', '.join(e['authors'])}   published {e['published']}")
        else:
            print(f"  {label:<42} arXiv:{aid}  -> NO RECORD FOUND (do not cite)")
    except Exception as exc:
        print(f"  {label:<42} arXiv:{aid}  -> LOOKUP FAILED: {exc}")
