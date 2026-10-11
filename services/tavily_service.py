"""
Tavily web search service with domain filtering.
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from langchain_tavily import TavilySearch

logger = logging.getLogger(__name__)

TRUSTED_DOMAINS: Set[str] = {
    "philips.com", "usa.philips.com", "fda.gov", "nih.gov",
    "pubmed.ncbi.nlm.nih.gov", "who.int", "reuters.com", "bloomberg.com",
    "medtechdive.com", "healthcaredive.com", "fiercehealthcare.com",
    "healthcareitnews.com", "mobihealthnews.com", "medicaldevice-network.com",
    "massdevice.com", "beckershospitalreview.com", "nature.com",
    "sciencedirect.com", "grandviewresearch.com", "globenewswire.com",
    "prnewswire.com", "businesswire.com",
}

COMPANY_DOMAINS: Dict[str, Set[str]] = {
    "mindray": {"mindray.com", "mindrayna.com"},
    "siemens": {"siemens-healthineers.com"},
    "ge healthcare": {"gehealthcare.com"},
    "ge": {"gehealthcare.com"},
    "medtronic": {"medtronic.com"},
    "masimo": {"masimo.com"},
    "nihon kohden": {"nihonkohden.com"},
    "drager": {"draeger.com"},
    "spacelabs": {"spacelabshealthcare.com"},
    "hamilton medical": {"hamilton-medical.com"},
    "getinge": {"getinge.com"},
    "baxter": {"baxter.com"},
    "bd": {"bd.com"},
    "stryker": {"stryker.com"},
    "boston scientific": {"bostonscientific.com"},
    "abbott": {"abbott.com"},
    "edwards lifesciences": {"edwards.com"},
    "hologic": {"hologic.com"},
    "intuitive surgical": {"intuitive.com"},
    "hill-rom": {"hillrom.com"},
    "hillrom": {"hillrom.com"},
    "edan": {"edan.com"},
    "comen": {"comen.com"},
    "zoll": {"zoll.com"},
    "canon medical": {"global.medical.canon"},
    "fujifilm": {"fujifilm.com"},
    "nonin": {"nonin.com"},
}

BLOCKED_DOMAINS: Set[str] = {
    "linkedin.com", "medium.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "reddit.com", "quora.com", "youtube.com",
    "slideshare.net", "scribd.com", "glassdoor.com", "indeed.com",
    "pinterest.com", "tiktok.com",
}


@dataclass
class TavilyConfig:
    api_key: str = ""
    max_results: int = 6
    topic: str = "general"
    search_depth: str = "advanced"
    use_trusted_domains_only: bool = False
    filter_blocked_domains: bool = True
    trusted_domains: Set[str] = field(default_factory=lambda: set(TRUSTED_DOMAINS))
    blocked_domains: Set[str] = field(default_factory=lambda: set(BLOCKED_DOMAINS))

    @classmethod
    def from_env(cls) -> "TavilyConfig":
        return cls(
            api_key=os.getenv("TAVILY_API_KEY", ""),
            use_trusted_domains_only=os.getenv("TAVILY_USE_TRUSTED_DOMAINS_ONLY", "false").lower() == "true",
            filter_blocked_domains=os.getenv("TAVILY_FILTER_BLOCKED_DOMAINS", "true").lower() == "true",
        )


@dataclass
class WebSearchResult:
    url: str
    title: str
    content: str
    score: Optional[float] = None

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "content": self.content, "score": self.score}


class TavilyServiceError(Exception):
    pass


class TavilyService:
    """Web search with domain filtering — trusted, blocked, and company-specific."""

    def __init__(self, config: Optional[TavilyConfig] = None):
        self.config = config or TavilyConfig.from_env()
        if self.config.api_key:
            os.environ["TAVILY_API_KEY"] = self.config.api_key
        logger.info(f"TavilyService initialized: strict={self.config.use_trusted_domains_only}")

    def search(self, query: str) -> Dict[str, Any]:
        try:
            detected = self._detect_companies_in_query(query)
            company_doms = set()
            for c in detected:
                company_doms.update(COMPANY_DOMAINS.get(c, set()))

            tool = TavilySearch(
                max_results=self.config.max_results,
                topic=self.config.topic,
                search_depth=self.config.search_depth,
            )
            raw = tool.invoke({"query": query})

            if isinstance(raw, list):
                results = raw
            elif isinstance(raw, dict):
                results = raw.get("results", [raw])
            elif isinstance(raw, str):
                return {"results": [{"content": raw, "url": "", "title": "Web Result"}]}
            else:
                results = []

            filtered = self._filter_results(results, company_doms)
            return {"results": filtered}
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            raise TavilyServiceError(f"Web search failed: {e}") from e

    def search_multiple(self, queries: List[str], deduplicate: bool = True) -> List[Dict]:
        all_results, seen_urls = [], set()
        for q in queries:
            try:
                resp = self.search(q)
                for r in resp.get("results", []):
                    url = r.get("url", "")
                    if deduplicate and url in seen_urls:
                        continue
                    seen_urls.add(url)
                    all_results.append(r)
            except TavilyServiceError:
                continue
        return all_results

    def _detect_companies_in_query(self, query: str) -> Set[str]:
        detected = set()
        q_lower = query.lower()
        for name in COMPANY_DOMAINS:
            if re.search(r"\b" + re.escape(name) + r"\b", q_lower):
                detected.add(name)
        return detected

    def _filter_results(self, results: List[Dict], company_domains: Optional[Set[str]] = None) -> List[Dict]:
        filtered = []
        for r in results:
            url = r.get("url", "")
            if not url:
                filtered.append(r)
                continue
            domain = self._extract_domain(url)
            if self.config.filter_blocked_domains and domain in self.config.blocked_domains:
                continue
            if company_domains:
                if domain in company_domains or domain in self.config.trusted_domains:
                    filtered.append(r)
            elif self.config.use_trusted_domains_only:
                if domain in self.config.trusted_domains:
                    filtered.append(r)
            else:
                filtered.append(r)
        return filtered

    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            d = urlparse(url).netloc
            return d[4:] if d.startswith("www.") else d
        except Exception:
            return ""

    @staticmethod
    def split_query(query: str, delimiter: str = r"[.]") -> List[str]:
        return [p.strip() for p in re.split(delimiter, query) if p.strip()]

    def add_trusted_domain(self, domain: str):
        self.config.trusted_domains.add(domain)

    def remove_trusted_domain(self, domain: str):
        self.config.trusted_domains.discard(domain)

    def add_blocked_domain(self, domain: str):
        self.config.blocked_domains.add(domain)

    def remove_blocked_domain(self, domain: str):
        self.config.blocked_domains.discard(domain)

    def set_strict_mode(self, enabled: bool):
        self.config.use_trusted_domains_only = enabled
