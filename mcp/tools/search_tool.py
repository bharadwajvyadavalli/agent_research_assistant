"""
Search tool for the agent research assistant.

This module provides a web search capability to allow agents to retrieve information
from the internet. It supports different search engines and caching of results to
improve performance and reduce API calls.
"""

import os
import json
import time
import hashlib
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import urllib.parse
from abc import ABC, abstractmethod

from utils.logging_utils import AgentLogger


class SearchResult:
    """Class representing a single search result."""
    
    def __init__(
        self,
        title: str,
        url: str,
        snippet: str,
        source: str = "unknown",
        published_date: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a new search result.
        
        Args:
            title: Title of the search result
            url: URL of the search result
            snippet: Text snippet from the search result
            source: Source of the result (e.g., website name)
            published_date: Date the content was published (if available)
            metadata: Additional metadata for the result
        """
        self.title = title
        self.url = url
        self.snippet = snippet
        self.source = source
        self.published_date = published_date
        self.metadata = metadata or {}
        self.retrieved_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the search result to a dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date,
            "metadata": self.metadata,
            "retrieved_at": self.retrieved_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create a search result from a dictionary."""
        result = cls(
            title=data["title"],
            url=data["url"],
            snippet=data["snippet"],
            source=data.get("source", "unknown"),
            published_date=data.get("published_date"),
            metadata=data.get("metadata", {})
        )
        result.retrieved_at = data.get("retrieved_at", datetime.now().isoformat())
        return result
    
    def __str__(self) -> str:
        """String representation of the search result."""
        return f"{self.title} - {self.url}"


class SearchProvider(ABC):
    """
    Abstract base class for search providers.
    
    Implement this class to add support for different search engines or APIs.
    """
    
    @abstractmethod
    def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """
        Execute a search query and return results.
        
        Args:
            query: The search query
            num_results: Number of results to return
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        pass


class GoogleSearchProvider(SearchProvider):
    """
    Google Search provider using the Google Custom Search API.
    
    Requires an API key and Custom Search Engine ID.
    """
    
    def __init__(
        self,
        api_key: str,
        cse_id: str,
        logger: Optional[AgentLogger] = None
    ):
        """
        Initialize the Google search provider.
        
        Args:
            api_key: Google API key
            cse_id: Custom Search Engine ID
            logger: Logger instance
        """
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.logger = logger or AgentLogger(agent_id="google_search")
    
    def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """
        Execute a search query using Google's API.
        
        Args:
            query: The search query
            num_results: Number of results to return (max 10 per API call)
            **kwargs: Additional parameters for the API
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        # Ensure num_results is within API limits (max 10 per request)
        requests_needed = (num_results + 9) // 10
        
        try:
            for i in range(requests_needed):
                start_index = i * 10 + 1  # Google uses 1-based indexing
                
                params = {
                    "key": self.api_key,
                    "cx": self.cse_id,
                    "q": query,
                    "num": min(10, num_results - len(results)),
                    "start": start_index
                }
                
                # Add any additional parameters
                params.update(kwargs)
                
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "items" not in data:
                    self.logger.warning(f"No search results found for query: {query}")
                    break
                
                for item in data["items"]:
                    # Extract the source from the URL
                    source = urllib.parse.urlparse(item["link"]).netloc
                    
                    # Extract publication date if available
                    published_date = None
                    if "pagemap" in item and "metatags" in item["pagemap"]:
                        metatags = item["pagemap"]["metatags"][0]
                        date_candidates = [
                            metatags.get("article:published_time"),
                            metatags.get("date"),
                            metatags.get("og:article:published_time")
                        ]
                        published_date = next((d for d in date_candidates if d), None)
                    
                    result = SearchResult(
                        title=item["title"],
                        url=item["link"],
                        snippet=item.get("snippet", ""),
                        source=source,
                        published_date=published_date,
                        metadata={
                            "html_title": item.get("htmlTitle", ""),
                            "html_snippet": item.get("htmlSnippet", ""),
                            "display_link": item.get("displayLink", "")
                        }
                    )
                    results.append(result)
                
                # If we have enough results or there are no more pages, stop
                if len(results) >= num_results or "queries" not in data or "nextPage" not in data["queries"]:
                    break
            
            self.logger.info(f"Retrieved {len(results)} results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error executing Google search: {str(e)}")
            return []


class BingSearchProvider(SearchProvider):
    """
    Bing Search provider using the Bing Web Search API.
    
    Requires an API key.
    """
    
    def __init__(
        self,
        api_key: str,
        logger: Optional[AgentLogger] = None
    ):
        """
        Initialize the Bing search provider.
        
        Args:
            api_key: Bing API key
            logger: Logger instance
        """
        self.api_key = api_key
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"
        self.logger = logger or AgentLogger(agent_id="bing_search")
    
    def search(self, query: str, num_results: int = 5, **kwargs) -> List[SearchResult]:
        """
        Execute a search query using Bing's API.
        
        Args:
            query: The search query
            num_results: Number of results to return
            **kwargs: Additional parameters for the API
            
        Returns:
            List of SearchResult objects
        """
        results = []
        
        try:
            # Bing allows up to 50 results per request
            requests_needed = (num_results + 49) // 50
            
            for i in range(requests_needed):
                offset = i * 50
                
                headers = {
                    "Ocp-Apim-Subscription-Key": self.api_key
                }
                
                params = {
                    "q": query,
                    "count": min(50, num_results - len(results)),
                    "offset": offset,
                    "responseFilter": "Webpages"
                }
                
                # Add any additional parameters
                params.update(kwargs)
                
                response = requests.get(self.base_url, headers=headers, params=params)
                response.raise_for_status()
                data = response.json()
                
                if "webPages" not in data or "value" not in data["webPages"]:
                    self.logger.warning(f"No search results found for query: {query}")
                    break
                
                for item in data["webPages"]["value"]:
                    # Extract the source from the URL
                    source = urllib.parse.urlparse(item["url"]).netloc
                    
                    result = SearchResult(
                        title=item["name"],
                        url=item["url"],
                        snippet=item.get("snippet", ""),
                        source=source,
                        published_date=None,  # Bing doesn't provide this directly
                        metadata={
                            "id": item.get("id", ""),
                            "display_url": item.get("displayUrl", "")
                        }
                    )
                    results.append(result)
                
                # If we have enough results or there are no more pages, stop
                if len(results) >= num_results or offset + len(data["webPages"]["value"]) >= data["webPages"]["totalEstimatedMatches"]:
                    break
            
            self.logger.info(f"Retrieved {len(results)} results for query: {query}")
            return results
        
        except Exception as e:
            self.logger.error(f"Error executing Bing search: {str(e)}")
            return []


class SearchCache:
    """
    Cache for search results to reduce API calls and improve performance.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/search",
        cache_expiration: timedelta = timedelta(hours=24),
        logger: Optional[AgentLogger] = None
    ):
        """
        Initialize the search cache.
        
        Args:
            cache_dir: Directory for storing cached results
            cache_expiration: Time after which cached results are considered stale
            logger: Logger instance
        """
        self.cache_dir = cache_dir
        self.cache_expiration = cache_expiration
        self.logger = logger or AgentLogger(agent_id="search_cache")
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, query: str, provider_name: str, num_results: int) -> str:
        """
        Generate a cache key for a search query.
        
        Args:
            query: The search query
            provider_name: Name of the search provider
            num_results: Number of results requested
            
        Returns:
            Cache key string
        """
        # Create a unique but deterministic key
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{provider_name}_{query_hash}_{num_results}"
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Path to the cache file
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def get(self, query: str, provider_name: str, num_results: int) -> Optional[List[SearchResult]]:
        """
        Get cached search results if available and not expired.
        
        Args:
            query: The search query
            provider_name: Name of the search provider
            num_results: Number of results requested
            
        Returns:
            List of SearchResult objects if cache hit, None otherwise
        """
        cache_key = self._get_cache_key(query, provider_name, num_results)
        cache_file = self._get_cache_file_path(cache_key)
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is expired
            cached_time = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_time > self.cache_expiration:
                self.logger.debug(f"Cache expired for query: {query}")
                return None
            
            # Deserialize search results
            results = [SearchResult.from_dict(item) for item in cache_data["results"]]
            
            self.logger.debug(f"Cache hit for query: {query}")
            return results
            
        except Exception as e:
            self.logger.warning(f"Error reading cache for query '{query}': {str(e)}")
            return None
    
    def set(self, query: str, provider_name: str, num_results: int, results: List[SearchResult]) -> None:
        """
        Cache search results.
        
        Args:
            query: The search query
            provider_name: Name of the search provider
            num_results: Number of results requested
            results: Search results to cache
        """
        cache_key = self._get_cache_key(query, provider_name, num_results)
        cache_file = self._get_cache_file_path(cache_key)
        
        try:
            cache_data = {
                "query": query,
                "provider": provider_name,
                "num_results": num_results,
                "cached_at": datetime.now().isoformat(),
                "results": [r.to_dict() for r in results]
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            self.logger.debug(f"Cached results for query: {query}")
            
        except Exception as e:
            self.logger.warning(f"Error caching results for query '{query}': {str(e)}")


class SearchTool:
    """
    Tool for executing web searches and processing results.
    """
    
    def __init__(
        self,
        providers: Dict[str, SearchProvider],
        default_provider: str,
        cache: Optional[SearchCache] = None,
        logger: Optional[AgentLogger] = None
    ):
        """
        Initialize the search tool.
        
        Args:
            providers: Dictionary of search providers (name -> provider instance)
            default_provider: Name of the default search provider
            cache: Search cache instance or None to disable caching
            logger: Logger instance
        """
        self.providers = providers
        self.default_provider = default_provider
        self.cache = cache or SearchCache()
        self.logger = logger or AgentLogger(agent_id="search_tool")
    
    def search(
        self,
        query: str,
        provider_name: Optional[str] = None,
        num_results: int = 5,
        use_cache: bool = True,
        **kwargs
    ) -> List[SearchResult]:
        """
        Execute a search query using the specified or default provider.
        
        Args:
            query: The search query
            provider_name: Name of the search provider to use (or None for default)
            num_results: Number of results to return
            use_cache: Whether to use cached results if available
            **kwargs: Additional provider-specific parameters
            
        Returns:
            List of SearchResult objects
        """
        provider_name = provider_name or self.default_provider
        
        if provider_name not in self.providers:
            self.logger.error(f"Unknown search provider: {provider_name}")
            return []
        
        provider = self.providers[provider_name]
        
        # Check cache if enabled
        if use_cache:
            cached_results = self.cache.get(query, provider_name, num_results)
            if cached_results:
                self.logger.info(f"Using cached results for query: {query}")
                return cached_results
        
        # Execute search with the provider
        self.logger.info(f"Executing search with provider {provider_name}: {query}")
        results = provider.search(query, num_results=num_results, **kwargs)
        
        # Cache results if enabled
        if use_cache and results:
            self.cache.set(query, provider_name, num_results, results)
        
        return results
    
    def extract_key_information(self, results: List[SearchResult]) -> Dict[str, Any]:
        """
        Extract key information from search results.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary with extracted information
        """
        if not results:
            return {"status": "no_results"}
        
        # Extract sources
        sources = list(set(result.source for result in results))
        
        # Count results by date (if available)
        dates = {}
        for result in results:
            if result.published_date:
                date = result.published_date.split("T")[0]  # Just the date part
                dates[date] = dates.get(date, 0) + 1
        
        # Extract potential entities (simple approach)
        entities = set()
        for result in results:
            # Look for capitalized phrases in titles and snippets
            for text in [result.title, result.snippet]:
                words = text.split()
                for i in range(len(words) - 1):
                    if words[i][0].isupper() and words[i+1][0].isupper():
                        entities.add(f"{words[i]} {words[i+1]}")
        
        return {
            "total_results": len(results),
            "sources": sources,
            "top_sources": [source for source, count in 
                           sorted(((s, sources.count(s)) for s in set(sources)), 
                                 key=lambda x: x[1], reverse=True)],
            "date_distribution": dates,
            "potential_entities": list(entities)[:10]  # Limit to top 10
        }
    
    def format_results_as_markdown(self, results: List[SearchResult]) -> str:
        """
        Format search results as a Markdown string.
        
        Args:
            results: List of search results
            
        Returns:
            Markdown formatted string
        """
        if not results:
            return "No search results found."
        
        markdown = "## Search Results\n\n"
        
        for i, result in enumerate(results, 1):
            markdown += f"### {i}. {result.title}\n"
            markdown += f"**Source:** {result.source}\n"
            if result.published_date:
                markdown += f"**Published:** {result.published_date.split('T')[0]}\n"
            markdown += f"**URL:** {result.url}\n\n"
            markdown += f"{result.snippet}\n\n"
            markdown += "---\n\n"
        
        return markdown


# Factory function for creating search tools
def create_search_tool(
    google_api_key: Optional[str] = None,
    google_cse_id: Optional[str] = None,
    bing_api_key: Optional[str] = None,
    default_provider: Optional[str] = None,
    cache_dir: str = "cache/search",
    cache_expiration_hours: int = 24
) -> SearchTool:
    """
    Create a SearchTool instance with configured providers.
    
    Args:
        google_api_key: Google API key (if using Google search)
        google_cse_id: Google Custom Search Engine ID (if using Google search)
        bing_api_key: Bing API key (if using Bing search)
        default_provider: Name of the default provider
        cache_dir: Directory for caching search results
        cache_expiration_hours: Cache expiration time in hours
        
    Returns:
        Configured SearchTool instance
    """
    providers = {}
    logger = AgentLogger(agent_id="search_tool_factory")
    
    # Add Google provider if credentials are provided
    if google_api_key and google_cse_id:
        providers["google"] = GoogleSearchProvider(
            api_key=google_api_key,
            cse_id=google_cse_id,
            logger=logger
        )
    
    # Add Bing provider if credentials are provided
    if bing_api_key:
        providers["bing"] = BingSearchProvider(
            api_key=bing_api_key,
            logger=logger
        )
    
    # Determine default provider
    if not providers:
        raise ValueError("No search providers configured. At least one provider must be available.")
    
    if default_provider is None or default_provider not in providers:
        default_provider = list(providers.keys())[0]
        logger.info(f"Using {default_provider} as the default search provider")
    
    # Create cache
    cache = SearchCache(
        cache_dir=cache_dir,
        cache_expiration=timedelta(hours=cache_expiration_hours),
        logger=logger
    )
    
    # Create and return the search tool
    return SearchTool(
        providers=providers,
        default_provider=default_provider,
        cache=cache,
        logger=logger
    )
