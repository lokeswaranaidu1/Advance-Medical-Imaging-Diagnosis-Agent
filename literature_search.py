"""
PubMed Literature Search Module
Integrates with Biopython to fetch relevant medical research articles
"""

from Bio import Entrez
from Bio import Medline
import requests
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from config import config

logger = logging.getLogger(__name__)

class PubMedLiteratureSearch:
    """PubMed literature search and retrieval system"""
    
    def __init__(self):
        self.email = config.PUBMED_EMAIL
        self.tool = config.PUBMED_TOOL
        self.max_results = config.PUBMED_MAX_RESULTS
        
        # Setup Entrez
        Entrez.email = self.email
        Entrez.tool = self.tool
        
        # Cache for search results
        self.search_cache = {}
        self.cache_expiry = timedelta(hours=24)
    
    def search_medical_conditions(self, condition: str, max_results: int = None) -> List[Dict[str, Any]]:
        """
        Search PubMed for medical conditions
        
        Args:
            condition: Medical condition or diagnosis to search for
            max_results: Maximum number of results to return
            
        Returns:
            List of relevant medical articles
        """
        try:
            if max_results is None:
                max_results = self.max_results
            
            # Check cache first
            cache_key = f"{condition}_{max_results}"
            if cache_key in self.search_cache:
                cache_entry = self.search_cache[cache_key]
                if datetime.now() - cache_entry['timestamp'] < self.cache_expiry:
                    logger.info(f"Returning cached results for: {condition}")
                    return cache_entry['results']
            
            # Perform search
            search_term = self._build_search_query(condition)
            logger.info(f"Searching PubMed for: {search_term}")
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=search_term,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()
            
            if not search_results['IdList']:
                logger.info(f"No results found for: {condition}")
                return []
            
            # Fetch article details
            articles = self._fetch_article_details(search_results['IdList'])
            
            # Cache results
            self.search_cache[cache_key] = {
                'results': articles,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Found {len(articles)} articles for: {condition}")
            return articles
            
        except Exception as e:
            logger.error(f"Error searching PubMed for {condition}: {str(e)}")
            return []
    
    def search_by_diagnosis(self, diagnosis: str, modality: str = None, body_part: str = None) -> List[Dict[str, Any]]:
        """
        Search PubMed by diagnosis with additional context
        
        Args:
            diagnosis: Primary diagnosis
            modality: Imaging modality (CT, MRI, X-ray, etc.)
            body_part: Body part examined
            
        Returns:
            List of relevant medical articles
        """
        try:
            # Build enhanced search query
            search_terms = [diagnosis]
            
            if modality:
                search_terms.append(f"imaging[Title/Abstract] AND {modality}[Title/Abstract]")
            
            if body_part:
                search_terms.append(f"{body_part}[Title/Abstract]")
            
            # Combine search terms
            combined_query = " AND ".join(search_terms)
            
            return self.search_medical_conditions(combined_query)
            
        except Exception as e:
            logger.error(f"Error in diagnosis search: {str(e)}")
            return []
    
    def search_recent_articles(self, condition: str, days: int = 365) -> List[Dict[str, Any]]:
        """
        Search for recent articles within specified time period
        
        Args:
            condition: Medical condition to search for
            days: Number of days to look back
            
        Returns:
            List of recent medical articles
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for PubMed
            start_str = start_date.strftime("%Y/%m/%d")
            end_str = end_date.strftime("%Y/%m/%d")
            
            # Build date-restricted search
            date_query = f"{condition} AND ({start_str}[Date - Publication] : {end_str}[Date - Publication])"
            
            return self.search_medical_conditions(date_query)
            
        except Exception as e:
            logger.error(f"Error searching recent articles: {str(e)}")
            return []
    
    def search_systematic_reviews(self, condition: str) -> List[Dict[str, Any]]:
        """
        Search specifically for systematic reviews and meta-analyses
        
        Args:
            condition: Medical condition to search for
            
        Returns:
            List of systematic reviews
        """
        try:
            # Add publication type filters for systematic reviews
            review_query = f"{condition} AND (systematic review[Publication Type] OR meta-analysis[Publication Type])"
            
            return self.search_medical_conditions(review_query)
            
        except Exception as e:
            logger.error(f"Error searching systematic reviews: {str(e)}")
            return []
    
    def _build_search_query(self, condition: str) -> str:
        """Build optimized PubMed search query"""
        try:
            # Clean and enhance search terms
            condition = condition.strip().lower()
            
            # Add common medical search enhancements
            enhancements = [
                "diagnosis[Title/Abstract]",
                "imaging[Title/Abstract]",
                "radiology[Title/Abstract]"
            ]
            
            # Combine base condition with enhancements
            enhanced_query = f"{condition} AND ({' OR '.join(enhancements)})"
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error building search query: {str(e)}")
            return condition
    
    def _fetch_article_details(self, pmid_list: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed information for PubMed articles"""
        try:
            articles = []
            
            # Process in batches to avoid overwhelming the API
            batch_size = 50
            for i in range(0, len(pmid_list), batch_size):
                batch = pmid_list[i:i + batch_size]
                
                # Fetch article details
                handle = Entrez.efetch(
                    db="pubmed",
                    id=batch,
                    rettype="medline",
                    retmode="text"
                )
                
                # Parse Medline records
                records = Medline.parse(handle)
                for record in records:
                    article = self._parse_medline_record(record)
                    if article:
                        articles.append(article)
                
                handle.close()
                
                # Rate limiting
                time.sleep(0.1)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching article details: {str(e)}")
            return []
    
    def _parse_medline_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Medline record into structured format"""
        try:
            article = {
                'pmid': record.get('PMID', ''),
                'title': record.get('TI', ''),
                'abstract': record.get('AB', ''),
                'authors': record.get('AU', []),
                'journal': record.get('JT', ''),
                'publication_date': record.get('DP', ''),
                'publication_type': record.get('PT', []),
                'mesh_terms': record.get('MH', []),
                'keywords': record.get('KW', []),
                'doi': record.get('LID', ''),
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{record.get('PMID', '')}/"
            }
            
            # Clean up abstract
            if article['abstract']:
                article['abstract'] = article['abstract'].replace('\n', ' ').strip()
            
            # Extract year from publication date
            if article['publication_date']:
                try:
                    year = article['publication_date'].split()[-1]
                    article['year'] = int(year) if year.isdigit() else None
                except:
                    article['year'] = None
            
            return article
            
        except Exception as e:
            logger.error(f"Error parsing Medline record: {str(e)}")
            return None
    
    def get_article_summary(self, pmid: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific PubMed article"""
        try:
            # Fetch summary
            handle = Entrez.esummary(db="pubmed", id=pmid)
            summary = Entrez.read(handle)
            handle.close()
            
            if pmid in summary['PubmedArticle']:
                article_data = summary['PubmedArticle'][pmid]
                return {
                    'pmid': pmid,
                    'title': article_data.get('ArticleTitle', ''),
                    'abstract': article_data.get('Abstract', ''),
                    'authors': article_data.get('AuthorList', []),
                    'journal': article_data.get('Source', ''),
                    'publication_date': article_data.get('PubDate', ''),
                    'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching article summary for {pmid}: {str(e)}")
            return None
    
    def search_clinical_guidelines(self, condition: str) -> List[Dict[str, Any]]:
        """Search for clinical practice guidelines"""
        try:
            guidelines_query = f"{condition} AND (practice guideline[Publication Type] OR guideline[Title/Abstract])"
            return self.search_medical_conditions(guidelines_query)
            
        except Exception as e:
            logger.error(f"Error searching clinical guidelines: {str(e)}")
            return []
    
    def get_related_articles(self, pmid: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get related articles based on a specific PMID"""
        try:
            # Use PubMed's related articles feature
            handle = Entrez.elink(dbfrom="pubmed", db="pubmed", LinkName="pubmed_pubmed_citedin", id=pmid)
            results = Entrez.read(handle)
            handle.close()
            
            if results and results[0]['LinkSetDb']:
                related_pmids = [link['Id'] for link in results[0]['LinkSetDb'][0]['Link']]
                related_pmids = related_pmids[:max_results]
                
                return self._fetch_article_details(related_pmids)
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching related articles for {pmid}: {str(e)}")
            return []
    
    def clear_cache(self):
        """Clear the search cache"""
        self.search_cache.clear()
        logger.info("PubMed search cache cleared")
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about search cache usage"""
        try:
            total_searches = len(self.search_cache)
            cache_size = sum(len(str(v)) for v in self.search_cache.values())
            
            return {
                'total_searches': total_searches,
                'cache_size_bytes': cache_size,
                'cache_entries': list(self.search_cache.keys())
            }
            
        except Exception as e:
            logger.error(f"Error getting search statistics: {str(e)}")
            return {}

# Global instance
pubmed_search = PubMedLiteratureSearch()
