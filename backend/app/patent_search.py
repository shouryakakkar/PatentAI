from typing import List, Dict, Any, Optional
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PatentSearch:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text using sentence-transformers"""
        try:
            # The model will automatically handle tokenization and generate embeddings
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

    def search_patents(self, query: str, max_results: int = 10, search_mode: str = "semantic",
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      patent_type: Optional[str] = None, inventor_name: Optional[str] = None,
                      assignee_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search patents using either semantic or keyword search"""
        try:
            # Get patents from USPTO API with appropriate query
            uspto_url = "https://api.patentsview.org/patents/query"
            
            # Build query conditions
            conditions = []
            
            # Add text search condition
            if search_mode == "keyword":
                conditions.append({
                    "_or": [
                        {"_text_all": {"patent_title": query}},
                        {"_text_all": {"patent_abstract": query}}
                    ]
                })
            else:
                # For semantic search, use a broader initial query
                terms = query.split()[:3]  # Use first 3 words for broad search
                conditions.append({
                    "_or": [
                        {"_text_any": {"patent_title": terms}},
                        {"_text_any": {"patent_abstract": terms}}
                    ]
                })
            
            # Add date range condition if provided
            if start_date and end_date:
                conditions.append({
                    "_and": [
                        {"_gte": {"patent_date": start_date}},
                        {"_lte": {"patent_date": end_date}}
                    ]
                })
            
            # Add patent type condition if provided
            if patent_type:
                conditions.append({"_text_all": {"patent_type": patent_type}})
            
            # Add inventor name condition if provided
            if inventor_name:
                conditions.append({
                    "_or": [
                        {"_text_all": {"inventor_first_name": inventor_name}},
                        {"_text_all": {"inventor_last_name": inventor_name}}
                    ]
                })
            
            # Add assignee name condition if provided
            if assignee_name:
                conditions.append({"_text_all": {"assignee_organization": assignee_name}})
            
            # Combine all conditions with AND
            uspto_query = {
                "q": {"_and": conditions} if len(conditions) > 1 else conditions[0],
                "f": [
                    "patent_number",
                    "patent_title",
                    "patent_abstract",
                    "patent_date",
                    "inventor_first_name",
                    "inventor_last_name",
                    "assignee_organization"
                ],
                "o": {
                    "per_page": 100 if search_mode == "semantic" else max_results
                }
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(uspto_url, json=uspto_query, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if not data.get("patents"):
                return []
            
            patents = []
            
            if search_mode == "keyword":
                # For keyword search, use USPTO ranking
                for patent_data in data["patents"][:max_results]:
                    patent = {
                        "patent_number": patent_data.get("patent_number", ""),
                        "title": patent_data.get("patent_title", ""),
                        "abstract": patent_data.get("patent_abstract", ""),
                        "inventors": [
                            f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
                            for inv in patent_data.get("inventors", [])
                            if inv.get("inventor_first_name") or inv.get("inventor_last_name")
                        ],
                        "filing_date": patent_data.get("patent_date"),
                        "assignee": patent_data.get("assignee_organization", ""),
                        "patent_type": "",
                        "status": "",
                        "citations": 0,
                        "similarity_score": 1.0  # Default score for keyword matches
                    }
                    patents.append(patent)
            else:
                # For semantic search, calculate similarity scores
                query_embedding = self.get_embedding(query)
                if not query_embedding:
                    return []
                
                for patent_data in data["patents"]:
                    patent_text = f"{patent_data.get('patent_title', '')} {patent_data.get('patent_abstract', '')}"
                    patent_embedding = self.get_embedding(patent_text)
                    
                    if patent_embedding:
                        similarity = cosine_similarity(
                            [query_embedding],
                            [patent_embedding]
                        )[0][0]
                        
                        normalized_similarity = (similarity + 1) / 2
                        
                        if normalized_similarity > 0.5:  # Only include reasonably similar patents
                            patent = {
                                "patent_number": patent_data.get("patent_number", ""),
                                "title": patent_data.get("patent_title", ""),
                                "abstract": patent_data.get("patent_abstract", ""),
                                "inventors": [
                                    f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
                                    for inv in patent_data.get("inventors", [])
                                    if inv.get("inventor_first_name") or inv.get("inventor_last_name")
                                ],
                                "filing_date": patent_data.get("patent_date"),
                                "assignee": patent_data.get("assignee_organization", ""),
                                "patent_type": "",
                                "status": "",
                                "citations": 0,
                                "similarity_score": float(normalized_similarity)
                            }
                            patents.append(patent)
                
                # Sort by similarity score
                patents.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return patents[:max_results]

        except requests.exceptions.RequestException as e:
            print(f"Error querying USPTO API: {str(e)}")
            return []
        except Exception as e:
            print(f"Error processing patent data: {str(e)}")
            return [] 