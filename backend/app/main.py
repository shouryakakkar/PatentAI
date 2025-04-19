from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import requests
import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from collections import Counter
from .model_training import ModelTrainer, create_training_pairs

# Load environment variables
load_dotenv()

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# IPC section labels (main technology categories)
IPC_SECTIONS = {
    "A": "Human Necessities",
    "B": "Performing Operations, Transporting",
    "C": "Chemistry, Metallurgy",
    "D": "Textiles, Paper",
    "E": "Fixed Constructions",
    "F": "Mechanical Engineering, Lighting, Heating, Weapons",
    "G": "Physics",
    "H": "Electricity"
}

# Initialize the patent classifier model
classifier_name = "anferico/bert-for-patents"
classifier_tokenizer = AutoTokenizer.from_pretrained(classifier_name)
classifier_model = AutoModelForSequenceClassification.from_pretrained(classifier_name, num_labels=len(IPC_SECTIONS))

# Initialize the model trainer
model_trainer = ModelTrainer()

app = FastAPI(title="PatentAI - AI Patent Search Tool")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatentSearchQuery(BaseModel):
    query: str
    max_results: int = 10
    search_mode: str = "semantic"  # "semantic" or "keyword"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    patent_type: Optional[str] = None
    inventor_name: Optional[str] = None
    assignee_name: Optional[str] = None

class Patent(BaseModel):
    patent_number: str
    title: str
    abstract: str
    inventors: Optional[List[str]] = None
    filing_date: Optional[str] = None
    assignee: Optional[str] = None
    patent_type: Optional[str] = None
    status: Optional[str] = None
    citations: Optional[int] = None
    similarity_score: Optional[float] = None
    predicted_classes: Optional[List[dict]] = None  # New field for predicted classes

class FeedbackRequest(BaseModel):
    query: str
    relevant_patents: List[Dict]
    irrelevant_patents: List[Dict]

class TrainingRequest(BaseModel):
    patents: List[Dict]
    batch_size: int = 16
    epochs: int = 4
    learning_rate: float = 2e-5

def get_embedding(text: str) -> List[float]:
    """Get embedding for a text using sentence-transformers"""
    try:
        # The model will automatically handle tokenization and generate embeddings
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def predict_patent_classes(text: str, top_k: int = 3) -> List[dict]:
    """Predict the most likely patent classifications for a given text"""
    try:
        # Tokenize the text with proper max length
        inputs = classifier_tokenizer(
            text, 
            truncation=True, 
            padding=True, 
            return_tensors="pt",
            max_length=512  # Standard BERT max length
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = classifier_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Ensure top_k doesn't exceed the number of classes
        num_classes = len(IPC_SECTIONS)
        top_k = min(top_k, num_classes)
        
        # Get top k predictions
        top_probs, top_indices = torch.topk(probabilities[0], k=top_k)
        
        # Convert to list of dictionaries with class and probability
        predictions = []
        sections = list(IPC_SECTIONS.keys())
        for prob, idx in zip(top_probs, top_indices):
            if idx < len(sections):  # Add safety check
                section = sections[idx]
                predictions.append({
                    "section": section,
                    "description": IPC_SECTIONS[section],
                    "probability": float(prob)
                })
        
        return predictions
    except Exception as e:
        print(f"Error predicting patent classes: {str(e)}")
        return []

def search_patents(query: str, max_results: int = 10, search_mode: str = "semantic") -> List[Patent]:
    """Search patents using either semantic or keyword search"""
    try:
        # Get patents from USPTO API with appropriate query
        uspto_url = "https://api.patentsview.org/patents/query"
        
        if search_mode == "keyword":
            # Keyword search: exact matches in title or abstract
            uspto_query = {
                "q": {
                    "_or": {
                        "_text_all": {
                            "patent_title": query
                        },
                        "_text_all": {
                            "patent_abstract": query
                        }
                    }
                },
                "f": [
                    "patent_number",
                    "patent_title",
                    "patent_abstract",
                    "patent_date",
                    "inventor_first_name",
                    "inventor_last_name",
                    "assignee_organization"
                ]
            }
        else:
            # Semantic search: broader query to get candidates for semantic ranking
            # Get embedding for the query
            query_embedding = get_embedding(query)
            if not query_embedding:
                raise HTTPException(status_code=500, detail="Failed to generate query embedding")
                
            # Use key terms from the query for initial broad search
            terms = query.split()[:3]  # Use first 3 words for broad search
            uspto_query = {
                "q": {
                    "_or": [
                        {
                            "_text_any": {
                                "patent_title": terms
                            }
                        },
                        {
                            "_text_any": {
                                "patent_abstract": terms
                            }
                        }
                    ]
                },
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
                    "per_page": 100  # Get more results for semantic filtering
                }
            }
        
        print(f"USPTO Query: {json.dumps(uspto_query, indent=2)}")
        headers = {
            "Content-Type": "application/json"
        }
        response = requests.post(uspto_url, json=uspto_query, headers=headers)
        print(f"USPTO Response Status: {response.status_code}")
        response.raise_for_status()
        data = response.json()
        
        if not data.get("patents"):
            return []
            
        patents = []
        
        # Process results based on search mode
        if search_mode == "keyword":
            # For keyword search, use USPTO ranking
            for patent_data in data["patents"][:max_results]:
                patent = Patent(
                    patent_number=patent_data.get("patent_number", ""),
                    title=patent_data.get("patent_title", ""),
                    abstract=patent_data.get("patent_abstract", ""),
                    inventors=[
                        f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
                        for inv in patent_data.get("inventors", [])
                        if inv.get("inventor_first_name") or inv.get("inventor_last_name")
                    ],
                    filing_date=patent_data.get("patent_date"),
                    assignee=patent_data.get("assignee_organization", ""),
                    patent_type="",
                    status="",
                    citations=0,
                    similarity_score=1.0  # Default score for keyword matches
                )
                patents.append(patent)
        else:
            # For semantic search, calculate similarity scores
            query_embedding = get_embedding(query)
            if not query_embedding:
                raise HTTPException(status_code=500, detail="Failed to generate query embedding")
                
            for patent_data in data["patents"]:
                patent_text = f"{patent_data.get('patent_title', '')} {patent_data.get('patent_abstract', '')}"
                patent_embedding = get_embedding(patent_text)
                
                if patent_embedding:
                    similarity = cosine_similarity(
                        [query_embedding],
                        [patent_embedding]
                    )[0][0]
                    
                    normalized_similarity = (similarity + 1) / 2
                    
                    if normalized_similarity > 0.5:  # Only include reasonably similar patents
                        patent = Patent(
                            patent_number=patent_data.get("patent_number", ""),
                            title=patent_data.get("patent_title", ""),
                            abstract=patent_data.get("patent_abstract", ""),
                            inventors=[
                                f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
                                for inv in patent_data.get("inventors", [])
                                if inv.get("inventor_first_name") or inv.get("inventor_last_name")
                            ],
                            filing_date=patent_data.get("patent_date"),
                            assignee=patent_data.get("assignee_organization", ""),
                            patent_type="",
                            status="",
                            citations=0,
                            similarity_score=float(normalized_similarity)
                        )
                        patents.append(patent)
            
            # Sort by similarity score
            patents.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return patents[:max_results]

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error querying USPTO API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing patent data: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to Patent AI Search Tool"}

@app.post("/search", response_model=List[Patent])
async def search_patents_endpoint(query: PatentSearchQuery):
    """Search patents with semantic or keyword search"""
    return search_patents(query.query, query.max_results, query.search_mode)

@app.get("/patent-types")
async def get_patent_types():
    """Get list of available patent types"""
    return {
        "types": [
            "Utility",
            "Design",
            "Plant",
            "Reissue",
            "Defensive Publication",
            "Statutory Invention Registration"
        ]
    }

@app.get("/patent-statuses")
async def get_patent_statuses():
    """Get list of available patent statuses"""
    return {
        "statuses": [
            "Active",
            "Expired",
            "Pending",
            "Abandoned",
            "Rejected",
            "Withdrawn"
        ]
    }

@app.get("/test-openai")
async def test_openai():
    """Test endpoint to verify embedding model"""
    try:
        # Try to get an embedding for a simple test text
        test_text = "This is a test for PatentAI"
        embedding = get_embedding(test_text)
        
        if embedding:
            return {
                "status": "success",
                "message": "Embedding model loaded successfully",
                "embedding_length": len(embedding)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to get embedding")
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error testing embedding model: {str(e)}"
        )

@app.get("/similar-patents/{patent_number}")
async def get_similar_patents(patent_number: str, max_results: int = 5):
    """Find similar patents based on semantic similarity"""
    try:
        # First, get the source patent details
        uspto_url = "https://api.patentsview.org/patents/query"
        uspto_query = {
            "q": {
                "patent_number": patent_number
            },
            "f": [
                "patent_number",
                "patent_title",
                "patent_abstract",
                "patent_date",
                "inventor_first_name",
                "inventor_last_name",
                "assignee_organization"
            ]
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Get source patent
        response = requests.post(uspto_url, json=uspto_query, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=404,
                detail=f"Patent {patent_number} not found"
            )
            
        data = response.json()
        if not data.get("patents"):
            raise HTTPException(
                status_code=404,
                detail=f"Patent {patent_number} not found"
            )
            
        source_patent = data["patents"][0]
        source_text = f"{source_patent.get('patent_title', '')} {source_patent.get('patent_abstract', '')}"
        source_embedding = get_embedding(source_text)
        
        if not source_embedding:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embedding for source patent"
            )
            
        # Search for similar patents using semantic search
        # Use a broader search query based on the patent title
        search_query = {
            "q": {
                "_text_all": {
                    "patent_title": source_patent.get('patent_title', '').split()[:3]  # Use first 3 words
                }
            },
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
                "per_page": 50  # Get more results for better similarity matching
            }
        }
        
        response = requests.post(uspto_url, json=search_query, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail="Failed to search for similar patents"
            )
            
        data = response.json()
        similar_patents = []
        
        # Process results and calculate similarities
        for patent_data in data.get("patents", []):
            # Skip the source patent
            if patent_data.get("patent_number") == patent_number:
                continue
                
            patent_text = f"{patent_data.get('patent_title', '')} {patent_data.get('patent_abstract', '')}"
            patent_embedding = get_embedding(patent_text)
            
            if patent_embedding:
                # Calculate similarity score
                similarity = cosine_similarity(
                    [source_embedding],
                    [patent_embedding]
                )[0][0]
                
                # Convert similarity to a proper percentage (0 to 1)
                normalized_similarity = (similarity + 1) / 2
                
                # Predict patent classes
                predicted_classes = predict_patent_classes(patent_text)
                
                patent = Patent(
                    patent_number=patent_data.get("patent_number", ""),
                    title=patent_data.get("patent_title", ""),
                    abstract=patent_data.get("patent_abstract", ""),
                    inventors=[
                        f"{inv.get('inventor_first_name', '')} {inv.get('inventor_last_name', '')}".strip()
                        for inv in patent_data.get("inventors", [])
                        if inv.get("inventor_first_name") or inv.get("inventor_last_name")
                    ],
                    filing_date=patent_data.get("patent_date"),
                    assignee=patent_data.get("assignee_organization", ""),
                    patent_type="",
                    status="",
                    citations=0,
                    similarity_score=float(normalized_similarity),
                    predicted_classes=predicted_classes
                )
                similar_patents.append(patent)
        
        # Sort by similarity score and return top results
        similar_patents.sort(key=lambda x: x.similarity_score, reverse=True)
        return similar_patents[:max_results]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error finding similar patents: {str(e)}"
        )

@app.get("/patent-analytics")
async def get_patent_analytics(query: str, time_range: int = 5):
    """Get patent analytics including technology trends"""
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * time_range)  # Default 5 years
        
        # Query USPTO API for patents in date range
        uspto_url = "https://api.patentsview.org/patents/query"
        uspto_query = {
            "q": {
                "_and": [
                    {
                        "_text_all": {
                            "patent_title": query
                        }
                    },
                    {
                        "_gte": {
                            "patent_date": start_date.strftime("%Y-%m-%d")
                        }
                    }
                ]
            },
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
                "per_page": 100
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        response = requests.post(uspto_url, json=uspto_query, headers=headers)
        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail="Failed to fetch patent data for analytics"
            )
            
        data = response.json()
        patents = data.get("patents", [])
        
        # Initialize analytics
        analytics = {
            "total_patents": len(patents),
            "time_range_years": time_range,
            "technology_trends": [],
            "top_companies": [],
            "filing_trends": {},
            "inventor_stats": {
                "total_inventors": 0,
                "avg_inventors_per_patent": 0
            }
        }
        
        # Process patents for analytics
        company_counter = Counter()
        year_counter = Counter()
        total_inventors = 0
        
        for patent in patents:
            # Company analysis
            if patent.get("assignees"):
                for assignee in patent.get("assignees", []):
                    if org := assignee.get("assignee_organization"):
                        company_counter[org] += 1
            
            # Filing trends
            if date_str := patent.get("patent_date"):
                year = date_str[:4]  # Extract year
                year_counter[year] += 1
            
            # Inventor counting
            total_inventors += len(patent.get("inventors", []))
            
            # Technology classification
            text = f"{patent.get('patent_title', '')} {patent.get('patent_abstract', '')}"
            if text.strip():
                predicted_classes = predict_patent_classes(text, top_k=1)
                if predicted_classes:
                    analytics["technology_trends"].append(predicted_classes[0])
        
        # Calculate statistics
        analytics["top_companies"] = [
            {"name": company, "patent_count": count}
            for company, count in company_counter.most_common(5)
        ]
        
        analytics["filing_trends"] = {
            year: count for year, count in sorted(year_counter.items())
        }
        
        if patents:
            analytics["inventor_stats"]["total_inventors"] = total_inventors
            analytics["inventor_stats"]["avg_inventors_per_patent"] = round(total_inventors / len(patents), 2)
        
        # Analyze technology trends
        tech_counter = Counter()
        for tech in analytics["technology_trends"]:
            tech_counter[tech["description"]] += 1
        
        analytics["technology_trends"] = [
            {"category": tech, "count": count, "percentage": round(count * 100 / len(patents), 1)}
            for tech, count in tech_counter.most_common()
        ]
        
        return analytics
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating patent analytics: {str(e)}"
        )

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback for search results to improve the model"""
    try:
        model_trainer.collect_training_data_from_search(
            feedback.query,
            feedback.relevant_patents,
            feedback.irrelevant_patents
        )
        return {"message": "Feedback collected successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error collecting feedback: {str(e)}"
        )

@app.post("/train")
async def train_model(request: TrainingRequest):
    """Train the model with new patent data"""
    try:
        # Create training pairs from patents
        training_pairs = create_training_pairs(request.patents)
        
        # Add training pairs to the model trainer
        for text1, text2, similarity in training_pairs:
            model_trainer.add_training_pair(text1, text2, similarity)
        
        # Train the model
        model_trainer.train(
            batch_size=request.batch_size,
            epochs=request.epochs,
            learning_rate=request.learning_rate
        )
        
        # Evaluate the model
        evaluation_results = model_trainer.evaluate()
        
        # Save the model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"models/patent_search_model_{timestamp}"
        model_trainer.save_model(model_path)
        
        return {
            "message": "Model training completed successfully",
            "evaluation_results": evaluation_results,
            "model_path": model_path
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error training model: {str(e)}"
        )

@app.get("/model-stats")
async def get_model_stats():
    """Get current model statistics"""
    try:
        return {
            "model_name": model_trainer.base_model_name,
            "training_data_size": len(model_trainer.training_data),
            "validation_data_size": len(model_trainer.validation_data)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model stats: {str(e)}"
        ) 