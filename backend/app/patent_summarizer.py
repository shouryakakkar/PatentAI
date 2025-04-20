from transformers import pipeline
from typing import Dict, Any

class PatentSummarizer:
    def __init__(self):
        # Initialize the summarization pipeline
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_patent(self, patent_data: Dict[str, Any]) -> str:
        """
        Generate an easy-to-understand summary of a patent.
        
        Args:
            patent_data: Dictionary containing patent information
                        (title, abstract, etc.)
        
        Returns:
            str: Easy-to-understand summary of the patent
        """
        # Combine title and abstract for summarization
        text = f"{patent_data['title']}\n\n{patent_data['abstract']}"
        
        # Generate summary
        summary = self.summarizer(
            text,
            max_length=150,
            min_length=50,
            do_sample=False
        )[0]['summary_text']
        
        # Make the summary more conversational
        summary = self._make_conversational(summary)
        
        return summary
    
    def _make_conversational(self, text: str) -> str:
        """
        Make the summary more conversational and easier to understand.
        """
        # Add a friendly introduction
        text = f"This patent is about {text.lower()}"
        
        # Replace technical terms with simpler explanations
        replacements = {
            "comprises": "includes",
            "wherein": "where",
            "thereof": "of it",
            "therein": "in it",
            "thereby": "by doing this",
            "therefor": "for this",
            "therefrom": "from it",
            "therethrough": "through it",
            "thereunder": "under it",
            "therewith": "with it",
            "thereafter": "after that",
            "therebefore": "before that",
            "thereabout": "about that",
            "thereagainst": "against that",
            "thereamong": "among that",
            "therebetween": "between those",
            "thereby": "by that",
            "therefor": "for that",
            "therefrom": "from that",
            "therein": "in that",
            "thereof": "of that",
            "thereon": "on that",
            "thereto": "to that",
            "thereunder": "under that",
            "therewith": "with that",
            "therewithin": "within that",
            "therewithout": "without that"
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text 