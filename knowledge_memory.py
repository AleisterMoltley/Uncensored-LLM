"""
Knowledge Memory System for LLM Servant
Stores learned knowledge from PDFs to shape bot personality.
Uses compression to keep memory size bounded even after many PDFs.
"""

import json
import hashlib
import gzip
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Default configuration
DEFAULT_MAX_MEMORY_SIZE_MB = 10  # Maximum memory file size in MB
DEFAULT_MAX_INSIGHTS_PER_TOPIC = 50  # Max insights per topic before compression
DEFAULT_SUMMARY_THRESHOLD = 20  # Number of insights before summarizing

# Insight extraction constants
MIN_INSIGHT_WORDS = 5  # Minimum words for a sentence to be considered an insight
MAX_INSIGHT_WORDS = 40  # Maximum words for a sentence to be considered an insight
SIMILARITY_THRESHOLD = 0.6  # Word overlap threshold for merging similar insights (60%)


class KnowledgeMemory:
    """
    Persistent knowledge memory that stores learned insights from PDFs.
    
    Features:
    - Extracts key insights, facts, and arguments from PDF content
    - Organizes knowledge by topics
    - Compresses and summarizes to prevent unbounded growth
    - Enables rational argument comparison
    - Shapes bot personality based on learned knowledge
    """
    
    def __init__(self, memory_dir: Path, config: Optional[Dict] = None):
        """
        Initialize knowledge memory.
        
        Args:
            memory_dir: Directory to store memory files
            config: Configuration dict with optional settings
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.config = config or {}
        self.max_size_mb = self.config.get("max_knowledge_memory_mb", DEFAULT_MAX_MEMORY_SIZE_MB)
        self.max_insights_per_topic = self.config.get("max_insights_per_topic", DEFAULT_MAX_INSIGHTS_PER_TOPIC)
        self.summary_threshold = self.config.get("summary_threshold", DEFAULT_SUMMARY_THRESHOLD)
        
        self.memory_file = self.memory_dir / "knowledge_memory.json.gz"
        self.memory: Dict[str, Any] = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "statistics": {
                "total_pdfs_processed": 0,
                "total_insights_extracted": 0,
                "compressions_performed": 0
            },
            "topics": {},  # Topic -> list of insights
            "core_beliefs": [],  # Most important, frequently reinforced beliefs
            "arguments": [],  # Arguments and counter-arguments learned
            "source_hashes": []  # Track which PDFs have been processed
        }
        self._load()
    
    def _load(self):
        """Load memory from compressed file."""
        if self.memory_file.exists():
            try:
                with gzip.open(self.memory_file, 'rt', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Merge with defaults to handle version upgrades
                    for key in self.memory:
                        if key in loaded:
                            self.memory[key] = loaded[key]
            except (json.JSONDecodeError, IOError, gzip.BadGzipFile) as e:
                print(f"⚠️ Could not load knowledge memory: {e}")
    
    def _save(self):
        """Save memory to compressed file with size check."""
        self.memory["updated"] = datetime.now().isoformat()
        
        # Serialize to JSON
        json_data = json.dumps(self.memory, ensure_ascii=False, indent=None)
        
        # Check size before compression
        size_bytes = len(json_data.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)
        
        # If too large, trigger compression
        if size_mb > self.max_size_mb:
            self._compress_memory()
            json_data = json.dumps(self.memory, ensure_ascii=False, indent=None)
        
        # Save compressed
        try:
            with gzip.open(self.memory_file, 'wt', encoding='utf-8') as f:
                f.write(json_data)
        except IOError as e:
            print(f"⚠️ Could not save knowledge memory: {e}")
    
    def _compress_memory(self):
        """
        Compress memory by summarizing insights.
        Called automatically when memory exceeds size limit.
        """
        self.memory["statistics"]["compressions_performed"] += 1
        
        # 1. Compress topics by keeping only most relevant insights
        for topic, insights in self.memory["topics"].items():
            if len(insights) > self.max_insights_per_topic:
                # Keep most recent and highest-weighted insights
                sorted_insights = sorted(
                    insights,
                    key=lambda x: (x.get("weight", 1), x.get("added", "")),
                    reverse=True
                )
                self.memory["topics"][topic] = sorted_insights[:self.max_insights_per_topic]
        
        # 2. Merge similar insights within topics
        for topic in self.memory["topics"]:
            self.memory["topics"][topic] = self._merge_similar_insights(
                self.memory["topics"][topic]
            )
        
        # 3. Prune low-weight arguments
        if len(self.memory["arguments"]) > 100:
            self.memory["arguments"] = sorted(
                self.memory["arguments"],
                key=lambda x: x.get("strength", 1),
                reverse=True
            )[:100]
        
        # 4. Consolidate core beliefs
        if len(self.memory["core_beliefs"]) > 20:
            self.memory["core_beliefs"] = self._consolidate_beliefs(
                self.memory["core_beliefs"]
            )[:20]
    
    def _merge_similar_insights(self, insights: List[Dict]) -> List[Dict]:
        """Merge insights with similar content to reduce redundancy."""
        if len(insights) <= 5:
            return insights
        
        merged = []
        used_indices = set()
        
        for i, insight in enumerate(insights):
            if i in used_indices:
                continue
            
            # Find similar insights
            similar_group = [insight]
            content_i = insight.get("content", "").lower()
            
            for j, other in enumerate(insights[i+1:], start=i+1):
                if j in used_indices:
                    continue
                content_j = other.get("content", "").lower()
                
                # Simple similarity check using common words
                words_i = set(content_i.split())
                words_j = set(content_j.split())
                min_word_count = min(len(words_i), len(words_j))
                if min_word_count > 0:
                    overlap = len(words_i & words_j) / min_word_count
                    if overlap > SIMILARITY_THRESHOLD:
                        similar_group.append(other)
                        used_indices.add(j)
            
            # Merge the group into one insight with combined weight
            if len(similar_group) > 1:
                merged_insight = {
                    "content": similar_group[0]["content"],  # Keep first content
                    "weight": sum(s.get("weight", 1) for s in similar_group),
                    "sources": list(set(
                        s.get("source", "unknown") for s in similar_group
                    )),
                    "added": similar_group[0].get("added", datetime.now().isoformat()),
                    "merged_count": len(similar_group)
                }
                merged.append(merged_insight)
            else:
                merged.append(insight)
            
            used_indices.add(i)
        
        return merged
    
    def _consolidate_beliefs(self, beliefs: List[Dict]) -> List[Dict]:
        """Consolidate beliefs by merging similar ones and boosting weights."""
        return self._merge_similar_insights(beliefs)
    
    def extract_knowledge_from_chunks(
        self,
        chunks: List[str],
        source_filename: str,
        file_hash: str
    ) -> Dict[str, Any]:
        """
        Extract knowledge from PDF chunks and store in memory.
        
        Args:
            chunks: List of text chunks from PDF
            source_filename: Name of the source PDF
            file_hash: Hash of the PDF file for tracking
            
        Returns:
            Statistics about extracted knowledge
        """
        # Check if already processed
        if file_hash in self.memory["source_hashes"]:
            return {
                "status": "already_processed",
                "filename": source_filename
            }
        
        extracted_insights = 0
        extracted_arguments = 0
        
        for chunk in chunks:
            # Extract insights (key statements, facts)
            insights = self._extract_insights_from_text(chunk, source_filename)
            for topic, insight_list in insights.items():
                if topic not in self.memory["topics"]:
                    self.memory["topics"][topic] = []
                self.memory["topics"][topic].extend(insight_list)
                extracted_insights += len(insight_list)
            
            # Extract arguments (claims with supporting evidence)
            arguments = self._extract_arguments_from_text(chunk, source_filename)
            self.memory["arguments"].extend(arguments)
            extracted_arguments += len(arguments)
        
        # Mark as processed
        self.memory["source_hashes"].append(file_hash)
        self.memory["statistics"]["total_pdfs_processed"] += 1
        self.memory["statistics"]["total_insights_extracted"] += extracted_insights
        
        # Check if compression needed
        self._check_and_compress_topics()
        
        # Save
        self._save()
        
        return {
            "status": "processed",
            "filename": source_filename,
            "insights_extracted": extracted_insights,
            "arguments_extracted": extracted_arguments
        }
    
    def _extract_insights_from_text(
        self,
        text: str,
        source: str
    ) -> Dict[str, List[Dict]]:
        """
        Extract key insights from text and categorize by topic.
        Uses heuristics to identify important statements.
        """
        insights_by_topic: Dict[str, List[Dict]] = {}
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20 or len(sentence) > 500:
                continue
            
            # Identify if this is a valuable insight
            if not self._is_valuable_insight(sentence):
                continue
            
            # Detect topic from sentence
            topic = self._detect_topic(sentence)
            
            insight = {
                "content": sentence,
                "source": source,
                "weight": 1,
                "added": datetime.now().isoformat()
            }
            
            if topic not in insights_by_topic:
                insights_by_topic[topic] = []
            insights_by_topic[topic].append(insight)
        
        return insights_by_topic
    
    def _extract_arguments_from_text(
        self,
        text: str,
        source: str
    ) -> List[Dict]:
        """
        Extract arguments (claims with reasoning) from text.
        Looks for patterns indicating logical arguments.
        """
        arguments = []
        
        # Argument indicators (works for both German and English)
        argument_patterns = [
            "because", "therefore", "thus", "hence", "consequently",
            "weil", "daher", "deshalb", "folglich", "somit",
            "as a result", "this means", "it follows that",
            "das bedeutet", "daraus folgt", "dies zeigt",
            "according to", "research shows", "studies indicate",
            "laut", "studien zeigen", "forschung belegt"
        ]
        
        sentences = self._split_into_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # Check if sentence contains argument indicators
            for pattern in argument_patterns:
                if pattern in sentence_lower:
                    # Try to identify claim and evidence
                    argument = {
                        "claim": sentence.strip(),
                        "source": source,
                        "type": "logical_argument",
                        "strength": 1,
                        "added": datetime.now().isoformat()
                    }
                    
                    # Add context from surrounding sentences
                    context = []
                    if i > 0:
                        context.append(sentences[i-1].strip())
                    if i < len(sentences) - 1:
                        context.append(sentences[i+1].strip())
                    if context:
                        argument["context"] = context
                    
                    arguments.append(argument)
                    break
        
        return arguments
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _is_valuable_insight(self, sentence: str) -> bool:
        """
        Determine if a sentence contains valuable insight worth storing.
        Uses heuristics to filter out noise.
        """
        sentence_lower = sentence.lower()
        
        # Skip common noise patterns
        noise_patterns = [
            "page ", "chapter ", "table of contents", "index",
            "copyright", "all rights reserved", "isbn",
            "seite ", "kapitel ", "inhaltsverzeichnis",
            "figure ", "abbildung ", "see also", "siehe auch"
        ]
        for pattern in noise_patterns:
            if pattern in sentence_lower:
                return False
        
        # Value indicators - sentences with these are likely insights
        value_indicators = [
            "important", "key", "critical", "essential", "fundamental",
            "wichtig", "wesentlich", "entscheidend", "grundlegend",
            "is defined as", "means that", "refers to",
            "wird definiert als", "bedeutet", "bezieht sich auf",
            "the main", "a major", "significant",
            "der hauptsächliche", "ein wesentlicher", "bedeutsam"
        ]
        
        for indicator in value_indicators:
            if indicator in sentence_lower:
                return True
        
        # Check for definitional patterns
        if " is " in sentence_lower or " are " in sentence_lower:
            return True
        if " ist " in sentence_lower or " sind " in sentence_lower:
            return True
        
        # Check sentence has substantive content (enough words)
        words = sentence.split()
        if len(words) >= MIN_INSIGHT_WORDS and len(words) <= MAX_INSIGHT_WORDS:
            return True
        
        return False
    
    def _detect_topic(self, sentence: str) -> str:
        """
        Detect the topic of a sentence.
        Returns a normalized topic name.
        """
        sentence_lower = sentence.lower()
        
        # Define topic keywords
        topic_keywords = {
            "philosophy": ["philosophy", "philosophie", "ethics", "ethik", "moral", "existenz"],
            "science": ["science", "wissenschaft", "research", "forschung", "experiment", "theory", "theorie"],
            "technology": ["technology", "technologie", "computer", "software", "digital", "AI", "KI"],
            "history": ["history", "geschichte", "historical", "historisch", "century", "jahrhundert"],
            "psychology": ["psychology", "psychologie", "mental", "behavior", "verhalten", "mind", "geist"],
            "economics": ["economy", "wirtschaft", "market", "markt", "finance", "finanzen", "business"],
            "politics": ["politics", "politik", "government", "regierung", "law", "gesetz", "democracy"],
            "art": ["art", "kunst", "artist", "künstler", "creative", "kreativ", "aesthetic", "ästhetik"],
            "literature": ["literature", "literatur", "book", "buch", "author", "autor", "novel", "roman"],
            "society": ["society", "gesellschaft", "social", "sozial", "culture", "kultur", "community"]
        }
        
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in sentence_lower:
                    return topic
        
        return "general"
    
    def _check_and_compress_topics(self):
        """Check if any topic has too many insights and compress if needed."""
        for topic, insights in self.memory["topics"].items():
            if len(insights) > self.summary_threshold:
                self.memory["topics"][topic] = self._merge_similar_insights(insights)
    
    def add_core_belief(self, belief: str, source: str = "user", weight: int = 5):
        """
        Add a core belief that strongly shapes the bot's personality.
        
        Args:
            belief: The belief statement
            source: Where this belief came from
            weight: Importance weight (higher = more influential)
        """
        self.memory["core_beliefs"].append({
            "content": belief,
            "source": source,
            "weight": weight,
            "added": datetime.now().isoformat()
        })
        self._save()
    
    def get_relevant_knowledge(
        self,
        query: str,
        max_insights: int = 10,
        max_arguments: int = 5
    ) -> Dict[str, Any]:
        """
        Get knowledge relevant to a query for use in chat context.
        
        Args:
            query: The user's query
            max_insights: Maximum number of insights to return
            max_arguments: Maximum number of arguments to return
            
        Returns:
            Dict with relevant knowledge components
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Find relevant topics
        relevant_topics = []
        for topic in self.memory["topics"]:
            topic_words = set(topic.lower().split())
            if topic_words & query_words:
                relevant_topics.append(topic)
        
        # If no direct topic match, check all topics
        if not relevant_topics:
            relevant_topics = list(self.memory["topics"].keys())
        
        # Collect relevant insights
        relevant_insights = []
        for topic in relevant_topics:
            for insight in self.memory["topics"].get(topic, []):
                content_lower = insight.get("content", "").lower()
                content_words = set(content_lower.split())
                
                # Score based on word overlap
                overlap = len(query_words & content_words)
                if overlap > 0:
                    relevant_insights.append({
                        **insight,
                        "relevance_score": overlap * insight.get("weight", 1)
                    })
        
        # Sort by relevance and take top insights
        relevant_insights.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_insights = relevant_insights[:max_insights]
        
        # Find relevant arguments
        relevant_arguments = []
        for arg in self.memory["arguments"]:
            claim_lower = arg.get("claim", "").lower()
            claim_words = set(claim_lower.split())
            overlap = len(query_words & claim_words)
            if overlap > 0:
                relevant_arguments.append({
                    **arg,
                    "relevance_score": overlap * arg.get("strength", 1)
                })
        
        relevant_arguments.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        top_arguments = relevant_arguments[:max_arguments]
        
        return {
            "core_beliefs": self.memory["core_beliefs"][:5],
            "insights": top_insights,
            "arguments": top_arguments,
            "topics_matched": relevant_topics[:5]
        }
    
    def format_knowledge_for_prompt(self, query: str) -> str:
        """
        Format relevant knowledge as context for the chat prompt.
        This shapes the bot's personality based on learned knowledge.
        
        Args:
            query: The user's query
            
        Returns:
            Formatted string to include in the prompt
        """
        knowledge = self.get_relevant_knowledge(query)
        
        if not knowledge["core_beliefs"] and not knowledge["insights"] and not knowledge["arguments"]:
            return ""
        
        parts = []
        
        # Add core beliefs (most influential)
        if knowledge["core_beliefs"]:
            beliefs = [b["content"] for b in knowledge["core_beliefs"][:3]]
            parts.append("CORE BELIEFS (shapes my personality):\n" + "\n".join(f"• {b}" for b in beliefs))
        
        # Add relevant knowledge
        if knowledge["insights"]:
            insights = [i["content"] for i in knowledge["insights"][:5]]
            parts.append("LEARNED KNOWLEDGE:\n" + "\n".join(f"• {i}" for i in insights))
        
        # Add arguments for rational reasoning
        if knowledge["arguments"]:
            args = [a["claim"] for a in knowledge["arguments"][:3]]
            parts.append("ARGUMENTS I'VE LEARNED:\n" + "\n".join(f"• {a}" for a in args))
        
        if not parts:
            return ""
        
        return (
            "=== MY LEARNED KNOWLEDGE & PERSONALITY ===\n"
            "Use this knowledge to reason rationally, compare arguments, and form coherent opinions.\n\n"
            + "\n\n".join(parts)
            + "\n=== END LEARNED KNOWLEDGE ==="
        )
    
    def compare_arguments(self, topic: str) -> List[Dict]:
        """
        Compare different arguments learned about a topic.
        Useful for rational, human-like reasoning.
        
        Args:
            topic: Topic to find arguments about
            
        Returns:
            List of related arguments for comparison
        """
        topic_lower = topic.lower()
        related_args = []
        
        for arg in self.memory["arguments"]:
            claim_lower = arg.get("claim", "").lower()
            if topic_lower in claim_lower:
                related_args.append(arg)
        
        # Sort by strength
        related_args.sort(key=lambda x: x.get("strength", 1), reverse=True)
        return related_args[:10]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        total_insights = sum(
            len(insights) for insights in self.memory["topics"].values()
        )
        
        # Calculate file size
        file_size_bytes = 0
        if self.memory_file.exists():
            file_size_bytes = self.memory_file.stat().st_size
        
        return {
            "version": self.memory["version"],
            "created": self.memory["created"],
            "updated": self.memory["updated"],
            "total_pdfs_processed": self.memory["statistics"]["total_pdfs_processed"],
            "total_insights": total_insights,
            "total_arguments": len(self.memory["arguments"]),
            "total_core_beliefs": len(self.memory["core_beliefs"]),
            "topics_count": len(self.memory["topics"]),
            "topics": list(self.memory["topics"].keys()),
            "compressions_performed": self.memory["statistics"]["compressions_performed"],
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 3)
        }
    
    def clear(self):
        """Clear all knowledge memory."""
        self.memory = {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "statistics": {
                "total_pdfs_processed": 0,
                "total_insights_extracted": 0,
                "compressions_performed": 0
            },
            "topics": {},
            "core_beliefs": [],
            "arguments": [],
            "source_hashes": []
        }
        self._save()
    
    def export_knowledge(self) -> Dict[str, Any]:
        """Export knowledge memory for backup or inspection."""
        return {
            **self.memory,
            "export_date": datetime.now().isoformat()
        }
    
    def import_knowledge(self, data: Dict[str, Any]):
        """Import knowledge from backup."""
        if "topics" in data:
            for topic, insights in data["topics"].items():
                if topic not in self.memory["topics"]:
                    self.memory["topics"][topic] = []
                self.memory["topics"][topic].extend(insights)
        
        if "arguments" in data:
            self.memory["arguments"].extend(data["arguments"])
        
        if "core_beliefs" in data:
            self.memory["core_beliefs"].extend(data["core_beliefs"])
        
        self._check_and_compress_topics()
        self._save()
