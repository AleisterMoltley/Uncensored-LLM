"""
Knowledge Memory System for LLM Servant
Stores learned knowledge from PDFs to shape bot personality.
Uses compression to keep memory size bounded even after many PDFs.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict

from pydantic import BaseModel, Field, field_validator, ConfigDict
from utils import PersistentStorage


# ============================================================
# TypedDict Definitions for Configurations and Return Types
# ============================================================


class KnowledgeMemoryConfig(TypedDict, total=False):
    """Configuration dictionary for KnowledgeMemory initialization."""
    max_knowledge_memory_mb: float
    max_insights_per_topic: int
    summary_threshold: int


class ExtractionResult(TypedDict):
    """Return type for extract_knowledge_from_chunks."""
    status: str
    filename: str
    insights_extracted: int
    arguments_extracted: int


class AlreadyProcessedResult(TypedDict):
    """Return type when a PDF has already been processed."""
    status: str
    filename: str


class RelevantKnowledgeResult(TypedDict):
    """Return type for get_relevant_knowledge."""
    core_beliefs: List[Dict[str, Any]]
    insights: List[Dict[str, Any]]
    arguments: List[Dict[str, Any]]
    topics_matched: List[str]


class StatisticsResult(TypedDict):
    """Return type for get_statistics."""
    version: str
    created: str
    updated: str
    total_pdfs_processed: int
    total_insights: int
    total_arguments: int
    total_core_beliefs: int
    topics_count: int
    topics: List[str]
    compressions_performed: int
    file_size_bytes: int
    file_size_mb: float


class ExportResult(TypedDict):
    """Return type for export_knowledge."""
    version: str
    created: str
    updated: str
    statistics: Dict[str, int]
    topics: Dict[str, List[Dict[str, Any]]]
    core_beliefs: List[Dict[str, Any]]
    arguments: List[Dict[str, Any]]
    source_hashes: List[str]
    export_date: str


# ============================================================
# Pydantic Models for Data Validation
# ============================================================


class InsightModel(BaseModel):
    """Pydantic model for validating insight entries."""
    content: str = Field(..., min_length=1)
    source: str = Field(default="unknown")
    weight: int = Field(default=1, ge=1)
    added: str = Field(default_factory=lambda: datetime.now().isoformat())
    sources: Optional[List[str]] = None
    merged_count: Optional[int] = None
    relevance_score: Optional[float] = None

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is stripped of leading/trailing whitespace."""
        return v.strip()


class ArgumentModel(BaseModel):
    """Pydantic model for validating argument entries."""
    claim: str = Field(..., min_length=1)
    source: str = Field(default="unknown")
    type: str = Field(default="logical_argument")
    strength: int = Field(default=1, ge=1)
    added: str = Field(default_factory=lambda: datetime.now().isoformat())
    context: Optional[List[str]] = None
    relevance_score: Optional[float] = None

    @field_validator('claim')
    @classmethod
    def validate_claim(cls, v: str) -> str:
        """Ensure claim is stripped of leading/trailing whitespace."""
        return v.strip()


class CoreBeliefModel(BaseModel):
    """Pydantic model for validating core belief entries."""
    content: str = Field(..., min_length=1)
    source: str = Field(default="user")
    weight: int = Field(default=5, ge=1)
    added: str = Field(default_factory=lambda: datetime.now().isoformat())

    @field_validator('content')
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Ensure content is stripped of leading/trailing whitespace."""
        return v.strip()


class StatisticsModel(BaseModel):
    """Pydantic model for validating statistics."""
    total_pdfs_processed: int = Field(default=0, ge=0)
    total_insights_extracted: int = Field(default=0, ge=0)
    compressions_performed: int = Field(default=0, ge=0)


class MemoryModel(BaseModel):
    """Pydantic model for validating the entire memory structure."""
    version: str = Field(default="1.0")
    created: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated: str = Field(default_factory=lambda: datetime.now().isoformat())
    statistics: StatisticsModel = Field(default_factory=StatisticsModel)
    topics: Dict[str, List[InsightModel]] = Field(default_factory=dict)
    core_beliefs: List[CoreBeliefModel] = Field(default_factory=list)
    arguments: List[ArgumentModel] = Field(default_factory=list)
    source_hashes: List[str] = Field(default_factory=list)

    model_config = ConfigDict(validate_assignment=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for storage."""
        data = self.model_dump()
        # Convert nested models to dicts
        data["statistics"] = self.statistics.model_dump()
        data["topics"] = {
            topic: [insight.model_dump() for insight in insights]
            for topic, insights in self.topics.items()
        }
        data["core_beliefs"] = [belief.model_dump() for belief in self.core_beliefs]
        data["arguments"] = [arg.model_dump() for arg in self.arguments]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryModel":
        """Create model from dictionary (handles raw dict data)."""
        # Parse statistics
        statistics = StatisticsModel(**data.get("statistics", {}))

        # Parse topics
        topics: Dict[str, List[InsightModel]] = {}
        for topic, insights in data.get("topics", {}).items():
            topics[topic] = [InsightModel(**insight) for insight in insights]

        # Parse core beliefs
        core_beliefs = [
            CoreBeliefModel(**belief) for belief in data.get("core_beliefs", [])
        ]

        # Parse arguments
        arguments = [
            ArgumentModel(**arg) for arg in data.get("arguments", [])
        ]

        return cls(
            version=data.get("version", "1.0"),
            created=data.get("created", datetime.now().isoformat()),
            updated=data.get("updated", datetime.now().isoformat()),
            statistics=statistics,
            topics=topics,
            core_beliefs=core_beliefs,
            arguments=arguments,
            source_hashes=data.get("source_hashes", [])
        )

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
    
    def __init__(
        self,
        memory_dir: Path,
        config: Optional[KnowledgeMemoryConfig] = None
    ):
        """
        Initialize knowledge memory.
        
        Args:
            memory_dir: Directory to store memory files
            config: Configuration TypedDict with optional settings
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        
        self.config: KnowledgeMemoryConfig = config or {}
        self.max_size_mb = self.config.get("max_knowledge_memory_mb", DEFAULT_MAX_MEMORY_SIZE_MB)
        self.max_insights_per_topic = self.config.get("max_insights_per_topic", DEFAULT_MAX_INSIGHTS_PER_TOPIC)
        self.summary_threshold = self.config.get("summary_threshold", DEFAULT_SUMMARY_THRESHOLD)
        
        self.memory_file = self.memory_dir / "knowledge_memory.json.gz"
        
        # Create default memory model for initialization
        self._default_memory_model = MemoryModel()
        
        # Internal memory dict (for compatibility with existing code)
        # Uses Pydantic model for validation
        self._memory_model: MemoryModel = MemoryModel()
        self.memory: Dict[str, Any] = self._memory_model.to_dict()
        
        # Initialize persistent storage with compression callback
        self._storage = PersistentStorage(
            self.memory_file,
            max_size_mb=self.max_size_mb,
            on_size_exceeded=self._compress_memory_callback
        )
        self._load()
    
    def _compress_memory_callback(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Callback for PersistentStorage when size limit is exceeded.
        Compresses the memory data and returns the compressed version.
        """
        # Update local memory, compress it, and return
        self.memory = data
        self._compress_memory()
        return self.memory
    
    def _load(self) -> None:
        """Load memory from compressed file with Pydantic validation."""
        loaded = self._storage.load(default={})
        if loaded:
            try:
                # Validate and parse loaded data using Pydantic model
                self._memory_model = MemoryModel.from_dict(loaded)
                self.memory = self._memory_model.to_dict()
            except Exception:
                # Fall back to default memory if validation fails
                self._memory_model = MemoryModel()
                self.memory = self._memory_model.to_dict()
        else:
            # Use default memory
            self._memory_model = MemoryModel()
            self.memory = self._memory_model.to_dict()
    
    def _save(self) -> None:
        """Save memory to compressed file with size check."""
        self.memory["updated"] = datetime.now().isoformat()
        self._storage.save(self.memory)
    
    def _validate_memory(self) -> bool:
        """
        Validate current memory state using Pydantic model.
        
        Returns:
            True if memory is valid, False otherwise
        """
        try:
            self._memory_model = MemoryModel.from_dict(self.memory)
            return True
        except Exception:
            return False
    
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
    
    def _merge_similar_insights(
        self, insights: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge insights with similar content to reduce redundancy."""
        if len(insights) <= 5:
            return insights
        
        merged: List[Dict[str, Any]] = []
        used_indices: set[int] = set()
        
        for i, insight in enumerate(insights):
            if i in used_indices:
                continue
            
            # Find similar insights
            similar_group: List[Dict[str, Any]] = [insight]
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
                merged_insight: Dict[str, Any] = {
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
    
    def _consolidate_beliefs(self, beliefs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate beliefs by merging similar ones and boosting weights."""
        return self._merge_similar_insights(beliefs)
    
    def extract_knowledge_from_chunks(
        self,
        chunks: List[str],
        source_filename: str,
        file_hash: str
    ) -> ExtractionResult | AlreadyProcessedResult:
        """
        Extract knowledge from PDF chunks and store in memory.
        
        Args:
            chunks: List of text chunks from PDF
            source_filename: Name of the source PDF
            file_hash: Hash of the PDF file for tracking
            
        Returns:
            ExtractionResult with statistics about extracted knowledge,
            or AlreadyProcessedResult if the PDF was already processed
        """
        # Check if already processed
        if file_hash in self.memory["source_hashes"]:
            result: AlreadyProcessedResult = {
                "status": "already_processed",
                "filename": source_filename
            }
            return result
        
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
        
        result: ExtractionResult = {
            "status": "processed",
            "filename": source_filename,
            "insights_extracted": extracted_insights,
            "arguments_extracted": extracted_arguments
        }
        return result
    
    def _extract_insights_from_text(
        self,
        text: str,
        source: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract key insights from text and categorize by topic.
        Uses heuristics to identify important statements.
        """
        insights_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        
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
    ) -> List[Dict[str, Any]]:
        """
        Extract arguments (claims with reasoning) from text.
        Looks for patterns indicating logical arguments.
        """
        arguments: List[Dict[str, Any]] = []
        
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
                    argument: Dict[str, Any] = {
                        "claim": sentence.strip(),
                        "source": source,
                        "type": "logical_argument",
                        "strength": 1,
                        "added": datetime.now().isoformat()
                    }
                    
                    # Add context from surrounding sentences
                    context: List[str] = []
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
    
    def _check_and_compress_topics(self) -> None:
        """Check if any topic has too many insights and compress if needed."""
        for topic, insights in self.memory["topics"].items():
            if len(insights) > self.summary_threshold:
                self.memory["topics"][topic] = self._merge_similar_insights(insights)
    
    def add_core_belief(self, belief: str, source: str = "user", weight: int = 5) -> None:
        """
        Add a core belief that strongly shapes the bot's personality.
        
        Args:
            belief: The belief statement
            source: Where this belief came from
            weight: Importance weight (higher = more influential)
        """
        # Validate using Pydantic model
        validated_belief = CoreBeliefModel(
            content=belief,
            source=source,
            weight=weight
        )
        self.memory["core_beliefs"].append(validated_belief.model_dump())
        self._save()
    
    def get_relevant_knowledge(
        self,
        query: str,
        max_insights: int = 10,
        max_arguments: int = 5
    ) -> RelevantKnowledgeResult:
        """
        Get knowledge relevant to a query for use in chat context.
        
        Args:
            query: The user's query
            max_insights: Maximum number of insights to return
            max_arguments: Maximum number of arguments to return
            
        Returns:
            RelevantKnowledgeResult with relevant knowledge components
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
        relevant_arguments: List[Dict[str, Any]] = []
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
        
        result: RelevantKnowledgeResult = {
            "core_beliefs": self.memory["core_beliefs"][:5],
            "insights": top_insights,
            "arguments": top_arguments,
            "topics_matched": relevant_topics[:5]
        }
        return result
    
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
    
    def compare_arguments(self, topic: str) -> List[Dict[str, Any]]:
        """
        Compare different arguments learned about a topic.
        Useful for rational, human-like reasoning.
        
        Args:
            topic: Topic to find arguments about
            
        Returns:
            List of related arguments for comparison
        """
        topic_lower = topic.lower()
        related_args: List[Dict[str, Any]] = []
        
        for arg in self.memory["arguments"]:
            claim_lower = arg.get("claim", "").lower()
            if topic_lower in claim_lower:
                related_args.append(arg)
        
        # Sort by strength
        related_args.sort(key=lambda x: x.get("strength", 1), reverse=True)
        return related_args[:10]
    
    def get_statistics(self) -> StatisticsResult:
        """Get memory statistics."""
        total_insights = sum(
            len(insights) for insights in self.memory["topics"].values()
        )
        
        # Calculate file size
        file_size_bytes = 0
        if self.memory_file.exists():
            file_size_bytes = self.memory_file.stat().st_size
        
        result: StatisticsResult = {
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
        return result
    
    def clear(self) -> None:
        """Clear all knowledge memory."""
        self._memory_model = MemoryModel()
        self.memory = self._memory_model.to_dict()
        self._save()
    
    def export_knowledge(self) -> ExportResult:
        """Export knowledge memory for backup or inspection."""
        result: ExportResult = {
            "version": self.memory["version"],
            "created": self.memory["created"],
            "updated": self.memory["updated"],
            "statistics": self.memory["statistics"],
            "topics": self.memory["topics"],
            "core_beliefs": self.memory["core_beliefs"],
            "arguments": self.memory["arguments"],
            "source_hashes": self.memory["source_hashes"],
            "export_date": datetime.now().isoformat()
        }
        return result
    
    def import_knowledge(self, data: Dict[str, Any]) -> None:
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
