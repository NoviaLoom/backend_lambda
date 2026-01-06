"""
Theme models for shared use
Modèles SQLAlchemy pour business.themes et prompts associés
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON, UUID as SQLUUID, ARRAY, Numeric
from sqlalchemy.orm import relationship, foreign
from sqlalchemy.sql import func

from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Theme(Base):
    """
    Thème d'audit (business.themes)
    """
    __tablename__ = 'themes'
    __table_args__ = {'schema': 'business'}
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False)  # Slug requis par la DB
    description = Column(Text, nullable=True)
    theme_type = Column(String(50), nullable=False, default='b2b_audit')  # Type requis par la DB
    
    # Client associé (NULL pour thèmes système)
    # Note: Foreign key désactivée temporairement pour éviter les erreurs de référence
    client_id = Column(SQLUUID(as_uuid=True), nullable=True)
    
    # Statut (utiliser status de la DB au lieu de is_active)
    # Note: La table DB utilise 'status' (varchar) au lieu de 'is_active' (boolean)
    # On mappe status='active' comme is_active=True
    status = Column(String(50), nullable=True, default='active')
    is_system_default = Column(Boolean, default=False, nullable=False)  # Phase 0: catalogue B2C
    
    # Configuration additionnelle (JSONB)
    config = Column(JSON, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    # Note: Foreign key désactivée temporairement pour éviter les erreurs de référence
    created_by = Column(SQLUUID(as_uuid=True), nullable=True)
    
    # Relationships (primaryjoin avec foreign() car FK désactivée)
    captation_prompts = relationship(
        "ThemeCaptationPrompt",
        back_populates="theme",
        cascade="all, delete-orphan",
        primaryjoin="Theme.id == foreign(ThemeCaptationPrompt.theme_id)"
    )
    analyzer_prompts = relationship(
        "ThemeAnalyzerPrompt",
        back_populates="theme",
        cascade="all, delete-orphan",
        primaryjoin="Theme.id == foreign(ThemeAnalyzerPrompt.theme_id)"
    )
    
    # Propriété pour mapper status vers is_active (compatibilité avec l'API)
    @property
    def is_active(self) -> bool:
        """Mappe status vers is_active pour compatibilité API"""
        return self.status == 'active' if self.status else True
    
    @is_active.setter
    def is_active(self, value: bool):
        """Setter pour is_active qui met à jour status"""
        self.status = 'active' if value else 'inactive'


class ThemeCaptationPrompt(Base):
    """
    Prompts de captation pour un thème (business.theme_captation_prompts)
    """
    __tablename__ = 'theme_captation_prompts'
    __table_args__ = {'schema': 'business'}
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    theme_id = Column(SQLUUID(as_uuid=True), nullable=False)  # FK désactivée temporairement
    
    # Ordre d'exécution (nom réel dans la DB: prompt_number)
    prompt_number = Column(Integer, nullable=False, default=0)
    prompt_name = Column(String(255), nullable=False)
    
    # Contenu du prompt (nom réel dans la DB: prompt_template)
    prompt_template = Column(Text, nullable=False)
    system_message = Column(Text, nullable=True)
    
    # Configuration LLM
    model = Column(String(100), nullable=True, default='gemini-2.5-flash')
    # Note: temperature est numeric(3,2) dans la DB
    temperature = Column(Numeric(3, 2), nullable=True, default=0.7)
    max_tokens = Column(Integer, nullable=True)
    use_search = Column(Boolean, nullable=True, default=False)
    
    # Type de captation (nouveau)
    # Valeurs possibles: 'llm', 'llm_search', 'llm_maps', 'api_insee', 'api_datagouv', 'api_weather', 'api_custom'
    captation_type = Column(String(50), nullable=False, default='llm')
    
    # Configuration pour les collecteurs API externes (nouveau)
    # Exemple: {"collector_type": "b2c_market", "params": {"region": "fr", "metrics": ["size"]}}
    collector_config = Column(JSON, nullable=True)
    
    # Dépendances (JSON dans la DB)
    depends_on_prompts = Column(JSON, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    expected_output = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship (primaryjoin avec foreign() car FK désactivée)
    theme = relationship(
        "Theme",
        back_populates="captation_prompts",
        primaryjoin="foreign(ThemeCaptationPrompt.theme_id) == Theme.id"
    )
    
    # Propriétés de compatibilité pour l'API
    @property
    def order(self) -> int:
        """Mappe prompt_number vers order pour compatibilité API"""
        return self.prompt_number
    
    @property
    def content(self) -> str:
        """Mappe prompt_template vers content pour compatibilité API"""
        return self.prompt_template


class ThemeAnalyzerPrompt(Base):
    """
    Prompts d'analyse pour un thème (business.theme_analyzer_prompts)
    """
    __tablename__ = 'theme_analyzer_prompts'
    __table_args__ = {'schema': 'business'}
    
    id = Column(SQLUUID(as_uuid=True), primary_key=True, default=uuid4)
    theme_id = Column(SQLUUID(as_uuid=True), nullable=False)  # FK désactivée temporairement
    
    # Ordre d'exécution (nom réel dans la DB: processor_number)
    processor_number = Column(Integer, nullable=False, default=0)
    processor_name = Column(String(255), nullable=False)
    
    # Contenu du prompt (nom réel dans la DB: prompt_template)
    prompt_template = Column(Text, nullable=False)
    system_message = Column(Text, nullable=True)
    
    # Configuration LLM
    model = Column(String(100), nullable=True, default='gemini-2.5-flash')
    # Note: temperature est numeric(3,2) dans la DB
    temperature = Column(Numeric(3, 2), nullable=True, default=0.7)
    max_tokens = Column(Integer, nullable=True)
    use_search = Column(Boolean, nullable=True, default=False)
    
    # Dépendances (JSONB dans la DB)
    depends_on_prompts = Column(JSON, nullable=True)
    depends_on_processors = Column(JSON, nullable=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    expected_sections = Column(JSON, nullable=True)
    output_schema = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationship (primaryjoin avec foreign() car FK désactivée)
    theme = relationship(
        "Theme",
        back_populates="analyzer_prompts",
        primaryjoin="foreign(ThemeAnalyzerPrompt.theme_id) == Theme.id"
    )
    
    # Propriétés de compatibilité pour l'API
    @property
    def order(self) -> int:
        """Mappe processor_number vers order pour compatibilité API"""
        return self.processor_number
    
    @property
    def content(self) -> str:
        """Mappe prompt_template vers content pour compatibilité API"""
        return self.prompt_template

