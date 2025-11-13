# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Database Agent: Agente para análise de sentimento e extração de tópicos em textos de reclamações e interações."""

import logging
import os
from typing import Any, Dict, Optional

from ...utils.utils import get_env_var, USER_AGENT
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import BaseTool, ToolContext
from google.adk.tools.bigquery import BigQueryToolset
from google.adk.tools.bigquery.config import BigQueryToolConfig, WriteMode
from google.genai import types
from . import tools
from .chase_sql import chase_db_tools
from .prompts import return_instructions_bigquery

logger = logging.getLogger(__name__)

from adk import Agent, Message
from .prompts import SYSTEM_PROMPT
from .tools import analyze_sentiment_topics

social_listening_agent = Agent(
    name="social_listening_agent",
    description="Agente para análise de sentimento e extração de tópicos em textos de reclamações e interações.",
    instructions=SYSTEM_PROMPT,
    tools=[analyze_sentiment_topics],
)
