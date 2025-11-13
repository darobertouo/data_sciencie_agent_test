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

"""This file contains the tools used by the database agent."""

import datetime
import logging
import os

import numpy as np
import pandas as pd
from data_science.utils.utils import get_env_var, USER_AGENT
from google.adk.tools import ToolContext
from google.adk.tools.bigquery.client import get_bigquery_client
from google.cloud import bigquery
from google.genai import Client
from google.genai.types import HttpOptions

from .chase_sql import chase_constants
from ...utils.utils import USER_AGENT

logger = logging.getLogger(__name__)

from vertexai.preview.language_models import TextGenerationModel

# Assume that `BQ_COMPUTE_PROJECT_ID` and `BQ_DATA_PROJECT_ID` are set in the
# environment. See the `data_agent` README for more details.
dataset_id = get_env_var("BQ_DATASET_ID")
data_project = get_env_var("BQ_DATA_PROJECT_ID")
compute_project = get_env_var("BQ_COMPUTE_PROJECT_ID")
vertex_project = get_env_var("GOOGLE_CLOUD_PROJECT")
location = get_env_var("GOOGLE_CLOUD_LOCATION")
http_options = HttpOptions(headers={"user-agent": USER_AGENT})
llm_client = Client(
    vertexai=True,
    project=vertex_project,
    location=location,
    http_options=http_options,
)


def analyze_sentiment_topics(texts, model_name="gemini-2.5-pro"):
    """Usa o modelo Gemini para classificar sentimento e tópicos."""
    model = TextGenerationModel.from_pretrained(model_name)
    results = []
    for text in texts:
        prompt = f"Analise o texto e retorne JSON com sentimento e tópicos:\n{text}"
        response = model.predict(prompt)
        results.append(response.text)
    return results




 
