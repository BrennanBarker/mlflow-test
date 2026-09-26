"""Augment a gold dataset with corruptions intended to correspond to hallucinatory behavior"""
import pandas as pd
import openai

sample = pd.read_parquet('data/xsum/test.parquet').sample(20)

client = openai.OpenAI()

prompt = """You are assisting a synthetic data generation process to 
produce a dataset to support the creation of a faithfullness model judge.

You are being provided with a source *document* and a *summary* from a gold dataset.

Your task is to corrupt this data by making a single change to the source "document"
that removes all support from a claim in the summary.
To produce labeled examples of hallucinatory or fabricated summaries, 
you will remove details from the source document 
"""
