#!/usr/bin/env python3
"""
Batch Processing Example for OpenMed NER Species Detection

This script demonstrates how to process multiple medical texts efficiently
using the OpenMed NER Species Detection model from AWS Marketplace.
"""

import boto3
import json
import pandas as pd
from typing import List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

class OpenMedSpeciesDetector:
    """
    A class for batch processing medical texts using OpenMed NER Species Detection model.
    """

    def __init__(self, endpoint_name: str, region: str = 'us-east-1'):
        """
        Initialize the species detector.

        Args:
            endpoint_name: Name of the deployed SageMaker endpoint
            region: AWS region where the endpoint is deployed
        """
        self.endpoint_name = endpoint_name
        self.region = region
        self.runtime = boto3.client('sagemaker-runtime', region_name=region)

    def predict_single(self, text: str) -> List[Dict[str, Any]]:
        """
        Predict species entities in a single text.

        Args:
            text: Input medical text

        Returns:
            List of detected species entities
        """
        try:
            payload = json.dumps({"inputs": text})

            response = self.runtime.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=payload
            )

            result = json.loads(response['Body'].read().decode())
            return result

        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return []

    def predict_batch(self, texts: List[str], max_workers: int = 5) -> List[Dict[str, Any]]:
        """
        Process multiple texts in parallel.

        Args:
            texts: List of medical texts to process
            max_workers: Maximum number of parallel workers

        Returns:
            List of results for each text
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all texts for processing
            future_to_text = {
                executor.submit(self.predict_single, text): (i, text)
                for i, text in enumerate(texts)
            }

            # Collect results as they complete
            for future in as_completed(future_to_text):
                text_index, original_text = future_to_text[future]
                try:
                    entities = future.result()
                    results.append({
                        'index': text_index,
                        'text': original_text,
                        'entities': entities,
                        'species_count': len(entities),
                        'status': 'success'
                    })
                except Exception as e:
                    results.append({
                        'index': text_index,
                        'text': original_text,
                        'entities': [],
                        'species_count': 0,
                        'status': f'error: {str(e)}'
                    })

        # Sort results by original index
        results.sort(key=lambda x: x['index'])
        return results

    def process_file(self, file_path: str, text_column: str = 'text') -> pd.DataFrame:
        """
        Process texts from a CSV file.

        Args:
            file_path: Path to CSV file containing texts
            text_column: Name of the column containing text data

        Returns:
            DataFrame with processing results
        """
        # Read input file
        df = pd.read_csv(file_path)

        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in file")

        texts = df[text_column].tolist()

        print(f"Processing {len(texts)} texts...")
        start_time = time.time()

        # Process in batches
        results = self.predict_batch(texts)

        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")

        # Create results DataFrame
        processed_data = []
        for result in results:
            for entity in result['entities']:
                processed_data.append({
                    'original_index': result['index'],
                    'original_text': result['text'],
                    'species': entity['word'],
                    'confidence': entity['score'],
                    'start_position': entity['start'],
                    'end_position': entity['end'],
                    'status': result['status']
                })

        return pd.DataFrame(processed_data)

def main():
    """
    Example usage of the batch processing functionality.
    """
    # Configuration
    ENDPOINT_NAME = "openmed-ner-species-detection-endpoint"  # Update with your endpoint

    # Sample medical texts
    sample_texts = [
        "Patient diagnosed with pneumonia caused by Streptococcus pneumoniae.",
        "Blood culture positive for Escherichia coli and Staphylococcus aureus.",
        "Wound infection with Pseudomonas aeruginosa resistant to multiple antibiotics.",
        "Candida albicans isolated from respiratory specimen.",
        "Microbiome analysis shows Lactobacillus acidophilus and Bifidobacterium longum.",
        "MRSA (methicillin-resistant Staphylococcus aureus) infection in surgical site.",
        "Clostridium difficile-associated diarrhea following antibiotic treatment.",
        "Helicobacter pylori detected in gastric biopsy specimen.",
        "Aspergillus fumigatus infection in immunocompromised patient.",
        "Neisseria gonorrhoeae confirmed by nucleic acid amplification testing."
    ]

    # Initialize detector
    detector = OpenMedSpeciesDetector(ENDPOINT_NAME)

    print("=== OpenMed NER Species Detection - Batch Processing Example ===")
    print(f"Processing {len(sample_texts)} sample texts...")

    # Process texts
    start_time = time.time()
    results = detector.predict_batch(sample_texts)
    end_time = time.time()

    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Average time per text: {(end_time - start_time) / len(sample_texts):.3f} seconds")

    # Analyze results
    total_species = sum(result['species_count'] for result in results)
    successful_predictions = sum(1 for result in results if result['status'] == 'success')

    print(f"\n=== Results Summary ===")
    print(f"Texts processed: {len(sample_texts)}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Total species detected: {total_species}")
    print(f"Average species per text: {total_species / len(sample_texts):.2f}")

    # Detailed results
    print(f"\n=== Detailed Results ===")
    for i, result in enumerate(results):
        print(f"\nText {i+1}: {result['text'][:60]}...")
        print(f"Status: {result['status']}")
        print(f"Species found: {result['species_count']}")

        if result['entities']:
            for entity in result['entities']:
                print(f"  - {entity['word']} (confidence: {entity['score']:.3f})")

    # Create summary DataFrame
    all_entities = []
    for result in results:
        for entity in result['entities']:
            all_entities.append({
                'text_index': result['index'],
                'species': entity['word'],
                'confidence': entity['score']
            })

    if all_entities:
        df_entities = pd.DataFrame(all_entities)

        print(f"\n=== Species Analysis ===")
        print("Most common species:")
        species_counts = df_entities['species'].value_counts().head(5)
        for species, count in species_counts.items():
            avg_confidence = df_entities[df_entities['species'] == species]['confidence'].mean()
            print(f"  {species}: {count} occurrences (avg confidence: {avg_confidence:.3f})")

        print(f"\nConfidence statistics:")
        print(f"  Mean confidence: {df_entities['confidence'].mean():.3f}")
        print(f"  Min confidence: {df_entities['confidence'].min():.3f}")
        print(f"  Max confidence: {df_entities['confidence'].max():.3f}")

    print("\n=== Batch Processing Complete ===")

if __name__ == "__main__":
    main()