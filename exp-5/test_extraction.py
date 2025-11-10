"""
Quick test to verify explanation extraction logic works correctly
"""

import json
import sys

def extract_explanations_from_item(item: dict, explanation_model: str):
    """
    Extract explanations for all 4 options from a question item.
    """
    if 'explanations' not in item:
        return None

    explanations_list = []

    # Get explanations in order (0-3)
    for option_idx in range(4):
        # Find the explanation for this option index
        option_explanation = None
        for exp_item in item['explanations']:
            if exp_item.get('option_index') == option_idx:
                # Get the explanation from the specified model
                llm_explanations = exp_item.get('llm_explanations', {})
                if explanation_model in llm_explanations:
                    option_explanation = llm_explanations[explanation_model].get('explanation')
                break

        if option_explanation is None:
            # Missing explanation for this option
            return None

        explanations_list.append(option_explanation)

    if len(explanations_list) != 4:
        return None

    return explanations_list


def test_extraction():
    """Test explanation extraction on the first question"""
    dataset_path = '/Users/arman/code/Persian-Poetry-Semantic-Similarity-Benchmark/exp-4/benchmark_dataset_with_explanations.json'

    print("Testing explanation extraction...")
    print("="*60)

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        print(f"✓ Loaded dataset: {len(dataset)} questions")
    except FileNotFoundError:
        print(f"✗ Dataset not found at: {dataset_path}")
        return False
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False

    # Test with first item
    if not dataset:
        print("✗ Dataset is empty")
        return False

    first_item = dataset[0]
    question_id = first_item.get('id', 'unknown')

    print(f"\nTesting with question ID: {question_id}")

    # Try extracting with default model
    explanation_model = 'google/gemini-2.5-flash'
    print(f"Explanation model: {explanation_model}")

    explanations = extract_explanations_from_item(first_item, explanation_model)

    if explanations is None:
        print("✗ Failed to extract explanations")
        return False

    if len(explanations) != 4:
        print(f"✗ Wrong number of explanations: {len(explanations)} (expected 4)")
        return False

    print(f"✓ Successfully extracted {len(explanations)} explanations")

    # Display sample
    print("\n" + "="*60)
    print("Sample Explanations:")
    print("="*60)
    for i, exp in enumerate(explanations):
        print(f"\nOption {i}:")
        print(f"  {exp[:100]}..." if len(exp) > 100 else f"  {exp}")

    print("\n" + "="*60)
    print("✓ Extraction test PASSED")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_extraction()
    sys.exit(0 if success else 1)
