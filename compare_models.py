#!/usr/bin/env python3
"""
Test and compare base model vs fine-tuned adapter performance.
"""

import os
import sys
import torch
import pandas as pd
from transformers import pipeline, RobertaTokenizerFast

def test_base_model(test_texts):
    """Test base RoBERTa model without adapter."""
    print("\n" + "=" * 70)
    print("Testing BASE MODEL")
    print("=" * 70)
    
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=device
        )
        
        print("✅ Base model loaded")
        results = []
        
        for text in test_texts:
            result = pipe(text)[0]
            label = result['label'].lower()
            
            # Normalize label
            if 'pos' in label:
                sentiment = 'positive'
            elif 'neg' in label:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            results.append({
                'text': text[:60] + '...' if len(text) > 60 else text,
                'sentiment': sentiment,
                'confidence': result['score']
            })
            
            print(f"\n  {text[:60]}...")
            print(f"  → {sentiment.upper()} (conf: {result['score']:.4f})")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

def test_adapter_model(test_texts, adapter_path):
    """Test fine-tuned adapter model."""
    print("\n" + "=" * 70)
    print("Testing ADAPTER MODEL")
    print("=" * 70)
    print(f"Adapter path: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        print(f"❌ Adapter not found: {adapter_path}")
        return []
    
    try:
        from adapters import AutoAdapterModel
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with adapter
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        model = AutoAdapterModel.from_pretrained(model_name)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        
        # Load and activate adapter
        adapter_name = "sentiment_adapter"
        model.load_adapter(adapter_path, load_as=adapter_name, set_active=True)
        model.set_active_adapters(adapter_name)
        model.to(device)
        model.eval()
        
        print(f"✅ Adapter loaded")
        print(f"   Active: {model.active_adapters}")
        
        # Load label mapping if available
        import json
        label_mapping_path = os.path.join(adapter_path, "..", "artifacts", "label_mapping.json")
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                label_info = json.load(f)
                id2label = {int(k): v for k, v in label_info.get("id2label", {}).items()}
        else:
            id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        results = []
        
        for text in test_texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=200)
            enc = {k: v.to(device) for k, v in enc.items()}
            
            with torch.no_grad():
                outputs = model(**enc)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_id = probs.argmax(dim=-1).item()
                confidence = probs.max().item()
            
            sentiment = id2label.get(pred_id, f"LABEL_{pred_id}")
            
            results.append({
                'text': text[:60] + '...' if len(text) > 60 else text,
                'sentiment': sentiment,
                'confidence': confidence
            })
            
            print(f"\n  {text[:60]}...")
            print(f"  → {sentiment.upper()} (conf: {confidence:.4f})")
        
        return results
        
    except ImportError:
        print("❌ 'adapter-transformers' not installed")
        print("   Install with: pip install adapter-transformers")
        return []
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return []

def compare_results(base_results, adapter_results):
    """Compare base model vs adapter results."""
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    if not base_results or not adapter_results:
        print("❌ Cannot compare - missing results")
        return
    
    print(f"\n{'Text':<65} {'Base':<15} {'Adapter':<15} {'Match':<10}")
    print("-" * 110)
    
    matches = 0
    total = len(base_results)
    confidence_diff = []
    
    for base, adapter in zip(base_results, adapter_results):
        match = "✅ YES" if base['sentiment'] == adapter['sentiment'] else "❌ NO"
        if base['sentiment'] == adapter['sentiment']:
            matches += 1
        
        conf_diff = adapter['confidence'] - base['confidence']
        confidence_diff.append(conf_diff)
        
        print(f"{base['text']:<65} {base['sentiment'][:12]:<15} {adapter['sentiment'][:12]:<15} {match:<10}")
        print(f"{'':65} ({base['confidence']:.3f})      ({adapter['confidence']:.3f})      (Δ {conf_diff:+.3f})")
        print()
    
    print("=" * 110)
    print(f"\nAgreement: {matches}/{total} ({matches/total*100:.1f}%)")
    
    avg_conf_diff = sum(confidence_diff) / len(confidence_diff)
    print(f"Avg confidence difference: {avg_conf_diff:+.4f}")
    
    if avg_conf_diff > 0.05:
        print("  → Adapter is MORE confident on average")
    elif avg_conf_diff < -0.05:
        print("  → Base model is MORE confident on average")
    else:
        print("  → Similar confidence levels")
    
    print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare base model vs adapter")
    parser.add_argument(
        '--adapter-path',
        type=str,
        default='./models/sentiment_adapter_best',
        help='Path to adapter directory'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        help='CSV file with test texts (column: text)'
    )
    
    args = parser.parse_args()
    
    # Default test texts (news/finance domain)
    test_texts = [
        "Stock prices surged to record highs after the company announced exceptional quarterly earnings that beat all analyst expectations.",
        "The market experienced a devastating crash today, with investors losing billions as panic selling swept through Wall Street.",
        "Trading volumes remained steady with no significant price movements as investors awaited the Federal Reserve's policy decision.",
        "Tech giants reported strong revenue growth, sending their shares soaring in after-hours trading.",
        "The company filed for bankruptcy protection after years of declining sales and mounting debts.",
        "Analysts maintained their neutral rating on the stock, citing mixed economic signals and uncertain market conditions.",
        "Breaking: Major acquisition deal announced that will reshape the industry landscape and create significant shareholder value.",
        "Disappointing jobs report sent shockwaves through the market, raising concerns about economic recession.",
        "The cryptocurrency market showed little movement today, with Bitcoin trading in a narrow range around its current levels.",
        "Investors cheered the news of a successful product launch, pushing the company's valuation to new heights."
    ]
    
    # Load custom test texts if provided
    if args.test_file and os.path.exists(args.test_file):
        print(f"Loading test texts from: {args.test_file}")
        df = pd.read_csv(args.test_file)
        if 'text' in df.columns:
            test_texts = df['text'].head(20).tolist()
        else:
            print("⚠️  No 'text' column found, using default texts")
    
    print("=" * 70)
    print("MODEL COMPARISON TEST")
    print("=" * 70)
    print(f"\nTesting {len(test_texts)} examples")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    # Test both models
    base_results = test_base_model(test_texts)
    adapter_results = test_adapter_model(test_texts, args.adapter_path)
    
    # Compare
    compare_results(base_results, adapter_results)
    
    print("\n" + "=" * 70)
    print("✅ Comparison complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
